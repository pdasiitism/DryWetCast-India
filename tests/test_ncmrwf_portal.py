"""NCMRWF portal downloader against a local stand-in for the portal API.

Only <date>.nc is downloaded — never the .lag.nc, never the same-named files in forecast/.

The stand-in implements what the portal's own web page uses:
  GET file_manager.php?action=list&path=P       (header X-API-Key) -> JSON listing
  GET file_manager.php?action=download&file=F&api_key=K            -> file bytes
"""
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import xarray as xr

from pipeline import download_ncmrwf as dn

KEY = 'SECRET-TEST-KEY-123'


@pytest.fixture(scope='module')
def portal(tmp_path_factory):
    src = tmp_path_factory.mktemp('src')
    def nc(n_members):
        path = src / f'{n_members}.nc'
        dims = ('realization', 'forecast_period', 'latitude', 'longitude')
        xr.Dataset({'precipitation_amount': (dims, np.zeros((n_members, 5, 3, 4), np.float32))}
                   ).to_netcdf(path, engine='netcdf4')
        return path.read_bytes()
    data, single = nc(23), nc(1)
    files = {   # portal path -> bytes served
        'forecast/20251114.nc': single,          # other product, same name, found first: must be skipped
        'additional_dates_20260407/20251114.nc': single,
        '2025/nested/20251114.nc': data,
        '2025/nested/20251114.lag.nc': b'lag file: must never be downloaded',
        '5day/20251110_NPES.nc': data,           # alternative naming
        'OTHER/20251113.nc': single,             # same name, but not the 5-day ensemble
        'BAD/20251120.nc': data[:len(data) // 2],
        'HTML/20251121.nc': b'<html>Session expired</html>',
    }
    size = {p: len(b) for p, b in files.items()}
    size.update({p: len(data) for p in files if p.startswith(('BAD', 'HTML'))})   # listing advertises full size

    def listing(path):
        path, out, seen = path.strip('/'), [], set()
        for p in files:
            if path and not p.startswith(path + '/'):
                continue
            rest = p[len(path) + 1:] if path else p
            head = rest.split('/')[0]
            full = f'{path}/{head}' if path else head
            if full not in seen:
                seen.add(full)
                folder = '/' in rest
                out.append({'name': head, 'path': full, 'type': 'folder' if folder else 'file',
                            'size': 0 if folder else size[p]})
        return out

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}
            if q.get('action') == 'list':
                ok = self.headers.get('X-API-Key') == KEY
                body = json.dumps({'success': True, 'files': listing(q.get('path', ''))} if ok
                                  else {'success': False, 'error': 'Invalid API key'}).encode()
            elif q.get('api_key') == KEY:
                body = files[q['file']]
            else:
                self.send_response(403); self.end_headers(); return
            self.send_response(200); self.end_headers(); self.wfile.write(body)

    srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{srv.server_address[1]}/file_manager.php', data
    srv.shutdown()


@pytest.fixture
def env(portal, monkeypatch, tmp_path):
    url, data = portal
    monkeypatch.setattr(dn, 'BASE_URL', url)
    monkeypatch.setattr(dn, 'KEY_FILE', str(tmp_path / 'no_key_file'))   # never read the real key
    monkeypatch.delenv(dn.REMOTE_DIR_ENV, raising=False)
    monkeypatch.setenv(dn.KEY_ENV, KEY)
    return data


@pytest.mark.parametrize('date', ['20251114', '20251110'])
def test_download_and_reuse(env, tmp_path, monkeypatch, date):
    plain = dn.download_cycle(date, str(tmp_path / 'out'))
    assert open(plain, 'rb').read() == env
    assert os.listdir(tmp_path / 'out') == [f'{date}.nc'], 'only the ensemble file — never the lag file'
    monkeypatch.setattr(dn, 'BASE_URL', 'http://127.0.0.1:1/unreachable')   # rerun must not need the portal
    assert dn.download_cycle(date, str(tmp_path / 'out')) == plain


@pytest.mark.parametrize('date, key, message', [
    ('20251114', 'WRONG', 'wrong or expired API key'),
    ('20251125', KEY, 'uploads around 12:00'),
    ('20251113', KEY, 'not the 5-day ensemble'),
    ('20251120', KEY, 'incomplete'),
    ('20251121', KEY, 'not a usable NetCDF'),
])
def test_failures_are_clear_and_leave_nothing(env, tmp_path, monkeypatch, date, key, message):
    monkeypatch.setenv(dn.KEY_ENV, key)
    out = tmp_path / 'out'
    with pytest.raises((RuntimeError, FileNotFoundError)) as e:
        dn.download_cycle(date, str(out))
    assert message in str(e.value)
    assert KEY not in str(e.value)
    assert not out.exists() or not os.listdir(out), 'failed download left files behind'


def test_no_key(env, tmp_path, monkeypatch):
    monkeypatch.delenv(dn.KEY_ENV)
    with pytest.raises(RuntimeError, match='--ncmrwf-file'):
        dn.download_cycle('20251114', str(tmp_path / 'out'))
