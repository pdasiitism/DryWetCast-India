"""Download one cycle's NCMRWF (NEPS) files from the NCMRWF data portal.

Portal: https://cloud.ncmrwf.gov.in ("File Manager - NCMRWF Data Sharing
Portal"). Files are uploaded daily around 12:00. This uses the portal's own
API, as its web page does:

  list      GET file_manager.php?action=list&path=<folder>      header X-API-Key
            -> {"success": true, "files": [{"name", "path", "type", "size"}, ...]}
  download  GET file_manager.php?action=download&file=<path>&api_key=<key>

(The portal also offers a "one-click" command that pipes a server-generated
script into bash; that is deliberately not used — it runs whatever the
server sends.)

The API key is read from the NCMRWF_API_KEY environment variable, or from
~/.ncmrwf_api_key (one line; keep it private: chmod 600). It is never
printed; error messages have it stripped out.

One file per cycle: <init_date>.nc (also accepted: <init_date>_NPES.nc), the
23-member 5-day ensemble. It is found by name, searching down from
NCMRWF_REMOTE_DIR (default: the top folder); as of Sep 2026 it is in 2026/
(real time) or 5day/ (2025). Folders holding other NCMRWF products with the
same file names (forecast/, additional_dates_*/) are skipped, and every
download is checked to really be the 5-day ensemble before it is used.
"""

from __future__ import annotations

import os
from collections import deque

import requests

BASE_URL = 'https://cloud.ncmrwf.gov.in/file_manager.php'
KEY_ENV = 'NCMRWF_API_KEY'
KEY_FILE = os.path.expanduser('~/.ncmrwf_api_key')
REMOTE_DIR_ENV = 'NCMRWF_REMOTE_DIR'
TIMEOUT = 120
MAX_SEARCH_DEPTH = 4
MIN_MEMBERS = 10   # same floor as ncmrwf_features


def cycle_names(date: str) -> list[str]:
    """File names NCMRWF uses for an init date's ensemble file.

    The portal uses <date>.nc; files delivered by other routes have come as
    <date>_NPES.nc.
    """
    return [f'{date}.nc', f'{date}_NPES.nc']


def _other_product(folder_name: str) -> bool:
    """Portal folders holding a different NCMRWF product under the same file names."""
    return folder_name == 'forecast' or folder_name.startswith('additional_dates')


def local_name(date: str) -> str:
    """Name the downloaded plain file is saved under (the portal's convention)."""
    return cycle_names(date)[0]


def api_key() -> str | None:
    key = os.environ.get(KEY_ENV, '').strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    return key or None


def _require_key() -> str:
    key = api_key()
    if not key:
        raise RuntimeError(f'No NCMRWF API key. Set {KEY_ENV}, or put the key in {KEY_FILE} '
                           f'(chmod 600) — or pass --ncmrwf-file to use files already on disk.')
    return key


def _scrub(text: str, key: str) -> str:
    return text.replace(key, '<API_KEY>') if key else text


def list_folder(path: str = '', key: str | None = None) -> list[dict]:
    key = key or _require_key()
    try:
        r = requests.get(BASE_URL, params={'action': 'list', 'path': path},
                         headers={'X-API-Key': key}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise RuntimeError(f'NCMRWF portal: listing "{path or "/"}" failed — {_scrub(str(exc), key)}') from None
    if not data.get('success'):
        raise RuntimeError(f'NCMRWF portal: listing "{path or "/"}" refused — {data.get("error", "no reason given")} '
                           '(wrong or expired API key?)')
    return data.get('files', [])


def find_cycle(date: str, key: str | None = None, root: str | None = None) -> tuple[str, dict]:
    """Locate `date`'s 5-day ensemble file on the portal. Returns (folder, entry)."""
    key = key or _require_key()
    root = root if root is not None else os.environ.get(REMOTE_DIR_ENV, '')
    names = cycle_names(date)
    queue = deque([(root, 0)])
    while queue:
        folder, depth = queue.popleft()
        files = {}
        for entry in list_folder(folder, key):
            if entry.get('type') == 'folder':
                if depth + 1 <= MAX_SEARCH_DEPTH and not _other_product(entry.get('name', '')):
                    queue.append((entry['path'], depth + 1))
            else:
                files[entry.get('name', '')] = entry
        for name in names:
            if name in files:
                return folder, files[name]
    raise FileNotFoundError(f'NCMRWF portal: no {date}.nc under "{root or "/"}" — nothing for that date '
                            'yet (NCMRWF uploads around 12:00 daily).')


def _download(entry: dict, out_path: str, key: str) -> None:
    """Stream to <out>.part, check size and that it opens as NetCDF, then rename into place."""
    tmp = out_path + '.part'
    try:
        with requests.get(BASE_URL, params={'action': 'download', 'file': entry['path'], 'api_key': key},
                          headers={'X-API-Key': key}, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f'NCMRWF portal: downloading {entry["name"]} failed — {_scrub(str(exc), key)}') from None

    with open(tmp, 'rb') as f:
        head = f.read(300)
    if not head.startswith((b'\x89HDF', b'CDF')):   # NetCDF-4 / NetCDF-3 signatures: this is an error page
        os.remove(tmp)
        raise RuntimeError(f'NCMRWF portal: {entry["name"]} is not a usable NetCDF file; the server sent: '
                           f'{_scrub(head.decode("utf-8", "replace"), key)!r}')
    size, expected = os.path.getsize(tmp), entry.get('size')
    if expected and int(expected) != size:
        os.remove(tmp)
        raise RuntimeError(f'NCMRWF portal: {entry["name"]} arrived incomplete ({size} of {expected} bytes) — rerun')
    try:
        import xarray as xr
        with xr.open_dataset(tmp) as ds:
            shape = dict(ds['precipitation_amount'].sizes)
    except Exception as exc:
        os.remove(tmp)
        raise RuntimeError(f'NCMRWF portal: {entry["name"]} downloaded but can\'t be read ({exc}) — rerun') from None
    if shape.get('realization', 0) < MIN_MEMBERS or shape.get('forecast_period', 0) < 5:
        os.remove(tmp)
        raise RuntimeError(f'NCMRWF portal: {entry["path"]} is not the 5-day ensemble (dimensions {shape}) — '
                           f'set {REMOTE_DIR_ENV} to the folder that holds it')
    os.replace(tmp, out_path)


def download_cycle(date: str, out_dir: str) -> str:
    """Download the plain file for `date` into out_dir and return its path.

    A file already present (and complete) is not downloaded again.
    """
    existing = local_cycle(date, out_dir)
    if existing:
        return existing

    key = _require_key()
    os.makedirs(out_dir, exist_ok=True)
    folder, entry = find_cycle(date, key)
    print(f'  found in portal folder "{folder or "/"}"', flush=True)
    p = os.path.join(out_dir, local_name(date))
    _download(entry, p, key)
    print(f'  {entry["name"]}: {os.path.getsize(p) / 1e6:.1f} MB', flush=True)
    return p


def local_cycle(date: str, out_dir: str) -> str | None:
    """The plain file's path if this cycle was already downloaded into out_dir, else None."""
    p = os.path.join(out_dir, local_name(date))
    return p if os.path.exists(p) else None
