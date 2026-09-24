"""Download the GEFS predictor set for one forecast cycle from NOAA's
public S3 bucket — no credentials needed.

Uses the standard NOAA `.idx` sidecar files (wgrib2-style inventories, one
line per GRIB2 message: `n:byte_offset:d=...:VAR:LEVEL:...`) to do targeted
HTTP byte-range GETs, so each downloaded file is just the messages needed
(tens of KB each) instead of the full multi-variable pgrb2a file (~15MB).

Output layout matches gefs_features.py's expected input exactly:
  <out_dir>/apcp_members/<member>/<member>.t<CC>z.pgrb2s.0p25.f<LLL>.apcp.grb2    (0.25deg product)
  <out_dir>/mean_spread/<geavg|gespr>/<product>.t<CC>z.pgrb2a.0p50.f<LLL>.atmos.grb2  (0.5deg product)

Both products match what training used: rainfall from the 0.25deg members,
atmosphere fields from the 0.5deg ensemble mean/spread.

The atmos file bundles all 10 ensemble-mean/spread variables any config
might need (PWAT, CAPE, HGT@300/500/850, RH@850/700, VVEL@850, UGRD/VGRD@850)
in one combined multi-message file — Reduced configs only read 2 of them
(PWAT/CAPE, by GRIB_ELEMENT name — see gefs_features.py), Full configs read
all of them. Fetching the extra messages costs a handful of extra small
byte-range requests, not extra full-file downloads.

Verified against the live bucket (https://noaa-gefs-pds.s3.amazonaws.com) on
2026-09-21: key structure `gefs.<YYYYMMDD>/<CC>/atmos/pgrb2ap5/<file>`.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = 'https://noaa-gefs-pds.s3.amazonaws.com'

MOS_MEMBERS = ['gec00'] + [f'gep{i:02d}' for i in range(1, 31)]   # 31, matches gefs_features.MOS_MEMBERS
_LEAD_BASES = [3, 27, 51, 75, 99]
APCP_LEADS = sorted({lead for b in _LEAD_BASES for lead in [b, b + 3, b + 9, b + 15, b + 21, b + 24]})
ATMOS_LEADS = [27, 30, 36, 42, 48, 51, 54, 60, 66, 72, 75, 78, 84, 90, 96, 99, 102, 108, 114, 120, 123]

# (idx substring, matches gefs_features.py's element= lookups)
ATMOS_VARS = [
    ':PWAT:', ':CAPE:',
    ':HGT:300 mb:', ':HGT:500 mb:', ':HGT:850 mb:',
    ':RH:850 mb:', ':RH:700 mb:',
    ':VVEL:850 mb:', ':UGRD:850 mb:', ':VGRD:850 mb:',
]

N_WORKERS = 16
TIMEOUT = 60


def _apcp_key(date, member, cycle, lead):
    # Rainfall from the 0.25deg product (pgrb2sp25), as training did: its grid
    # points fall exactly on the 129x135 target grid, so nothing is
    # interpolated. The 0.5deg product would need interpolation and gives
    # several-mm differences from the training features.
    return f'gefs.{date}/{cycle}/atmos/pgrb2sp25/{member}.t{cycle}z.pgrb2s.0p25.f{lead:03d}'


def _parse_idx(idx_text: str) -> list[tuple[int, str]]:
    """Return [(byte_offset, inventory_line), ...] sorted by offset."""
    rows = []
    for line in idx_text.strip().splitlines():
        parts = line.split(':')
        if len(parts) < 2:
            continue
        rows.append((int(parts[1]), line))
    return sorted(rows, key=lambda r: r[0])


def _byte_range_for(rows: list[tuple[int, str]], predicate) -> tuple[int, int | None]:
    """Return (start, end) [end=None means to EOF] for the first row matching predicate."""
    for i, (offset, line) in enumerate(rows):
        if predicate(line):
            end = rows[i + 1][0] - 1 if i + 1 < len(rows) else None
            return offset, end
    raise ValueError('No matching message found in .idx')


def _fetch_range(session, url, start, end) -> bytes:
    headers = {'Range': f'bytes={start}-{end}' if end is not None else f'bytes={start}-'}
    r = session.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.content


def _download_one_message(session, url, predicate) -> bytes:
    idx = session.get(url + '.idx', timeout=TIMEOUT)
    idx.raise_for_status()
    rows = _parse_idx(idx.text)
    start, end = _byte_range_for(rows, predicate)
    return _fetch_range(session, url, start, end)


def _download_apcp(args):
    date, cycle, member, lead, out_dir, session = args
    url = f'{BASE_URL}/{_apcp_key(date, member, cycle, lead)}'
    out_path = os.path.join(out_dir, 'apcp_members', member,
                             f'{member}.t{cycle}z.pgrb2s.0p25.f{lead:03d}.apcp.grb2')
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = _download_one_message(session, url, lambda line: ':APCP:' in line)
    with open(out_path, 'wb') as f:
        f.write(data)
    return out_path


def _download_atmos_block(args):
    date, cycle, product, lead, out_dir, session = args
    url = f'{BASE_URL}/gefs.{date}/{cycle}/atmos/pgrb2ap5/{product}.t{cycle}z.pgrb2a.0p50.f{lead:03d}'
    out_path = os.path.join(out_dir, 'mean_spread', product,
                             f'{product}.t{cycle}z.pgrb2a.0p50.f{lead:03d}.atmos.grb2')
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    idx = session.get(url + '.idx', timeout=TIMEOUT)
    idx.raise_for_status()
    rows = _parse_idx(idx.text)

    with open(out_path, 'wb') as f:
        for pattern in ATMOS_VARS:
            byte_range = _byte_range_for(rows, lambda line, p=pattern: p in line)
            f.write(_fetch_range(session, url, *byte_range))
    return out_path


def download_cycle(date: str, cycle: str, out_dir: str, n_workers: int = N_WORKERS) -> None:
    """Download everything gefs_features.py needs for one forecast cycle.

    date: 'YYYYMMDD'. cycle: '00'|'06'|'12'|'18'.
    """
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()

    apcp_jobs = [(date, cycle, m, lead, out_dir, session) for m in MOS_MEMBERS for lead in APCP_LEADS]
    atmos_jobs = [(date, cycle, p, lead, out_dir, session)
                  for p in ('geavg', 'gespr') for lead in ATMOS_LEADS]

    total = len(apcp_jobs) + len(atmos_jobs)
    done, failed = 0, []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_download_apcp, j): ('apcp', j) for j in apcp_jobs}
        futures.update({pool.submit(_download_atmos_block, j): ('atmos', j) for j in atmos_jobs})
        for fut in as_completed(futures):
            kind, job = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as exc:
                failed.append((kind, job, str(exc)))
            if done % 50 == 0:
                print(f'  {done}/{total} downloaded', flush=True)

    print(f'GEFS download: {done}/{total} OK, {len(failed)} failed', flush=True)
    for kind, job, err in failed:
        print(f'  FAILED [{kind}] {job[:4]}: {err}', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} GEFS files failed to download')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--date', required=True, help='YYYYMMDD init date')
    p.add_argument('--cycle', default='00', choices=['00', '06', '12', '18'])
    p.add_argument('--out', required=True, help='output cycle root directory')
    args = p.parse_args()
    download_cycle(args.date, args.cycle, args.out)
