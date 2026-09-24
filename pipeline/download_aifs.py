"""Download the reduced AIFS-ENS v2 predictor set for one forecast cycle from
ECMWF's public open-data feed — no credentials needed.

Uses ECMWF's `.index` sidecar files (JSON-lines, one per GRIB2 message, with
explicit `_offset`/`_length` keyed by `param` and ensemble `number`) to do
targeted HTTP byte-range GETs for just the `tp` (total precipitation)
messages, instead of downloading each ~150-200MB multi-parameter file.

Output layout matches aifs_features.py's expected input exactly:
  <out_dir>/members/<member>/<member>.t<CC>z.f<LLL>.tp.grib2

Verified against the live feed (https://data.ecmwf.int/forecasts) on
2026-09-21: URL pattern
  {base}/<YYYYMMDD>/<CC>z/aifs-ens/0p25/enfo/<YYYYMMDD><CC>0000-<lead>h-enfo-{cf,pf}.grib2
control (cf) = 1 member, no `number` field in its index; perturbed (pf) = one
file per lead containing all 50 members as separate messages, `number` 1-50.

D5's daily window needs lead hour 126 in addition to the native 6-hourly
steps out to 120 — both are included in LEADS below.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = 'https://data.ecmwf.int/forecasts'

LEAD_BASES = [0, 24, 48, 72, 96]   # D1..D5, 03Z-to-03Z bases
LEADS = sorted({base + off for base in LEAD_BASES for off in (6, 12, 18, 24, 30)})   # 6..126, 21 leads

N_WORKERS = 4   # ECMWF's open-data feed rate-limits (429) bursts; see download_cycle
TIMEOUT = 60
MAX_RETRIES = 10
MAX_BACKOFF = 60
CLEANUP_PASSES = 3
COOLDOWN = 60


def _url(date, cycle, lead, kind):
    return f'{BASE_URL}/{date}/{cycle}z/aifs-ens/0p25/enfo/{date}{cycle}0000-{lead}h-enfo-{kind}.grib2'


def _get_with_retry(session, url, **kwargs) -> requests.Response:
    """GET with exponential backoff on 429/5xx — the ECMWF feed rate-limits bursts."""
    for attempt in range(MAX_RETRIES):
        r = session.get(url, timeout=TIMEOUT, **kwargs)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get('Retry-After', 0)) or min(2 ** attempt, MAX_BACKOFF)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def _fetch_range(session, url, offset, length) -> bytes:
    r = _get_with_retry(session, url, headers={'Range': f'bytes={offset}-{offset + length - 1}'})
    return r.content


def _out_path(out_dir, member, cycle, lead):
    return os.path.join(out_dir, 'members', member, f'{member}.t{cycle}z.f{lead:03d}.tp.grib2')


def _download_control(args):
    date, cycle, lead, out_dir, session = args
    out_path = _out_path(out_dir, 'aifs_c00', cycle, lead)
    if os.path.exists(out_path):
        return [out_path]
    url = _url(date, cycle, lead, 'cf')
    idx = _get_with_retry(session, url[:-len('.grib2')] + '.index')
    for line in idx.text.strip().splitlines():
        rec = json.loads(line)
        if rec['param'] == 'tp':
            data = _fetch_range(session, url, rec['_offset'], rec['_length'])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(data)
            return [out_path]
    raise ValueError(f'No tp message in control index for lead {lead}')


def _download_perturbed(args):
    """One lead's pf file holds all 50 members — fetch the index once, then
    range-GET each member's tp message."""
    date, cycle, lead, out_dir, session = args
    url = _url(date, cycle, lead, 'pf')
    idx = _get_with_retry(session, url[:-len('.grib2')] + '.index')

    written = []
    for line in idx.text.strip().splitlines():
        rec = json.loads(line)
        if rec['param'] != 'tp':
            continue
        member = f"aifs_p{int(rec['number']):02d}"
        out_path = _out_path(out_dir, member, cycle, lead)
        if os.path.exists(out_path):
            written.append(out_path)
            continue
        data = _fetch_range(session, url, rec['_offset'], rec['_length'])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        written.append(out_path)

    if len(written) < 50:
        raise ValueError(f'Only found {len(written)}/50 perturbed tp messages for lead {lead}')
    return written


def _run_jobs(jobs, n_workers):
    """Run (kind, args) download jobs; return (files_ok, failed_jobs)."""
    files_ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for kind, job in jobs:
            fn = _download_control if kind == 'cf' else _download_perturbed
            futures[pool.submit(fn, job)] = (kind, job)
        for fut in as_completed(futures):
            kind, job = futures[fut]
            lead = job[2]
            try:
                n = len(fut.result())
                files_ok += n
                print(f'  lead f{lead:03d} [{kind}]: {n} file(s) OK', flush=True)
            except Exception as exc:
                failed.append((kind, job))
                print(f'  lead f{lead:03d} [{kind}]: failed ({exc.__class__.__name__}) — will retry', flush=True)
    return files_ok, failed


def download_cycle(date: str, cycle: str, out_dir: str, n_workers: int = N_WORKERS) -> None:
    """Download everything aifs_features.py needs for one forecast cycle.

    date: 'YYYYMMDD'. cycle: '00'|'06'|'12'|'18'.

    ECMWF's feed throttles bursts, and parallel threads each backing off on
    their own keep it throttled. So after the parallel pass, whatever failed
    is retried one job at a time after a cool-down. Files already on disk are
    skipped, so a retry only fetches what's missing.
    """
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()

    jobs = [('cf', (date, cycle, lead, out_dir, session)) for lead in LEADS]
    jobs += [('pf', (date, cycle, lead, out_dir, session)) for lead in LEADS]

    total_ok, failed = _run_jobs(jobs, n_workers)
    for attempt in range(1, CLEANUP_PASSES + 1):
        if not failed:
            break
        print(f'  {len(failed)} lead(s) throttled/failed — cooling down {COOLDOWN}s, '
              f'then retrying one at a time (pass {attempt}/{CLEANUP_PASSES})', flush=True)
        time.sleep(COOLDOWN)
        ok, failed = _run_jobs(failed, n_workers=1)
        total_ok += ok

    expected = 51 * len(LEADS)
    print(f'AIFS download: {total_ok}/{expected} files OK, {len(failed)} lead/kind jobs failed', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} AIFS lead downloads still failing after {CLEANUP_PASSES} '
                           'retry passes — rerun the same command later to fetch only what is missing')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--date', required=True, help='YYYYMMDD init date')
    p.add_argument('--cycle', default='00', choices=['00', '06', '12', '18'])
    p.add_argument('--out', required=True, help='output cycle root directory')
    args = p.parse_args()
    download_cycle(args.date, args.cycle, args.out)
