"""Preflight check: can this machine run the pipeline?

Uses only the standard library at import time, so it still works (and says
what's missing) when the scientific packages aren't installed yet.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys

from . import configs, runtime

MIN_PYTHON = (3, 10)   # XGBoost 3.x (which saved the bundled models) and NumPy 2 both need it
PACKAGES = ['numpy', 'rasterio', 'xarray', 'netCDF4', 'scipy', 'xgboost', 'matplotlib', 'requests']
OPTIONAL_PACKAGES = {'zarr': 'only needed for --aifs-zarr (AIFS run locally)'}
FEEDS = {
    'GEFS  (NOAA S3)': 'https://noaa-gefs-pds.s3.amazonaws.com/',
    'AIFS  (ECMWF)  ': 'https://data.ecmwf.int/forecasts/',
}
MIN_FREE_GB = 2   # one cycle's raw GEFS+AIFS download plus regrid cache is ~1.2 GB


def _line(status, msg):
    print(f'  [{status}] {msg}', flush=True)


def run_check(out_dir: str) -> bool:
    """Print a report; return True if nothing blocks a run on this machine."""
    ok = True

    print('Python', flush=True)
    py = sys.version_info
    if py >= MIN_PYTHON:
        _line('OK  ', f'{py.major}.{py.minor}.{py.micro}')
    else:
        ok = False
        _line('FAIL', f'{py.major}.{py.minor} — need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ '
                      '(use the conda environment in environment.yml, which brings its own Python)')

    print('Packages', flush=True)
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            _line('OK  ', f'{name} {getattr(mod, "__version__", "")}')
        except Exception as exc:
            ok = False
            _line('FAIL', f'{name} — {exc.__class__.__name__}: {exc}')
    for name, why in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(name)
            _line('OK  ', f'{name} {getattr(mod, "__version__", "")} (optional)')
        except Exception:
            _line('INFO', f'{name} not installed — {why}; pip install {name}')

    print('Model files', flush=True)
    missing = [p for c in configs.CONFIGS for p in configs.model_paths(c).values() if not os.path.exists(p)]
    for shared in ('land_mask.npy', 'climatology.npz'):
        p = os.path.join(configs.MODELS_ROOT, shared)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        ok = False
        for p in missing:
            _line('FAIL', f'missing {os.path.relpath(p)}')
    else:
        _line('OK  ', f'all {len(configs.CONFIGS)} configs + land mask + climatology present')

    print('Internet access to data feeds', flush=True)
    try:
        import requests
        for label, url in FEEDS.items():
            try:
                requests.head(url, timeout=10)
                _line('OK  ', f'{label} reachable')
            except Exception:
                _line('WARN', f'{label} not reachable — download on a machine with internet using '
                              '--download-only, then run here with --skip-download')
    except ImportError:
        _line('SKIP', 'requests not installed')

    print('NCMRWF data portal (optional — only for ncmrwf_* configs without --ncmrwf-file)', flush=True)
    from . import download_ncmrwf as dn
    key = dn.api_key()
    if key is None:
        _line('INFO', f'no API key set ({dn.KEY_ENV} or {dn.KEY_FILE}) — NCMRWF files must be given with --ncmrwf-file')
    else:
        source = dn.KEY_ENV if os.environ.get(dn.KEY_ENV, '').strip() else dn.KEY_FILE
        if source == dn.KEY_FILE and os.stat(dn.KEY_FILE).st_mode & 0o077:
            _line('WARN', f'{dn.KEY_FILE} is readable by other users — run: chmod 600 {dn.KEY_FILE}')
        try:
            n = len(dn.list_folder('', key))
            _line('OK  ', f'API key (from {source}) accepted by the portal — top folder has {n} entries')
        except Exception as exc:
            _line('WARN', f'API key (from {source}) not working: {exc}')

    print('Resources', flush=True)
    _line('INFO', f'{runtime.available_cpus()} CPUs available to this process -> '
                  f'{runtime.default_workers()} extraction workers (override with --workers)')
    probe = os.path.abspath(out_dir)
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 1e9
    status = 'OK  ' if free_gb >= MIN_FREE_GB else 'WARN'
    _line(status, f'{free_gb:.1f} GB free at {probe} (need ~{MIN_FREE_GB} GB per cycle)')

    print('\nReady to run.' if ok else '\nNot ready — fix the FAIL lines above.', flush=True)
    return ok
