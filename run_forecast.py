#!/usr/bin/env python3
"""Run the full operational DRY/WET forecast pipeline for one cycle.

Self-contained: downloads GEFS/AIFS input data from public feeds (no
credentials, no cluster/NFS paths), runs the bundled XGBoost weights for the
chosen config, and writes out the DRY/WET probability maps. The only inputs
that ship with the repo are the trained model weights (models/<config>/),
the land mask, and (Full configs only) a bundled climatology file — all
small, static files.

Configs (pipeline/configs.py):
  gefs_reduced    - GEFS rainfall + GEFS atmosphere + AIFS, 8-9 LASSO features
  ncmrwf_reduced  - NCMRWF(NEPS) rainfall + GEFS atmosphere + AIFS, 8-9 LASSO features
  gefs_full       - GEFS rainfall + GEFS atmosphere + AIFS, all 42 features
  ncmrwf_full     - NCMRWF(NEPS) rainfall + GEFS atmosphere + AIFS, all 42 features

NCMRWF comes either from the NCMRWF data portal (downloaded automatically;
needs your API key in NCMRWF_API_KEY or ~/.ncmrwf_api_key), or from files
already on this machine via --ncmrwf-file <init_date>.nc. Only that plain
file (23 members) is used; the lagged-ensemble file (.lag.nc) is not, because
the models were trained on the same-day ensemble without lagged members.

Usage:
    python run_forecast.py --check                      # is this machine set up correctly?
    python run_forecast.py --config gefs_reduced --date 20260920 --cycle 00
    python run_forecast.py --config ncmrwf_reduced --date 20260920            # NCMRWF from the portal
    python run_forecast.py --config ncmrwf_reduced --date 20260920 \\
        --ncmrwf-file /path/to/20260920.nc                                     # NCMRWF already on disk

    # Cluster whose compute nodes have no internet: download on the login
    # node, then run the processing as a batch job.
    python run_forecast.py --config gefs_reduced --date 20260920 --download-only
    python run_forecast.py --config gefs_reduced --date 20260920 --skip-download

    # AIFS run locally on the cluster: read it from disk, download only GEFS.
    python run_forecast.py --config gefs_reduced --date 20260920 \\
        --aifs-zarr /path/to/init_20260920T00.zarr

Requires Python 3.10+ and: numpy, rasterio, xarray, netCDF4, scipy, xgboost,
matplotlib, requests (`pip install -r requirements.txt`, or
`conda env create -f environment.yml`). No system GRIB library needed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from pipeline import check, configs

if sys.version_info < check.MIN_PYTHON:
    sys.exit(f'Python {check.MIN_PYTHON[0]}.{check.MIN_PYTHON[1]}+ required (this is '
             f'{sys.version_info.major}.{sys.version_info.minor}). The conda environment in '
             'environment.yml brings its own Python: conda env create -f environment.yml')


def _init_date(value: str) -> str:
    try:
        datetime.strptime(value, '%Y%m%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a valid YYYYMMDD date (e.g. 20251114)')
    return value


def ncmrwf_dir(out_dir: str, date: str) -> str:
    return os.path.join(out_dir, f'ncmrwf_{date}')


def download(date: str, cycle: str, out_dir: str, aifs_zarr: str | None = None,
             ncmrwf: bool = False) -> None:
    from pipeline import download_gefs, download_aifs, download_ncmrwf

    if ncmrwf:   # first: it's the one most likely not to be there yet, so fail before the long downloads
        print(f'=== Downloading NCMRWF ({date}) from the NCMRWF portal ===', flush=True)
        t0 = time.time()
        download_ncmrwf.download_cycle(date, ncmrwf_dir(out_dir, date))
        print(f'  done in {time.time()-t0:.0f}s', flush=True)

    print(f'=== Downloading GEFS ({date} {cycle}Z) ===', flush=True)
    t0 = time.time()
    download_gefs.download_cycle(date, cycle, os.path.join(out_dir, f'gefs_{date}_{cycle}z'))
    print(f'  done in {time.time()-t0:.0f}s', flush=True)

    if aifs_zarr:
        print(f'=== AIFS v2: using local store {aifs_zarr} — nothing to download ===', flush=True)
        return
    print(f'=== Downloading AIFS v2 ({date} {cycle}Z) ===', flush=True)
    t0 = time.time()
    download_aifs.download_cycle(date, cycle, os.path.join(out_dir, f'aifs_{date}_{cycle}z'))
    print(f'  done in {time.time()-t0:.0f}s', flush=True)


def run(config_name: str, date: str, cycle: str, out_dir: str, fig_out: str,
        skip_download: bool = False, ncmrwf_file: str | None = None,
        archive_dir: str | None = None, aifs_zarr: str | None = None) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from pipeline import grid, gefs_features, aifs_features, ncmrwf_features, inference, archive

    cfg = configs.CONFIGS[config_name]
    source, feature_set = cfg['source'], cfg['feature_set']
    is_full = feature_set == 'full'

    gefs_root = os.path.join(out_dir, f'gefs_{date}_{cycle}z')
    aifs_input = aifs_zarr or os.path.join(out_dir, f'aifs_{date}_{cycle}z')

    fetch_ncmrwf = False
    if source == 'ncmrwf':
        if ncmrwf_file:   # file already on this machine
            if not os.path.exists(ncmrwf_file):   # fail before downloading
                raise FileNotFoundError(f'--ncmrwf-file {ncmrwf_file} does not exist')
        elif skip_download:   # downloaded earlier, e.g. with --download-only
            from pipeline import download_ncmrwf
            local = download_ncmrwf.local_cycle(date, ncmrwf_dir(out_dir, date))
            if local is None:
                raise FileNotFoundError(f'No NCMRWF file for {date} in {ncmrwf_dir(out_dir, date)} — download '
                                        'them first (--download-only) or pass --ncmrwf-file.')
            ncmrwf_file = local
        else:   # fetch from the NCMRWF portal
            fetch_ncmrwf = True

    if not skip_download:
        download(date, cycle, out_dir, aifs_zarr, ncmrwf=fetch_ncmrwf)

    if fetch_ncmrwf:
        from pipeline import download_ncmrwf
        ncmrwf_file = download_ncmrwf.local_cycle(date, ncmrwf_dir(out_dir, date))
    if source == 'ncmrwf':
        print(f'=== NCMRWF: {ncmrwf_file} ===', flush=True)

    print(f'=== Extracting {source.upper()} rainfall features ===', flush=True)
    t0 = time.time()
    if source == 'gefs':
        rain_mean, rain_spread = gefs_features.compute_gefs_mos(gefs_root, cycle)
        rain_prob_dry, rain_prob_wet = gefs_features.compute_gefs_event_prob(gefs_root, cycle)
    else:
        rain_mean, rain_spread = ncmrwf_features.compute_ncmrwf_mos(ncmrwf_file, init_date=date)
        rain_prob_dry, rain_prob_wet = ncmrwf_features.compute_ncmrwf_event_prob(
            ncmrwf_file, init_date=date)
    print(f'  done in {time.time()-t0:.0f}s', flush=True)

    print(f'=== Extracting AIFS features ({"local zarr" if aifs_zarr else "downloaded GRIB"}) ===', flush=True)
    t0 = time.time()
    through_day = 5 if is_full else 4
    aifs_mean, aifs_spread = aifs_features.compute_aifs_mos(
        aifs_input, cycle, through_day=through_day, init_date=date)
    aifs_prob_dry, aifs_prob_wet = aifs_features.compute_aifs_event_prob(aifs_input, cycle, init_date=date)
    print(f'  done in {time.time()-t0:.0f}s', flush=True)

    land_mask = grid.load_land_mask()

    if is_full:
        print('=== Extracting full GEFS atmosphere (8 vars + Z300 index) ===', flush=True)
        t0 = time.time()
        atmos_full = gefs_features.compute_gefs_atmos_full(gefs_root, cycle)
        print(f'  done in {time.time()-t0:.0f}s', flush=True)

        logit_pi_dry_today, logit_pi_wet_today, doy = grid.climatology_for_init(date)

        print('=== Running inference (Full) ===', flush=True)
        feat_dict = inference.build_feature_dict_full(
            source=source,
            rain_mos=(rain_mean, rain_spread),
            rain_event_prob=(rain_prob_dry, rain_prob_wet),
            aifs_mos=(aifs_mean, aifs_spread),
            aifs_event_prob=(aifs_prob_dry, aifs_prob_wet),
            gefs_atmos_full=atmos_full,
            logit_pi_dry_today=logit_pi_dry_today,
            logit_pi_wet_today=logit_pi_wet_today,
            doy=doy,
        )
    else:
        print('=== Extracting GEFS atmosphere (PWAT/CAPE — used by every Reduced config) ===', flush=True)
        t0 = time.time()
        pw_mean, cape_mean, cape_spread = gefs_features.compute_gefs_pwat_cape(gefs_root, cycle)
        print(f'  done in {time.time()-t0:.0f}s', flush=True)

        print('=== Running inference (Reduced) ===', flush=True)
        feat_dict = inference.build_feature_dict(
            source=source,
            rain_mos=(rain_mean, rain_spread),
            rain_event_prob=(rain_prob_dry, rain_prob_wet),
            gefs_pwat_cape=(pw_mean, cape_mean, cape_spread),
            aifs_mos=(aifs_mean, aifs_spread),
            aifs_event_prob=(aifs_prob_dry, aifs_prob_wet),
        )

    paths = configs.model_paths(config_name)
    p_dry = inference.predict_xgb(paths['dry_json'], paths['dry_mask'], feat_dict, land_mask)
    p_wet = inference.predict_xgb(paths['wet_json'], paths['wet_mask'], feat_dict, land_mask)

    print(f'=== Saving forecast product -> {fig_out} ===', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(fig_out)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    panels = [(p_dry, f'DRY probability — XGB {feature_set} ({config_name})\nissued {date} {cycle}Z'),
              (p_wet, f'WET probability — XGB {feature_set} ({config_name})\nissued {date} {cycle}Z')]
    for ax, (field, title) in zip(axes, panels):
        im = ax.pcolormesh(grid.TARGET_LONS, grid.TARGET_LATS, field, cmap='viridis', vmin=0, vmax=1, shading='auto')
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.04, label='probability')
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200, bbox_inches='tight')
    plt.close(fig)

    npz_out = os.path.splitext(fig_out)[0] + '.npz'
    archive.save(npz_out, config_name, date, cycle, p_dry, p_wet)
    print(f'=== Saved forecast record -> {npz_out} ===', flush=True)

    if archive_dir:
        archived_path = archive.archive_path(archive_dir, config_name, date, cycle)
        archive.save(archived_path, config_name, date, cycle, p_dry, p_wet)
        print(f'=== Added to archive -> {archived_path} ===', flush=True)

    return {'p_dry': p_dry, 'p_wet': p_wet}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true',
                    help='check this machine is set up to run the pipeline, then exit')
    p.add_argument('--config', choices=sorted(configs.CONFIGS))
    p.add_argument('--date', type=_init_date, help='YYYYMMDD init date (the day the model run starts, 00Z)')
    p.add_argument('--cycle', default='00', choices=['00', '06', '12', '18'])
    p.add_argument('--out-dir', default='./_scratch', help='where to download/cache raw GEFS+AIFS data')
    p.add_argument('--fig-out', default='./forecast_product_dry_wet.png')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true',
                       help='only download the raw data (e.g. on a cluster login node with internet), then exit')
    mode.add_argument('--skip-download', action='store_true',
                       help='reuse data already downloaded under --out-dir (e.g. on a compute node without internet)')
    p.add_argument('--workers', type=int, default=None,
                    help='feature-extraction processes (default: CPUs available to this job, max 16)')
    p.add_argument('--ncmrwf-file', default=None,
                    help='NCMRWF plain file <init_date>.nc already on this machine (ncmrwf_* configs; default: download from the portal)')
    p.add_argument('--aifs-zarr', default=None,
                    help='AIFS run locally on this cluster: path to its init_<YYYYMMDD>T00.zarr store. '
                         'AIFS is then read from disk instead of downloaded')
    p.add_argument('--archive-dir', default=None,
                    help='also save this run into an archive folder '
                         '(archive_dir/<config>/<date>_<cycle>.npz) for accumulating many cycles')
    args = p.parse_args()

    if args.check:
        sys.exit(0 if check.run_check(args.out_dir) else 1)

    if args.date is None or (args.config is None and not args.download_only):
        p.error('--date is required, and --config too unless using --download-only')

    if args.aifs_zarr is not None:
        if not args.aifs_zarr.rstrip('/').endswith('.zarr'):
            p.error(f'--aifs-zarr must be a .zarr store (got {args.aifs_zarr})')
        if not os.path.isdir(args.aifs_zarr):
            p.error(f'--aifs-zarr: {args.aifs_zarr} does not exist')

    if args.download_only:
        from pipeline import download_ncmrwf
        if args.ncmrwf_file:
            want_ncmrwf = False
        elif args.config:
            want_ncmrwf = configs.CONFIGS[args.config]['source'] == 'ncmrwf'
        else:   # no config: fetch everything, so one download serves all four configs
            want_ncmrwf = download_ncmrwf.api_key() is not None
            if not want_ncmrwf:
                print(f'=== NCMRWF: no API key set ({download_ncmrwf.KEY_ENV} or {download_ncmrwf.KEY_FILE}) '
                      '— skipping; ncmrwf_* configs will need --ncmrwf-file ===', flush=True)
        download(args.date, args.cycle, args.out_dir, args.aifs_zarr, ncmrwf=want_ncmrwf)
        sys.exit(0)

    if args.workers is not None:
        from pipeline import gefs_features, aifs_features
        gefs_features.N_WORKERS = aifs_features.N_WORKERS = max(1, args.workers)

    run(args.config, args.date, args.cycle, args.out_dir, args.fig_out,
        args.skip_download, args.ncmrwf_file, args.archive_dir, args.aifs_zarr)
