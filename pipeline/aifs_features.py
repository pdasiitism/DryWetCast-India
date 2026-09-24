"""AIFS-ENS v2 -> rainfall features for ONE forecast cycle, from either source:

1. Downloaded GRIB2 (download_aifs.py output, ECMWF open data), laid out as
     <cycle_root>/members/<member>/<member>.t<CC>z.f<LLL>.tp.grib2
   `tp` there is cumulative since init, on ECMWF's global 0.25deg grid, so
   consecutive leads are differenced into 6-hourly amounts and regridded.

2. A zarr store written by running AIFS locally on the cluster — the same
   format the training data came from (/net/monsoon/prabal/AIFS_ens_v2/
   init_<YYYYMMDD>T00.zarr): `tp(time, number, prediction_timedelta, lat, lon)`
   in metres, 6-hourly incremental amounts, already on the 129x135 target
   grid. Selected by passing a path ending in `.zarr`. Read exactly as
   extract_aifs_ens_mos_features.py / build_aifs_ens_dry_wet.py did in
   training, but using the store's own coordinates (lat order, lead hours,
   init time) instead of assuming them, and refusing a store that doesn't
   match.

Both sources are turned into the same (members, days, 129, 135) daily-mm
array, and every feature is computed from that by the same code below.

Daily totals: 03Z-to-03Z, half-weighting the first and last 6-hourly period:
  day = 0.5*r[+6h] + r[+12h] + r[+18h] + r[+24h] + 0.5*r[+30h]
D5 therefore needs lead +126h; if a source lacks it, the event-probability
features (which look across all of D1-D5) raise a clear error instead of
silently using wrong values.
"""

import glob
import os
from datetime import datetime, timedelta
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from . import grid, runtime

DRY_MM = 1.0
N_WORKERS = runtime.default_workers()
MIN_MEMBERS = 20

LEAD_BASES = [0, 24, 48, 72, 96]   # D1..D5, 03Z-to-03Z bases

# A real domain-mean daily rainfall is a few mm; anything near this means the
# input's units or accumulation convention differ from what's assumed here.
MAX_PLAUSIBLE_DOMAIN_MEAN_MM = 100.0


def _hours_for(bases):
    return sorted({base + off for base in bases for off in (6, 12, 18, 24, 30)})


def is_zarr(path: str) -> bool:
    return path.rstrip('/').endswith('.zarr')


def _check_plausible(daily, source):
    domain_mean = float(np.nanmean(daily))
    if domain_mean > MAX_PLAUSIBLE_DOMAIN_MEAN_MM:
        raise ValueError(
            f'{source}: mean daily rainfall is {domain_mean:.0f} mm — implausible. The input '
            'probably isn\'t in the expected units/accumulation (zarr: metres, 6-hourly amounts; '
            'GRIB: cumulative since init).')


# ---------------------------------------------------------------- zarr source

@lru_cache(maxsize=2)
def _daily_from_zarr(zarr_path: str, bases: tuple, init_date: str | None) -> np.ndarray:
    """Return (members, len(bases), 129, 135) daily mm from a cluster AIFS zarr store."""
    try:
        import zarr
    except ImportError:
        raise ImportError('Reading a local AIFS zarr store needs the `zarr` package: pip install zarr')

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f'AIFS zarr store not found: {zarr_path}')
    z = zarr.open(zarr_path, mode='r')
    for key in ('tp', 'lat', 'lon', 'prediction_timedelta'):
        if key not in z:
            raise ValueError(f'{zarr_path}: missing `{key}` — not an AIFS store in the expected format')

    if init_date is not None and 'time' in z:
        units = z['time'].attrs.get('units', '')
        if units.startswith('days since '):
            base = datetime.strptime(units[len('days since '):][:10], '%Y-%m-%d')
            store_init = (base + timedelta(days=float(z['time'][0]))).strftime('%Y%m%d')
            if store_init != init_date:
                raise ValueError(f'{zarr_path} is initialised on {store_init}, but --date is {init_date}')

    lat, lon = z['lat'][:], z['lon'][:]
    flip = lat[0] > lat[-1]
    if not (np.allclose(lat[::-1] if flip else lat, grid.TARGET_LATS) and np.allclose(lon, grid.TARGET_LONS)):
        raise ValueError(f'{zarr_path}: lat/lon grid is not the 129x135 target grid '
                         f'(lat {lat[0]}..{lat[-1]}, lon {lon[0]}..{lon[-1]})')

    lead_hours = [int(h) for h in z['prediction_timedelta'][:]]
    h2idx = {h: i for i, h in enumerate(lead_hours)}
    missing = [h for h in _hours_for(bases) if h not in h2idx]
    if missing:
        raise ValueError(f'{zarr_path}: missing lead hours {missing} needed for day(s) '
                         f'{[LEAD_BASES.index(b) + 1 for b in bases]}')

    tp = np.asarray(z['tp'][0], dtype=np.float32)   # (members, leads, lat, lon), metres, incremental
    if flip:
        tp = tp[:, :, ::-1, :]
    if tp.shape[0] < MIN_MEMBERS:
        raise RuntimeError(f'{zarr_path}: only {tp.shape[0]} members (need {MIN_MEMBERS})')

    daily = np.zeros((tp.shape[0], len(bases), grid.N_LAT, grid.N_LON), dtype=np.float32)
    for d, base in enumerate(bases):
        i = [h2idx[base + off] for off in (6, 12, 18, 24, 30)]
        day = (0.5 * tp[:, i[0]] + tp[:, i[1]] + tp[:, i[2]] + tp[:, i[3]] + 0.5 * tp[:, i[4]]) * 1000.0
        np.clip(day, 0.0, None, out=day)
        daily[:, d] = day
    _check_plausible(daily, zarr_path)
    return daily


# ---------------------------------------------------------------- GRIB source

def _detect_cycle_hour(cycle_root: str) -> str:
    files = glob.glob(os.path.join(cycle_root, 'members', 'aifs_c00', 'aifs_c00.t*z.f*.tp.grib2'))
    if not files:
        raise FileNotFoundError(f'No aifs_c00 tp files found under {cycle_root}')
    name = os.path.basename(files[0])
    return name.split('.t')[1].split('z.')[0]


def _members(cycle_root):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(cycle_root, 'members', 'aifs_*')))


def _tp_path(cycle_root, cycle_hour, member, lead):
    return os.path.join(cycle_root, 'members', member, f'{member}.t{cycle_hour}z.f{lead:03d}.tp.grib2')


def _cache_dir(cycle_root):
    return os.path.join(cycle_root, '.regrid_cache')


def _available_hours(cycle_root, cycle_hour, member):
    pat = os.path.join(cycle_root, 'members', member, f'{member}.t{cycle_hour}z.f*.tp.grib2')
    hours = set()
    for p in glob.glob(pat):
        hours.add(int(os.path.basename(p).split('.f')[1].split('.tp')[0]))
    return hours


def _daily_one_member_grib(args):
    """Return (len(bases),129,135) daily mm for one member, or None if a lead file is missing."""
    cycle_root, cycle_hour, member, bases = args
    needed = _hours_for(bases)
    if not set(needed).issubset(_available_hours(cycle_root, cycle_hour, member)):
        return None

    cache_dir = _cache_dir(cycle_root)
    cum = {0: np.zeros((grid.N_LAT, grid.N_LON), dtype=np.float32)}
    for h in needed:
        cum[h] = grid.load_and_regrid_cached(_tp_path(cycle_root, cycle_hour, member, h), cache_dir)

    daily = np.zeros((len(bases), grid.N_LAT, grid.N_LON), dtype=np.float32)
    for d, base in enumerate(bases):
        hrs = [base + off for off in (6, 12, 18, 24, 30)]
        inc = [cum[h] - cum[h - 6] for h in hrs]
        day = 0.5 * inc[0] + inc[1] + inc[2] + inc[3] + 0.5 * inc[4]
        np.clip(day, 0.0, None, out=day)
        daily[d] = day
    return daily


def _daily_from_grib(cycle_root: str, cycle_hour: str | None, bases: tuple) -> np.ndarray:
    """Return (members, len(bases), 129, 135) daily mm from downloaded GRIB2 files."""
    cycle_hour = cycle_hour or _detect_cycle_hour(cycle_root)
    members = _members(cycle_root)
    if 96 in bases:
        ref = 'aifs_c00' if 'aifs_c00' in members else members[0]
        if 126 not in _available_hours(cycle_root, cycle_hour, ref):
            raise FileNotFoundError(
                'Lead hour f126 is missing — required for the D5 daily window that the '
                'dry/wet event-probability features depend on. Download f126 for all members '
                '(same cycle) and rerun.')

    args = [(cycle_root, cycle_hour, m, bases) for m in members]
    with Pool(min(N_WORKERS, len(members))) as pool:
        results = [d for d in pool.map(_daily_one_member_grib, args) if d is not None]
    if len(results) < MIN_MEMBERS:
        raise RuntimeError(f'Only {len(results)}/{len(members)} AIFS members have all needed leads')
    daily = np.stack(results, axis=0)
    _check_plausible(daily, cycle_root)
    return daily


# ---------------------------------------------------------------- features

def _daily(aifs_input: str, cycle_hour: str | None, bases, init_date: str | None) -> np.ndarray:
    bases = tuple(bases)
    if is_zarr(aifs_input):
        return _daily_from_zarr(aifs_input, bases, init_date)
    return _daily_from_grib(aifs_input, cycle_hour, bases)


def compute_aifs_mos(aifs_input: str, cycle_hour: str | None = None, through_day: int = 4,
                     init_date: str | None = None):
    """Return (ens_mean, ens_spread): each (through_day,129,135); index [d] is day d+1.

    `aifs_input` is a downloaded GRIB cycle folder or a local `.zarr` store.
    `through_day` limits which days must be complete — Reduced models only need
    D1-D4 spread + D3 mean, so they run even without the +126h lead.
    `init_date` (YYYYMMDD) is checked against a zarr store's own init time.
    """
    daily = _daily(aifs_input, cycle_hour, LEAD_BASES[:through_day], init_date)
    return np.nanmean(daily, axis=0), np.nanstd(daily, axis=0)


def compute_aifs_event_prob(aifs_input: str, cycle_hour: str | None = None, init_date: str | None = None):
    """Return (prob_dry_event, prob_wet_event): each (129,135).

    DRY: 3-consecutive-dry-day spell starting within D1-D3.
    WET: any day >= 1 mm within D1-D5.
    """
    daily = _daily(aifs_input, cycle_hour, LEAD_BASES, init_date)   # (members, 5, 129, 135)
    dry = daily < DRY_MM
    event_dry = ((dry[:, 0] & dry[:, 1] & dry[:, 2])
                 | (dry[:, 1] & dry[:, 2] & dry[:, 3])
                 | (dry[:, 2] & dry[:, 3] & dry[:, 4]))
    event_wet = (daily >= DRY_MM).any(axis=1)
    return event_dry.astype(np.float32).mean(axis=0), event_wet.astype(np.float32).mean(axis=0)
