"""Raw-GEFS -> feature extraction (Reduced and Full) for ONE forecast cycle.

Mirrors, feature-for-feature, the historical extraction scripts:
  - extract_gefs_mos_features.py       (gefs_mean_D*, gefs_spread_D*)
  - extract_gefs_event_frac_exp6.py    (gefs_prob_dry_event, gefs_prob_wet_event)
  - extract_atmos_future_exp6.py       (all 8 atmospheric roll vars + Z300 WD index —
                                         Reduced configs only use PWAT/CAPE of these)

Input layout expected (matches download_gefs.py's output):
  <cycle_root>/apcp_members/<member>/<member>.t<CC>z.pgrb2s.0p25.f<LLL>.apcp.grb2
  <cycle_root>/mean_spread/<geavg|gespr>/<product>.t<CC>z.pgrb2a.0p50.f<LLL>.atmos.grb2

Raw files are global 0.5deg GRIB2; every field is regridded (bilinear, via
grid.regrid_field) to the 129x135 IMD-aligned target grid. Member-level work
is parallelized with multiprocessing since a forecast cycle needs ~850 GRIB2
decodes (~1-2s each single-threaded).
"""

import glob
import os
from multiprocessing import Pool

import numpy as np

from . import grid, runtime

DRY_MM = 1.0
N_WORKERS = runtime.default_workers()

MOS_MEMBERS   = ['gec00'] + [f'gep{i:02d}' for i in range(1, 31)]   # 31 — matches extract_gefs_mos_features.py
EVENT_MEMBERS = [f'gep{i:02d}' for i in range(1, 31)]               # 30 — matches extract_gefs_event_frac_exp6.py

_LEAD_BASES = [3, 27, 51, 75, 99]
APCP_LEADS = sorted({lead for b in _LEAD_BASES for lead in [b, b + 3, b + 9, b + 15, b + 21, b + 24]})
ATMOS_LEADS = [27, 30, 36, 42, 48, 51, 54, 60, 66, 72, 75, 78, 84, 90, 96, 99, 102, 108, 114, 120, 123]


def _detect_cycle_hour(cycle_root: str) -> str:
    files = glob.glob(os.path.join(cycle_root, 'apcp_members', 'gec00', 'gec00.t*z.pgrb2s.0p25.f*.apcp.grb2'))
    if not files:
        if glob.glob(os.path.join(cycle_root, 'apcp_members', 'gec00', '*.pgrb2a.0p50.*')):
            raise FileNotFoundError(
                f'{cycle_root} holds GEFS rainfall from the old 0.5deg download. The pipeline now uses '
                'the 0.25deg product (what training used) — re-download this cycle (drop --skip-download).')
        raise FileNotFoundError(f'No gec00 APCP files found under {cycle_root}')
    name = os.path.basename(files[0])
    return name.split('.t')[1].split('z.')[0]


def _cache_dir(cycle_root):
    return os.path.join(cycle_root, '.regrid_cache')


def _apcp_path(cycle_root, cycle_hour, member, lead):
    return os.path.join(cycle_root, 'apcp_members', member,
                         f'{member}.t{cycle_hour}z.pgrb2s.0p25.f{lead:03d}.apcp.grb2')


def _atmos_path(cycle_root, cycle_hour, product, lead):
    return os.path.join(cycle_root, 'mean_spread', product,
                         f'{product}.t{cycle_hour}z.pgrb2a.0p50.f{lead:03d}.atmos.grb2')


# Band position within the combined atmos.grb2 file — matches ATMOS_VARS
# order in download_gefs.py exactly. Needed because GRIB_ELEMENT alone can't
# tell HGT@300 from HGT@500/850 (no level in the tag).
ATMOS_BAND = {
    'PWAT': 1, 'CAPE': 2, 'HGT300': 3, 'HGT500': 4, 'HGT850': 5,
    'RH850': 6, 'RH700': 7, 'VVEL850': 8, 'UGRD850': 9, 'VGRD850': 10,
}


def _daily_fields_one_member(args):
    """Return (5,129,135) daily mm for one member, or None if any lead file is missing."""
    cycle_root, cycle_hour, member = args
    cache_dir = _cache_dir(cycle_root)
    fields = {}
    for lead in APCP_LEADS:
        fpath = _apcp_path(cycle_root, cycle_hour, member, lead)
        if not os.path.exists(fpath):
            return None
        fields[lead] = grid.load_and_regrid_cached(fpath, cache_dir)

    daily = np.zeros((5, grid.N_LAT, grid.N_LON), dtype=np.float32)
    for d, base in enumerate(_LEAD_BASES):
        day_total = (
            (fields[base + 3] - fields[base])
            + fields[base + 9] + fields[base + 15]
            + fields[base + 21] + fields[base + 24]
        )
        np.clip(day_total, 0.0, None, out=day_total)
        daily[d] = day_total
    return daily


def _load_all_members(cycle_root, cycle_hour, members):
    args = [(cycle_root, cycle_hour, m) for m in members]
    with Pool(min(N_WORKERS, len(members))) as pool:
        results = pool.map(_daily_fields_one_member, args)
    return [d for d in results if d is not None]


def compute_gefs_mos(cycle_root: str, cycle_hour: str | None = None):
    """Return (ens_mean, ens_spread): each (5,129,135) — matches extract_gefs_mos_features.py."""
    cycle_hour = cycle_hour or _detect_cycle_hour(cycle_root)
    member_fields = _load_all_members(cycle_root, cycle_hour, MOS_MEMBERS)
    if len(member_fields) < 20:
        raise RuntimeError(f'Only {len(member_fields)}/{len(MOS_MEMBERS)} GEFS members available')
    stack = np.stack(member_fields, axis=0)   # (M,5,129,135)
    return stack.mean(axis=0), stack.std(axis=0)


def compute_gefs_event_prob(cycle_root: str, cycle_hour: str | None = None):
    """Return (prob_dry_event, prob_wet_event): each (129,135) — matches extract_gefs_event_frac_exp6.py."""
    cycle_hour = cycle_hour or _detect_cycle_hour(cycle_root)
    member_fields = _load_all_members(cycle_root, cycle_hour, EVENT_MEMBERS)
    if len(member_fields) < 20:
        raise RuntimeError(f'Only {len(member_fields)}/{len(EVENT_MEMBERS)} GEFS members available')

    dry_flags, wet_flags = [], []
    for daily in member_fields:
        dry = daily < DRY_MM
        event_dry = (dry[0] & dry[1] & dry[2]) | (dry[1] & dry[2] & dry[3]) | (dry[2] & dry[3] & dry[4])
        event_wet = (daily > DRY_MM).any(axis=0)
        dry_flags.append(event_dry)
        wet_flags.append(event_wet)
    prob_dry = np.stack(dry_flags).astype(np.float32).mean(axis=0)
    prob_wet = np.stack(wet_flags).astype(np.float32).mean(axis=0)
    return prob_dry, prob_wet


def _extract_regridded_bands(args):
    """Regrid every named band of one atmos.grb2 file. Returns {name: (129,135)}."""
    grib_path, cache_dir, names = args
    return {name: grid.load_and_regrid_cached(grib_path, cache_dir, band=ATMOS_BAND[name])
            for name in names}


def _atmos_paths(cycle_root, cycle_hour):
    avg_paths = [_atmos_path(cycle_root, cycle_hour, 'geavg', lead) for lead in ATMOS_LEADS]
    spr_paths = [_atmos_path(cycle_root, cycle_hour, 'gespr', lead) for lead in ATMOS_LEADS]
    for p in avg_paths + spr_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing GEFS ensemble mean/spread file: {p}')
    return avg_paths, spr_paths


def compute_gefs_pwat_cape(cycle_root: str, cycle_hour: str | None = None):
    """Return (pw_mean, cape_mean, cape_spread): each (129,135).

    Matches extract_atmos_future_exp6.py's "future-mean over D1-D5 leads",
    restricted to PWAT and CAPE (the only two non-rainfall vars the Reduced
    models use).
    """
    cycle_hour = cycle_hour or _detect_cycle_hour(cycle_root)
    cache_dir = _cache_dir(cycle_root)
    avg_paths, spr_paths = _atmos_paths(cycle_root, cycle_hour)

    with Pool(min(N_WORKERS, len(ATMOS_LEADS))) as pool:
        avg_results = pool.map(_extract_regridded_bands, [(p, cache_dir, ['PWAT', 'CAPE']) for p in avg_paths])
        spr_results = pool.map(_extract_regridded_bands, [(p, cache_dir, ['CAPE']) for p in spr_paths])

    pw_mean     = np.mean([r['PWAT'] for r in avg_results], axis=0)
    cape_mean   = np.mean([r['CAPE'] for r in avg_results], axis=0)
    cape_spread = np.mean([r['CAPE'] for r in spr_results], axis=0)
    return pw_mean, cape_mean, cape_spread


def _z300_box_mean_one(args):
    grib_path = args
    return grid.box_mean(grib_path, grid.WD_LAT, grid.WD_LON, band=ATMOS_BAND['HGT300'])


def compute_gefs_atmos_full(cycle_root: str, cycle_hour: str | None = None) -> dict:
    """Return every Full-config atmospheric feature GEFS supplies, as a dict:

      pw_mean_roll, pw_spread_roll, rh850_mean_roll, rh850_spread_roll,
      ws850_mean_roll, ws850_spread_roll, rh700_mean_roll, rh700_spread_roll,
      omega850_mean_roll, omega850_spread_roll, z850_mean_roll, z850_spread_roll,
      z500_mean_roll, z500_spread_roll, cape_mean_roll, cape_spread_roll
        -> each (129,135)
      z300_idx_mean, z300_idx_spread
        -> each a scalar float (Western-Disturbance box area-mean, matching
           extract_atmos_future_exp6.py's `_wd_box_mean`)
    """
    cycle_hour = cycle_hour or _detect_cycle_hour(cycle_root)
    cache_dir = _cache_dir(cycle_root)
    avg_paths, spr_paths = _atmos_paths(cycle_root, cycle_hour)

    spatial_vars = ['PWAT', 'RH850', 'UGRD850', 'VGRD850', 'RH700', 'VVEL850', 'HGT850', 'HGT500', 'CAPE']

    with Pool(min(N_WORKERS, len(ATMOS_LEADS))) as pool:
        avg_results = pool.map(_extract_regridded_bands, [(p, cache_dir, spatial_vars) for p in avg_paths])
        spr_results = pool.map(_extract_regridded_bands, [(p, cache_dir, spatial_vars) for p in spr_paths])
        z300_avg = pool.map(_z300_box_mean_one, avg_paths)
        z300_spr = pool.map(_z300_box_mean_one, spr_paths)

    def future_mean(results, name):
        return np.mean([r[name] for r in results], axis=0)

    pw_mean, pw_spread     = future_mean(avg_results, 'PWAT'), future_mean(spr_results, 'PWAT')
    rh850_mean, rh850_sprd = future_mean(avg_results, 'RH850'), future_mean(spr_results, 'RH850')
    rh700_mean, rh700_sprd = future_mean(avg_results, 'RH700'), future_mean(spr_results, 'RH700')
    om850_mean, om850_sprd = future_mean(avg_results, 'VVEL850'), future_mean(spr_results, 'VVEL850')
    z850_mean, z850_sprd   = future_mean(avg_results, 'HGT850'), future_mean(spr_results, 'HGT850')
    z500_mean, z500_sprd   = future_mean(avg_results, 'HGT500'), future_mean(spr_results, 'HGT500')
    cape_mean, cape_sprd   = future_mean(avg_results, 'CAPE'), future_mean(spr_results, 'CAPE')

    u850_mean, u850_sprd = future_mean(avg_results, 'UGRD850'), future_mean(spr_results, 'UGRD850')
    v850_mean, v850_sprd = future_mean(avg_results, 'VGRD850'), future_mean(spr_results, 'VGRD850')
    ws850_mean = np.sqrt(u850_mean ** 2 + v850_mean ** 2)
    ws850_sprd = np.sqrt(u850_sprd ** 2 + v850_sprd ** 2)

    return {
        'pw_mean_roll': pw_mean, 'pw_spread_roll': pw_spread,
        'rh850_mean_roll': rh850_mean, 'rh850_spread_roll': rh850_sprd,
        'ws850_mean_roll': ws850_mean, 'ws850_spread_roll': ws850_sprd,
        'rh700_mean_roll': rh700_mean, 'rh700_spread_roll': rh700_sprd,
        'omega850_mean_roll': om850_mean, 'omega850_spread_roll': om850_sprd,
        'z850_mean_roll': z850_mean, 'z850_spread_roll': z850_sprd,
        'z500_mean_roll': z500_mean, 'z500_spread_roll': z500_sprd,
        'cape_mean_roll': cape_mean, 'cape_spread_roll': cape_sprd,
        'z300_idx_mean': float(np.mean(z300_avg)),
        'z300_idx_spread': float(np.mean(z300_spr)),
    }
