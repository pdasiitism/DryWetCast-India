"""NCMRWF (NEPS) rainfall -> features for ONE forecast cycle.

Reads the operational NEPS NetCDF file: `precipitation_amount(realization,
forecast_period, latitude, longitude)`, each forecast_period slice a 24h
total ending 03Z. One file per cycle:

  <init>.nc  (or <init>_NPES.nc)   23 members

This is the same-day ensemble, as in training (2021-2024: NCMRWF's 11-member
same-day ensemble). NCMRWF's separate lagged-ensemble file (.lag.nc) is not used.

  mean/spread: each member's daily rainfall regridded, then averaged.
  event probabilities: events computed on NCMRWF's native grid, the
    probability field regridded afterwards (order matters with a 1 mm
    threshold); DRY < 1 mm, WET >= 1 mm — as in build_ncmrwf_dry_wet.py.

NCMRWF+AIFS configs use NCMRWF for rainfall only — atmospheric features come
from GEFS in every config (build_feature_matrix_exp10.py reuses Exp6's
GEFS-sourced atmosphere files).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import xarray as xr

from . import grid

DRY_MM = 1.0
MIN_MEMBERS = 10


def _read(path: str, init_date: str | None):
    """Return (rain (members,5,lat,lon), lats, lons), checking the valid times."""
    with xr.open_dataset(path) as ds:
        if init_date is not None:
            valid = datetime.strptime(init_date, '%Y%m%d') + timedelta(days=1)
            expected = np.array([np.datetime64(valid + timedelta(days=i, hours=3)) for i in range(5)],
                                dtype='datetime64[ns]')
            times = ds['time'].values.astype('datetime64[ns]')
            if len(times) < 5 or not np.array_equal(times[:5], expected):
                raise ValueError(f'{path}: forecast days are {times[:5]}, expected 03Z on the 5 days '
                                 f'starting {valid:%Y-%m-%d} — wrong file for --date {init_date}?')
        rain = ds['precipitation_amount'].isel(forecast_period=slice(0, 5)).values.astype(np.float32)
        lats = ds['latitude'].values.astype(np.float64)
        lons = ds['longitude'].values.astype(np.float64)
    if rain.shape[0] < MIN_MEMBERS:
        raise RuntimeError(f'Only {rain.shape[0]} NCMRWF members in {path}')
    return rain, lats, lons


@lru_cache(maxsize=2)
def _mos(nc_path: str, init_date: str | None):
    rain, lats, lons = _read(nc_path, init_date)
    daily = np.zeros((rain.shape[0], 5, grid.N_LAT, grid.N_LON), dtype=np.float32)
    for m in range(rain.shape[0]):
        for d in range(5):
            daily[m, d] = grid.regrid_field(rain[m, d], lats, lons)
    return np.nanmean(daily, axis=0), np.nanstd(daily, axis=0)


def compute_ncmrwf_mos(nc_path: str, init_date: str | None = None):
    """Return (ens_mean, ens_spread): each (5,129,135), from the plain file's 23 members."""
    return _mos(nc_path, init_date)


def compute_ncmrwf_event_prob(nc_path: str, init_date: str | None = None):
    """Return (prob_dry_event, prob_wet_event): each (129,135), from the plain file's members.

    DRY: 3-consecutive-dry-day spell (< 1 mm) starting within D1-D3.
    WET: any day >= 1 mm within D1-D5.
    """
    rain, lats, lons = _read(nc_path, init_date)
    dry = rain < DRY_MM
    event_dry = ((dry[:, 0] & dry[:, 1] & dry[:, 2])
                 | (dry[:, 1] & dry[:, 2] & dry[:, 3])
                 | (dry[:, 2] & dry[:, 3] & dry[:, 4]))
    event_wet = (rain >= DRY_MM).any(axis=1)
    return (grid.regrid_field(event_dry.mean(axis=0), lats, lons),
            grid.regrid_field(event_wet.mean(axis=0), lats, lons))
