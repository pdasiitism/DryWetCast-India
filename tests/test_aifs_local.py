"""Local AIFS input formats and precipitation-unit handling."""

import numpy as np
import xarray as xr

from pipeline import aifs_features, grid


def test_postprocessed_netcdf_mm_is_not_scaled(tmp_path):
    """Five 1-mm six-hour periods form a 4-mm 03Z-to-03Z daily total."""
    hours = np.array([6, 12, 18, 24, 30], dtype='timedelta64[h]')
    tp = np.ones((1, 20, 5, grid.N_LAT, grid.N_LON), dtype=np.float32)
    ds = xr.Dataset(
        {'tp': (('time', 'number', 'prediction_timedelta', 'lat', 'lon'), tp,
                {'units': 'mm'})},
        coords={
            'time': np.array(['2025-11-14'], dtype='datetime64[ns]'),
            'number': np.arange(20),
            'prediction_timedelta': hours,
            'lat': grid.TARGET_LATS[::-1],
            'lon': grid.TARGET_LONS,
        },
    )
    path = tmp_path / 'init_20251114T00.nc'
    ds.to_netcdf(path)

    daily = aifs_features._daily_from_postprocessed_nc(str(path), (0,), '20251114')

    assert daily.shape == (20, 1, grid.N_LAT, grid.N_LON)
    np.testing.assert_allclose(daily, 4.0)


def test_postprocessed_netcdf_rejects_metres(tmp_path):
    ds = xr.Dataset(
        {'tp': (('time', 'number', 'prediction_timedelta', 'lat', 'lon'),
                np.ones((1, 20, 1, grid.N_LAT, grid.N_LON), dtype=np.float32),
                {'units': 'm'})},
        coords={
            'time': np.array(['2025-11-14'], dtype='datetime64[ns]'),
            'number': np.arange(20),
            'prediction_timedelta': np.array([6], dtype='timedelta64[h]'),
            'lat': grid.TARGET_LATS,
            'lon': grid.TARGET_LONS,
        },
    )
    path = tmp_path / 'wrong_units.nc'
    ds.to_netcdf(path)

    with np.testing.assert_raises_regex(ValueError, 'must contain mm'):
        aifs_features._daily_from_postprocessed_nc(str(path), (0,), '20251114')
