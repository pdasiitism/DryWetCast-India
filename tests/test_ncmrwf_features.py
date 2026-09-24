"""NCMRWF features come from the plain file only, with training's thresholds and date check."""
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from pipeline import ncmrwf_features as nf

INIT = '20251114'


def write_plain(path, rain_by_day, init=INIT, n_members=23):
    """Synthetic NCMRWF plain file: same rain (mm) at every point, per forecast day."""
    lats, lons = np.arange(0.0, 45.1, 1.0), np.arange(55.0, 110.1, 1.0)
    valid = datetime.strptime(init, '%Y%m%d') + timedelta(days=1)
    rain = np.broadcast_to(np.asarray(rain_by_day, np.float32)[None, :, None, None],
                           (n_members, 5, len(lats), len(lons))).copy()
    xr.Dataset(
        {'precipitation_amount': (('realization', 'forecast_period', 'latitude', 'longitude'), rain)},
        coords={'realization': np.arange(n_members), 'latitude': lats, 'longitude': lons,
                'time': ('forecast_period', [valid + timedelta(days=i, hours=3) for i in range(5)])},
    ).to_netcdf(path)
    nf._mos.cache_clear()
    return str(path)


@pytest.mark.parametrize('rain, p_dry, p_wet', [
    ([0, 0, 0, 5, 5], 1.0, 1.0),       # dry spell D1-D3, then wet
    ([5, 0.99, 0.99, 0.99, 5], 1.0, 1.0),   # dry spell D2-D4 (< 1 mm is dry)
    ([1, 1, 1, 1, 1], 0.0, 1.0),       # exactly 1 mm is wet, never dry
    ([0, 0, 5, 0, 0], 0.0, 1.0),       # no 3-consecutive-day spell; wet on D3
])
def test_event_probability(tmp_path, rain, p_dry, p_wet):
    path = write_plain(tmp_path / f'{INIT}.nc', rain)
    dry, wet = nf.compute_ncmrwf_event_prob(path, init_date=INIT)
    assert dry.shape == wet.shape == (129, 135)
    np.testing.assert_allclose(dry, p_dry)
    np.testing.assert_allclose(wet, p_wet)


def test_lag_file_is_ignored(tmp_path):
    plain = write_plain(tmp_path / f'{INIT}.nc', [0, 0, 0, 0, 0])
    write_plain(tmp_path / f'{INIT}.lag.nc', [9, 9, 9, 9, 9], n_members=22)   # would change both probs
    dry, wet = nf.compute_ncmrwf_event_prob(plain, init_date=INIT)
    assert np.all(dry == 1.0) and np.all(wet == 0.0)


def test_mean_spread(tmp_path):
    path = write_plain(tmp_path / f'{INIT}.nc', [1, 2, 3, 4, 5])
    mean, spread = nf.compute_ncmrwf_mos(path, init_date=INIT)
    assert mean.shape == (5, 129, 135)
    np.testing.assert_allclose(mean[:, 60, 60], [1, 2, 3, 4, 5], rtol=1e-6)
    np.testing.assert_allclose(spread, 0, atol=1e-6)


def test_wrong_date_rejected(tmp_path):
    path = write_plain(tmp_path / 'x.nc', [0] * 5, init='20251113')
    with pytest.raises(ValueError, match='wrong file for --date'):
        nf.compute_ncmrwf_event_prob(path, init_date=INIT)
