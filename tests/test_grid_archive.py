"""Bundled grid data, the climatology date rule, and the .npz record round trip."""
import numpy as np
import pytest

from pipeline import archive, grid


def test_bundled_grid():
    land = grid.load_land_mask()
    assert land.shape == (129, 135) and land.dtype == bool and land.any()
    assert len(grid.TARGET_LATS) == 129 and len(grid.TARGET_LONS) == 135


@pytest.mark.parametrize('init, doy, idx', [
    ('20251114', 319, 318),   # valid = init + 1 day
    ('20251231', 1, 0),       # year rollover
    ('20240228', 60, 59),     # leap day is the valid date
    ('20241230', 366, 364),   # doy 366 clipped to the last climatology day, as in training
])
def test_climatology_uses_valid_date(init, doy, idx):
    dry_all, wet_all = grid.load_climatology()
    dry, wet, d = grid.climatology_for_init(init)
    assert d == doy
    np.testing.assert_array_equal(dry, dry_all[idx])
    np.testing.assert_array_equal(wet, wet_all[idx])


def test_archive_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    p_dry, p_wet = rng.random((129, 135)), rng.random((129, 135))
    p_dry[0, 0] = np.nan
    path = archive.archive_path(str(tmp_path), 'gefs_reduced', '20251114', '00')
    archive.save(path, 'gefs_reduced', '20251114', '00', p_dry, p_wet)
    rec = archive.read_archive(str(tmp_path), config='gefs_reduced')[0]
    assert (rec['date'], rec['cycle'], rec['config']) == ('20251114', '00', 'gefs_reduced')
    np.testing.assert_array_equal(rec['p_dry'], p_dry.astype(np.float32))
    np.testing.assert_array_equal(rec['lats'], grid.TARGET_LATS)


def test_aifs_unit_guard():
    from pipeline import aifs_features as af
    rng = np.random.default_rng(2)
    daily_mm = rng.gamma(0.3, 10.0, size=(51, 5, 129, 135)).astype(np.float32)
    af._check_plausible(daily_mm, 'ok')                       # realistic mm: accepted
    with pytest.raises(ValueError, match='metres where mm'):
        af._check_plausible(daily_mm / 1000, 'metres as mm')   # 1000x too small
    with pytest.raises(ValueError, match='implausible'):
        af._check_plausible(daily_mm * 1000, 'mm as metres')   # 1000x too large
