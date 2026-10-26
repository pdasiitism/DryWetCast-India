"""Forecast record storage (.npz, NumPy only).

The PNG is a visualization, not a record. Each run also writes the actual
post-processed and raw-ensemble DRY/WET probabilities as a compressed .npz:
dense (129,135) float32 grids, plus lat/lon and the date/cycle/config they
belong to. Exact (no rounding), loadable with a single np.load — no extra
dependency beyond NumPy.

Archive layout (one file per run): <archive_dir>/<config>/<date>_<cycle>.npz
"""

import glob
import os

import numpy as np

from . import grid


def save(out_path: str, config_name: str, date: str, cycle: str,
         p_dry: np.ndarray, p_wet: np.ndarray,
         raw_p_dry: np.ndarray | None = None,
         raw_p_wet: np.ndarray | None = None) -> None:
    if (raw_p_dry is None) != (raw_p_wet is None):
        raise ValueError('raw_p_dry and raw_p_wet must be supplied together')
    expected_shape = (grid.N_LAT, grid.N_LON)
    for name, value in (('p_dry', p_dry), ('p_wet', p_wet)):
        if np.shape(value) != expected_shape:
            raise ValueError(f'{name} has shape {np.shape(value)}, expected {expected_shape}')
    if raw_p_dry is not None:
        for name, value in (('raw_p_dry', raw_p_dry), ('raw_p_wet', raw_p_wet)):
            if np.shape(value) != expected_shape:
                raise ValueError(f'{name} has shape {np.shape(value)}, expected {expected_shape}')
    fields = {
        'p_dry': p_dry.astype(np.float32),
        'p_wet': p_wet.astype(np.float32),
        'lats': grid.TARGET_LATS,
        'lons': grid.TARGET_LONS,
        'date': np.array(date),
        'cycle': np.array(cycle),
        'config': np.array(config_name),
    }
    if raw_p_dry is not None:
        fields['raw_p_dry'] = np.asarray(raw_p_dry, dtype=np.float32)
        fields['raw_p_wet'] = np.asarray(raw_p_wet, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, **fields)


def archive_path(archive_dir: str, config_name: str, date: str, cycle: str) -> str:
    return os.path.join(archive_dir, config_name, f'{date}_{cycle}.npz')


def load(path: str) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    for k in ('date', 'cycle', 'config'):
        out[k] = str(out[k])
    return out


def read_archive(archive_dir: str, config: str | None = None, date: str | None = None) -> list[dict]:
    """Load every archived run (optionally filtered by config and/or date), sorted by path."""
    pattern = os.path.join(archive_dir, config or '*', f'{date or "*"}_*.npz')
    return [load(p) for p in sorted(glob.glob(pattern))]
