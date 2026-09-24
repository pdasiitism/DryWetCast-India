"""Forecast record storage (.npz, NumPy only).

The PNG is a visualization, not a record. Each run also writes the actual
DRY/WET probabilities as a compressed .npz: dense (129,135) float32 grids,
NaN over sea, plus lat/lon and the date/cycle/config they belong to. Exact
(no rounding), ~37KB per cycle, loadable with a single np.load — no extra
dependency beyond NumPy.

Archive layout (one file per run): <archive_dir>/<config>/<date>_<cycle>.npz
"""

import glob
import os

import numpy as np

from . import grid


def save(out_path: str, config_name: str, date: str, cycle: str,
         p_dry: np.ndarray, p_wet: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        p_dry=p_dry.astype(np.float32), p_wet=p_wet.astype(np.float32),
        lats=grid.TARGET_LATS, lons=grid.TARGET_LONS,
        date=np.array(date), cycle=np.array(cycle), config=np.array(config_name),
    )


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
