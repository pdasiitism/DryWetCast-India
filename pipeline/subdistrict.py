"""Aggregate gridded DRY/WET probabilities to Indian subdistricts.

The large, static boundary and grid-weight files are deliberately kept out of
the Git repository.  Download the separate ``DryWetCast-Subdistrict-Data``
folder and pass it to the CLI with ``--subdistrict-data-dir``.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    'source_to_index.pkl',
    'target_to_index.pkl',
    'weight_matrix.npy',
    'Sub_districts_India_ESRI.shp',
    'Sub_districts_India_ESRI.shx',
    'Sub_districts_India_ESRI.dbf',
    'Sub_districts_India_ESRI.prj',
    'STATE_BOUNDARY.shp',
    'STATE_BOUNDARY.shx',
    'STATE_BOUNDARY.dbf',
    'STATE_BOUNDARY.prj',
)

TITLES = {
    'dry_probability': 'India Probability of a 3-Day Dry Spell (D1–D5)',
    'wet_probability': 'India Probability of ≥1 Wet Day (D1–D5)',
}


def validate_data_dir(data_dir: str | os.PathLike) -> Path:
    """Return a validated support-data directory."""
    path = Path(data_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Subdistrict data directory does not exist: {path}')
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Subdistrict data directory {path} is missing: {", ".join(missing)}')
    return path


def output_paths(fig_out: str) -> tuple[str, str]:
    """Return ``(<pickle>, <png>)`` beside the ordinary forecast PNG."""
    stem, _ = os.path.splitext(fig_out)
    return f'{stem}_subdistrict.pkl', f'{stem}_subdistrict.png'


def _grid_values(probability: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 source_to_index: dict) -> np.ndarray:
    values = np.full(len(source_to_index), np.nan, dtype=np.float64)
    for source, index in source_to_index.items():
        if not (0 <= index < values.size):
            raise ValueError(f'source_to_index[{source!r}]={index} is out of range')
        lat, lon = map(float, source.split('_'))
        lat_index = int(np.argmin(np.abs(lats - lat)))
        lon_index = int(np.argmin(np.abs(lons - lon)))
        values[index] = probability[lat_index, lon_index]
    return values


def _weighted_average(weight_matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    numerator = weight_matrix @ np.where(valid, values, 0.0)
    denominator = weight_matrix @ valid.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def aggregate(p_dry: np.ndarray, p_wet: np.ndarray,
              lats: np.ndarray, lons: np.ndarray,
              source_to_index: dict, target_to_index: dict,
              weight_matrix: np.ndarray) -> dict:
    """Area-weight the two forecast grids onto every subdistrict polygon."""
    if weight_matrix.shape != (len(target_to_index), len(source_to_index)):
        raise ValueError(
            f'weight_matrix shape {weight_matrix.shape} does not match '
            f'{len(target_to_index)} targets x {len(source_to_index)} sources')

    dry = _weighted_average(
        weight_matrix, _grid_values(np.asarray(p_dry), lats, lons, source_to_index))
    wet = _weighted_average(
        weight_matrix, _grid_values(np.asarray(p_wet), lats, lons, source_to_index))

    result = {}
    for subdistrict_id, index in target_to_index.items():
        if not (0 <= index < len(target_to_index)):
            raise ValueError(f'target_to_index[{subdistrict_id!r}]={index} is out of range')
        result[str(subdistrict_id)] = {
            'dry_probability': float(dry[index]),
            'wet_probability': float(wet[index]),
        }
    return result


def _plot(probabilities: dict, date: str, png_path: str, data_dir: Path) -> None:
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'Subdistrict plotting needs pandas and geopandas; reinstall with '
            '`pip install -r requirements.txt` or update the conda environment.') from exc

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    frame = pd.DataFrame.from_dict(probabilities, orient='index').reset_index()
    frame = frame.rename(columns={'index': 'id'})
    frame['id'] = frame['id'].astype(str)

    subdistricts = gpd.read_file(data_dir / 'Sub_districts_India_ESRI.shp')
    states = gpd.read_file(data_dir / 'STATE_BOUNDARY.shp')
    subdistricts['id'] = subdistricts['id'].astype(str)
    subdistricts = subdistricts.to_crs('EPSG:4326')
    states = states.to_crs('EPSG:4326')
    merged = subdistricts.merge(frame, on='id', how='left')

    issued = datetime.strptime(date, '%Y%m%d')
    valid_start, valid_end = issued + timedelta(days=1), issued + timedelta(days=5)
    valid_text = f'{valid_start:%d%b%Y} - {valid_end:%d%b%Y}'.upper()

    cmap = matplotlib.colormaps['YlGnBu']
    norm = Normalize(vmin=0, vmax=1)
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))
    for column, ax in zip(('dry_probability', 'wet_probability'), axes):
        no_data = merged[merged[column].isna()]
        if not no_data.empty:
            no_data.plot(ax=ax, color='#d3d3d3', linewidth=0.0)
        data = merged[merged[column].notna()].copy()
        if not data.empty:
            data['color'] = data[column].apply(lambda value: cmap(norm(value)))
            data.plot(ax=ax, color=data['color'].tolist(), linewidth=0.1, edgecolor='none')
        merged.boundary.plot(ax=ax, linewidth=0.1, edgecolor='#888888')
        states.boundary.plot(ax=ax, linewidth=1.2, edgecolor='black')
        ax.set_title(f'{TITLES[column]}\nvalid {valid_text}', fontsize=13)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=axes, orientation='vertical',
                            fraction=0.03, pad=0.04)
    colorbar.set_label('probability')
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate(p_dry: np.ndarray, p_wet: np.ndarray,
             lats: np.ndarray, lons: np.ndarray,
             date: str, fig_out: str, data_dir: str | os.PathLike) -> dict:
    """Create the subdistrict probability pickle and map beside ``fig_out``."""
    data_path = validate_data_dir(data_dir)
    with open(data_path / 'source_to_index.pkl', 'rb') as handle:
        source_to_index = pickle.load(handle)
    with open(data_path / 'target_to_index.pkl', 'rb') as handle:
        target_to_index = pickle.load(handle)
    weights = np.load(data_path / 'weight_matrix.npy', mmap_mode='r')

    probabilities = aggregate(
        p_dry, p_wet, np.asarray(lats), np.asarray(lons),
        source_to_index, target_to_index, weights)
    pickle_path, png_path = output_paths(fig_out)
    Path(pickle_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(probabilities, handle)
    _plot(probabilities, date, png_path, data_path)
    return {'pickle': pickle_path, 'png': png_path}
