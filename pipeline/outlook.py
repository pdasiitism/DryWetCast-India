"""Fixed-category DRY/WET weather-outlook map.

The forecast probabilities are unchanged; this module only presents them in
fixed probability bands so that maps from different cycles can be compared
directly.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np


EXTENT = (66.5, 100.0, 6.5, 38.5)
BOUNDS = [0, 20, 40, 60, 80, 90, 100.001]
LABELS = ['<20%', '20–40%', '40–60%', '60–80%', '80–90%', '≥90%']
DRY_COLORS = ['#FFF9E8', '#FCE7A2', '#F9C65A', '#F08A3C', '#C94B32', '#7F1D1D']
WET_COLORS = ['#F1FAFE', '#CDECF7', '#91D0E8', '#4B9DCE', '#2166AC', '#08306B']


def _valid_span(first: datetime, last: datetime) -> str:
    """Format a compact D1–D5 date span for the map heading."""
    def part(value, month=True, year=True):
        return (f'{value:%d}' +
                (f'{value:%b}'.upper() if month else '') +
                (f'{value:%Y}' if year else ''))

    if first.month == last.month and first.year == last.year:
        return f'{part(first, False, False)}-{part(last)}'
    if first.year == last.year:
        return f'{part(first, year=False)}-{part(last)}'
    return f'{part(first)}-{part(last)}'


def output_path(fig_out: str) -> str:
    """Return the outlook PNG path beside the ordinary forecast PNG."""
    stem, _ = os.path.splitext(fig_out)
    return f'{stem}_outlook.png'


def render(p_dry, p_wet, date: str, cycle: str, out_path: str,
           lats=None, lons=None) -> str:
    """Draw the two-panel outlook and return ``out_path``."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    if lats is None or lons is None:
        from . import grid
        lats = grid.TARGET_LATS if lats is None else lats
        lons = grid.TARGET_LONS if lons is None else lons
    lats, lons = np.asarray(lats), np.asarray(lons)
    init = datetime.strptime(date, '%Y%m%d')
    first, last = init + timedelta(days=1), init + timedelta(days=5)
    land = np.isfinite(p_dry) | np.isfinite(p_wet)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.3))
    fig.subplots_adjust(top=0.83, bottom=0.20, left=0.07, right=0.97, wspace=0.13)
    fig.suptitle('MEDIUM RANGE WEATHER OUTLOOK', fontsize=21, fontweight='bold', y=0.965)
    fig.text(0.5, 0.915,
             f'Init {init:%d%b%Y}'.upper() +
             f' {cycle}Z  |  Valid {_valid_span(first, last)}',
             ha='center', fontsize=13, fontweight='bold')

    specs = [
        (p_dry, 'DRY HARVEST WINDOW OUTLOOK',
         'Probability of ≥3 consecutive days with rainfall <1 mm within the next 5 days',
         DRY_COLORS),
        (p_wet, 'WET WEATHER OUTLOOK',
         'Probability of ≥1 day with rainfall ≥1 mm within the next 5 days',
         WET_COLORS),
    ]
    for ax, (probability, title, subtitle, colors) in zip(axes, specs):
        cmap = ListedColormap(colors)
        ax.pcolormesh(
            lons, lats, np.ma.masked_invalid(np.asarray(probability, dtype=float) * 100),
            shading='auto', cmap=cmap, norm=BoundaryNorm(BOUNDS, cmap.N),
        )
        ax.contour(lons, lats, land.astype(float), levels=[0.5],
                   colors='#333333', linewidths=1.2)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('#EEEEEE')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=34)
        ax.text(0.5, 1.018, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=12)
        ax.set_xlabel('Longitude (°E)', fontsize=13)
        ax.set_ylabel('Latitude (°N)', fontsize=13)
        ax.set_xticks(np.arange(70, 101, 5))
        ax.set_yticks(np.arange(10, 39, 5))
        ax.tick_params(labelsize=12)
        ax.grid(linewidth=0.35, alpha=0.28)
        handles = [
            Patch(facecolor=color, edgecolor='#888888', linewidth=0.35, label=label)
            for color, label in zip(colors, LABELS)
        ]
        ax.legend(handles=handles, title='Forecast probability', loc='upper center',
                  bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
                  fontsize=11.5, title_fontsize=12.5,
                  handlelength=2.4, handleheight=1.3)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path
