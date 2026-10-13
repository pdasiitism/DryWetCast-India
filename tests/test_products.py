"""Portable product rendering and subdistrict aggregation helpers."""

import numpy as np
import pytest

from pipeline import outlook, subdistrict


def test_outlook_output_path():
    assert outlook.output_path('/tmp/forecast.png') == '/tmp/forecast_outlook.png'


def test_subdistrict_output_paths():
    assert subdistrict.output_paths('/tmp/forecast.png') == (
        '/tmp/forecast_subdistrict.pkl', '/tmp/forecast_subdistrict.png')


def test_subdistrict_weighted_aggregation():
    lats = np.array([10.0, 11.0])
    lons = np.array([70.0, 71.0])
    p_dry = np.array([[0.2, 0.4], [0.6, np.nan]])
    p_wet = np.array([[0.8, 0.6], [0.4, np.nan]])
    source = {'10.0_70.0': 0, '10.0_71.0': 1, '11.0_70.0': 2}
    target = {'A': 0, 'B': 1}
    weights = np.array([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])

    result = subdistrict.aggregate(
        p_dry, p_wet, lats, lons, source, target, weights)
    assert result['A']['dry_probability'] == pytest.approx(0.3)
    assert result['A']['wet_probability'] == pytest.approx(0.7)
    assert result['B']['dry_probability'] == pytest.approx(0.55)
    assert result['B']['wet_probability'] == pytest.approx(0.45)
