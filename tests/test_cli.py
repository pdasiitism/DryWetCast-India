"""Command-line validation fails fast, before any download."""
import os
import subprocess
import sys

import pytest

from conftest import REPO


def cli(*args):
    return subprocess.run([sys.executable, os.path.join(REPO, 'run_forecast.py'), *args],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('args, message', [
    (['--config', 'gefs_reduced', '--date', '2025-11-14'], 'not a valid YYYYMMDD'),
    (['--config', 'gefs_reduced', '--date', '20251340'], 'not a valid YYYYMMDD'),
    (['--config', 'bogus', '--date', '20251114'], 'invalid choice'),
    (['--config', 'gefs_reduced'], '--date is required'),
    (['--date', '20251114', '--download-only', '--skip-download'], 'not allowed with'),
    (['--config', 'gefs_reduced', '--date', '20251114', '--aifs-zarr', '/tmp/x.nc'], 'must be a .zarr'),
    (['--config', 'gefs_reduced', '--date', '20251114', '--aifs-zarr', '/nonexistent/x.zarr'], 'does not exist'),
])
def test_bad_arguments(args, message):
    r = cli(*args)
    assert r.returncode == 2
    assert message in r.stderr
