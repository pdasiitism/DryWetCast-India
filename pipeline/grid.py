"""Shared grid definitions and the GRIB2 read+regrid helper.

Target grid is the IMD-aligned 0.25 deg grid used throughout Exp1-Exp15:
lat 6.5-38.5N (129 pts), lon 66.5-100.0E (135 pts). Confirmed against an
actual pre-converted GEFS file on the cluster (/net/monsoon/prabal/
GEFS_INDIA_PRECIP/*/gec00/*_f024.nc: lat 5.0-40.0, lon 65.0-100.0, 141x141,
cropped via LAT_SLICE=slice(6,135)/LON_SLICE=slice(6,141) to this exact grid).

GRIB2 reading uses `rasterio` (GDAL's built-in GRIB driver) rather than
cfgrib/eccodes: rasterio ships prebuilt wheels with GDAL bundled in, so
`pip install rasterio` just works on Windows/Mac/Linux with no separate
system library to compile or version-match — unlike eccodes, whose Python
bindings must link against a matching system `libeccodes`, which is the
single most common setup failure for this kind of pipeline. It's also
faster in practice here (~0.05-0.1s/file vs cfgrib's ~1-2s).

Regridding itself is plain bilinear interpolation (scipy
RegularGridInterpolator) on the decoded numpy array — not a subprocess call
to wgrib2's `-new_grid`, which was measured at ~5-17s per file (GRIB
decode/encode overhead), far too slow for the ~1900 files one forecast
cycle needs.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

N_LAT, N_LON = 129, 135
TARGET_LATS = np.arange(6.5, 38.51, 0.25)   # 129 pts
TARGET_LONS = np.arange(66.5, 100.01, 0.25)  # 135 pts

# Generous crop box around India applied before interpolation, purely to keep
# the interpolator's source array small (cheap) — not a physical boundary.
_CROP_LAT = (0.0, 45.0)
_CROP_LON = (55.0, 110.0)

LAND_MASK_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'land_mask.npy')
CLIMATOLOGY_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'climatology.npz')

# Western-Disturbance box for the Z300 index (Full configs only) — the
# averaging box itself (not the wider download/regrid buffer historical
# scripts used, which isn't needed here since we always decode the full
# global field before any cropping).
WD_LAT = (20.0, 36.5)
WD_LON = (60.0, 80.0)


def load_land_mask() -> np.ndarray:
    """Return the (129,135) boolean land mask, bundled with the repo (no private-archive dependency)."""
    return np.load(LAND_MASK_PATH)


def load_climatology():
    """Return (logit_pi_dry, logit_pi_wet): each (365,129,135), bundled with the repo.

    Only needed by Full configs (Reduced XGB models never use the
    climatology term — see inference.py).
    """
    d = np.load(CLIMATOLOGY_PATH)
    return d['logit_pi_dry'], d['logit_pi_wet']


def climatology_for_init(init_date: str):
    """Return (logit_pi_dry, logit_pi_wet, doy) for a forecast initialised on init_date (YYYYMMDD).

    Training labelled every sample by its VALID date, which is the init date
    + 1 day (every extraction script does init = valid - 1 day), and took both
    the `doy` feature and the climatology slice from that valid date — not
    from the init date. The slice index is clipped to 364 exactly as in
    build_feature_matrix_exp{8,9}.py, so Dec 31 of a leap year (doy 366)
    reuses the last climatology day, as it did in training.
    """
    valid = datetime.strptime(init_date, '%Y%m%d') + timedelta(days=1)
    doy = valid.timetuple().tm_yday
    dry, wet = load_climatology()
    idx = min(doy - 1, 364)
    return dry[idx], wet[idx], doy


def regrid_field(arr: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray) -> np.ndarray:
    """Bilinear-regrid one (nlat,nlon) field onto the 129x135 target grid."""
    order = np.argsort(src_lats)
    src_lats = src_lats[order]
    arr = arr[order]

    lat_mask = (src_lats >= _CROP_LAT[0]) & (src_lats <= _CROP_LAT[1])
    lon_mask = (src_lons >= _CROP_LON[0]) & (src_lons <= _CROP_LON[1])
    sub = arr[np.ix_(lat_mask, lon_mask)]
    sub_lats = src_lats[lat_mask]
    sub_lons = src_lons[lon_mask]

    interp = RegularGridInterpolator((sub_lats, sub_lons), sub, method='linear',
                                      bounds_error=False, fill_value=np.nan)
    pts = np.array([[la, lo] for la in TARGET_LATS for lo in TARGET_LONS])
    return interp(pts).reshape(N_LAT, N_LON).astype(np.float32)


def _read_band(grib_path: str, band: int | None = None, element: str | None = None):
    """Return (arr, lats, lons) for one band of a GRIB2 file.

    `band` selects by 1-indexed position (default: 1, the common case for
    our single-message downloads). `element` instead selects by GRIB_ELEMENT
    tag (exact match), for multi-message files like the combined PWAT+CAPE
    downloads where band order isn't something to rely on.
    """
    with rasterio.open(grib_path) as ds:
        if element is not None:
            band = next(i for i in range(1, ds.count + 1)
                        if ds.tags(i).get('GRIB_ELEMENT', '').upper() == element.upper())
        arr = ds.read(band or 1).astype(np.float32)
        b = ds.bounds
        dx, dy = ds.transform.a, ds.transform.e   # dy is negative (north-to-south rows)
        nlat, nlon = arr.shape
        lats = (b.top + (np.arange(nlat) + 0.5) * dy).astype(np.float64)
        lons = (b.left + (np.arange(nlon) + 0.5) * dx).astype(np.float64)
    return arr, lats, lons


def load_and_regrid(grib_path: str, element: str | None = None, band: int | None = None) -> np.ndarray:
    """Read one GRIB2 message (by GRIB_ELEMENT name, band position, or band 1 if neither given) and regrid it.

    Use `band` when GRIB_ELEMENT is ambiguous (e.g. HGT appears at multiple
    pressure levels with no level info in the tag) — band position is
    reliable whenever you control the exact order messages were downloaded in.
    """
    arr, lats, lons = _read_band(grib_path, band=band, element=element)
    return regrid_field(arr, lats, lons)


def load_and_regrid_cached(grib_path: str, cache_dir: str, element: str | None = None,
                            band: int | None = None) -> np.ndarray:
    """Same as load_and_regrid, but memoized to disk under cache_dir.

    Makes repeat runs (and overlap between MOS and event-fraction extraction,
    which read the same member files) effectively free.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = grib_path.replace('/', '_') + (f'.{element}' if element else '') + (f'.b{band}' if band else '')
    cache_path = os.path.join(cache_dir, key + '.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)
    arr = load_and_regrid(grib_path, element=element, band=band)
    np.save(cache_path, arr)
    return arr


def box_mean(grib_path: str, lat_range: tuple[float, float], lon_range: tuple[float, float],
             element: str | None = None, band: int | None = None) -> float:
    """Unweighted mean of one GRIB2 message's raw (un-regridded) grid points
    inside a lat/lon box — matches extract_atmos_future_exp6.py's
    `_wd_box_mean` exactly (plain np.nanmean over the native-resolution
    source grid, no regridding).
    """
    arr, lats, lons = _read_band(grib_path, band=band, element=element)
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    lon_mask = (lons >= lon_range[0]) & (lons <= lon_range[1])
    return float(np.nanmean(arr[np.ix_(lat_mask, lon_mask)]))
