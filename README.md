# Operational DRY/WET forecast (XGBoost, all 4 configs)

Self-contained pipeline: downloads GEFS/NCMRWF + AIFS-ENS v2 data for one
forecast cycle, extracts the predictor set, runs the bundled XGBoost
weights, and produces DRY/WET probability maps over India.

No cluster paths, no private data — everything needed either ships in
`models/` or is fetched live:
- GEFS: `https://noaa-gefs-pds.s3.amazonaws.com` (NOAA public S3 bucket)
- AIFS-ENS v2: `https://data.ecmwf.int/forecasts` (ECMWF public open-data feed),
  or your own cluster's AIFS run (`--aifs-zarr`)
- NCMRWF (NEPS): the NCMRWF data portal `https://cloud.ncmrwf.gov.in` (needs
   API key), or files already on your machine (`--ncmrwf-file`)

## Configs

This repo supports all 4 XGBoost configs (`pipeline/configs.py`), each with
its own weights in `models/<config>/`, and all 4 run end-to-end:

| Config | Rainfall source | Feature set |
|---|---|---|
| `gefs_reduced` | GEFS | 8-9 LASSO-selected features |
| `ncmrwf_reduced` | NCMRWF (NEPS) | 8-9 LASSO-selected features |
| `gefs_full` | GEFS | all 42 features |
| `ncmrwf_full` | NCMRWF (NEPS) | all 42 features |

**Atmospheric features (PWAT, CAPE, and for Full also RH850/RH700/wind
speed@850/omega@850/Z850/Z500/Z300-index) always come from GEFS**, in every
config — NCMRWF+AIFS was never trained with its own atmospheric fields,
only GEFS's. So `ncmrwf_*` configs still download GEFS data, just use it
for the atmosphere half instead of the rainfall half.

## Setup

Needs Python 3.10+ (the bundled models were saved with XGBoost 3.x, which
requires it). Two ways to install — pick one:

```bash
# Laptop / any machine with Python 3.10+
pip install -r requirements.txt

# Cluster, or any machine with an older system Python — conda installs its
# own Python and libraries, independent of the system's
conda env create -f environment.yml
conda activate dry-wet-forecast
```

No separate system GRIB library to install. GRIB2 reading uses `rasterio`
(GDAL's built-in GRIB driver, bundled in the wheel); NCMRWF's NetCDF files
read via plain `xarray`/`netCDF4`. Neither needs `eccodes`.

**macOS (Apple Silicon) with pip:** XGBoost needs the OpenMP runtime, which
macOS doesn't ship — `brew install libomp` once, or XGBoost fails to import.
(The conda route includes it.) Tested on a Mac with Python 3.13 and XGBoost 3.4.

Then confirm the machine is ready before a real run:

```bash
python run_forecast.py --check
```

It checks the Python version, every package, the model files, whether the
GEFS/AIFS feeds are reachable from this machine, how many CPUs it will use,
and free disk space — and says exactly what to fix if anything's wrong.

## Run a forecast

```bash
# GEFS+AIFS — fully automated
python run_forecast.py --config gefs_reduced --date 20260920 --cycle 00
python run_forecast.py --config gefs_full --date 20260920 --cycle 00

# NCMRWF+AIFS — NCMRWF downloaded from the NCMRWF portal (API key needed, see below)
python run_forecast.py --config ncmrwf_reduced --date 20260920 --cycle 00
python run_forecast.py --config ncmrwf_full --date 20260920 --cycle 00

# NCMRWF+AIFS — NCMRWF files already on this machine (no key, nothing downloaded)
python run_forecast.py --config ncmrwf_reduced --date 20260920 --cycle 00 \
    --ncmrwf-file /path/to/20260920.nc
```

NCMRWF is one file per cycle: `<init_date>.nc` (also accepted:
`<init_date>_NPES.nc`), the 23-member ensemble, used for the daily mean/spread
and the DRY/WET event probabilities. Its forecast days are checked against
`--date`, so a wrong day's file is rejected. (NCMRWF also uploads a
`<init_date>.lag.nc`; the pipeline ignores it.)

### NCMRWF portal API key

The portal (`https://cloud.ncmrwf.gov.in`) uploads each day's files around
12:00. Give the pipeline our key in **one** of these ways — never on the
command line (it would end up in our shell history), and never commit it:

```bash
# Recommended, once: type the key at a hidden prompt (not echoed, not saved in shell history)
read -rs -p "NCMRWF API key: " K && printf '%s\n' "$K" > ~/.ncmrwf_api_key && chmod 600 ~/.ncmrwf_api_key && unset K

# or, for a session / batch job, an environment variable
export NCMRWF_API_KEY=...
```

`python run_forecast.py --check` then confirms the portal accepts the key
(without printing it). The file is found by name under the portal's top
folder (currently `2026/` for real time, `5day/` for 2025), skipping
`forecast/` and `additional_dates_*/`, which hold a different NCMRWF product
with the same file names. To look in one folder only, set `NCMRWF_REMOTE_DIR`
to it. Each download is also checked to be the 5-day ensemble. Downloads are checked (complete size,
readable NetCDF) before use, and re-running reuses files already downloaded.
If the day's file isn't up yet, the run stops with that message before
downloading anything else.

Each run writes two files:
- `forecast_product_dry_wet.png` — the DRY/WET probability maps (for viewing)
- `forecast_product_dry_wet.npz` — the actual forecast record (for archiving):
  `p_dry`, `p_wet` as (129,135) float32 grids (NaN over sea), plus `lats`,
  `lons`, `date`, `cycle`, `config`. Exact, ~37KB per cycle, one `np.load`.

Options:
- `--out-dir` — where to cache downloaded GEFS/AIFS data (default `./_scratch`; ~1.2 GB per cycle)
- `--fig-out` — output image path; the `.npz` is written next to it (default `./forecast_product_dry_wet.png`)
- `--archive-dir` — also append this run into a growing archive (see below)
- `--download-only` / `--skip-download` — split downloading from processing (see "Running on a cluster")
- `--workers N` — feature-extraction processes (default: CPUs available to this job, max 16)
- `--aifs-zarr PATH` — use AIFS run locally on your cluster instead of downloading it (see below)
- `--cycle` — `00`/`06`/`12`/`18` (00Z is what every model was trained against)
- `--check` — check this machine's setup, then exit

## Running on a cluster

On many clusters the compute nodes (where batch jobs run) have no internet,
while the login node has internet but shouldn't run heavy processing. Split
the two steps:

```bash
# 1. Login node — download only (light, ~1-2 min; needs internet)
python run_forecast.py --date 20260920 --download-only --out-dir /scratch/$USER/dwf

# 2. Batch job on a compute node — process only (no internet needed)
python run_forecast.py --config gefs_reduced --date 20260920 --skip-download \
    --out-dir /scratch/$USER/dwf
```

Use the same `--out-dir` for both. Without `--config`, step 1 also fetches the
NCMRWF files from the portal when an API key is set up (into
`<out-dir>/ncmrwf_<date>/`, where step 2 finds them). One download serves every config for that
date, so step 2 can be repeated for each config without downloading again.
The worker count follows the job's actual CPU allocation automatically (it
reads the CPU binding and `SLURM_CPUS_PER_TASK`, not the node's total core
count), so a `--cpus-per-task=4` job uses 4 processes. Run `--check` inside
a job to see what it detects.

Keep `--out-dir` on persistent storage (home or project/scratch space), not
`/tmp`: many clusters periodically delete old files from `/tmp`.

On a laptop none of this applies — just run the single command above.

## Using AIFS run on your own cluster (no AIFS download)

**Current practice:** AIFS is downloaded from ECMWF's open-data feed (the
default — nothing to set). **Once AIFS-ENS v2 is running operationally on
the cluster**, switch to that run's output with `--aifs-zarr`; it's what the
models were trained on, and reading it takes seconds instead of a ~10-minute
download. Nothing else changes: it works with every config, GEFS is still
downloaded (it supplies the atmosphere fields, and GEFS rainfall for
`gefs_*` configs), and NCMRWF still comes from the portal or `--ncmrwf-file`.

```bash
python run_forecast.py --config gefs_reduced --date 20251114 \
    --aifs-zarr /path/to/init_20251114T00.zarr

# on a cluster without internet on compute nodes, --download-only now
# fetches GEFS only
python run_forecast.py --date 20251114 --download-only --aifs-zarr /path/to/init_20251114T00.zarr
```

Needs the `zarr` package (`pip install zarr`) — optional, only for this mode.

**Expected format** — the same as the stores the models were trained on
(`/net/monsoon/prabal/AIFS_ens_v2/init_<YYYYMMDD>T00.zarr`), so a run
produced the same way is used exactly as in training:

| Array | Shape / content |
|---|---|
| `tp` | `(time=1, number, prediction_timedelta, lat, lon)`, **metres**, **6-hourly amounts** (not cumulative) |
| `lat`, `lon` | the 129×135 India grid: lat 6.5–38.5, lon 66.5–100.0 at 0.25° (either lat order) |
| `prediction_timedelta` | lead hours; must include every 6h from +6 to +126h (+6 to +102h is enough for Reduced configs) |
| `number` | ensemble members, at least 20 (training used 51) |
| `time` | init time, units `days since YYYY-MM-DD ...` |

**Units differ between the two AIFS sources, and the pipeline handles each:**
the cluster zarr stores `tp` in **metres** per 6 hours (training multiplied by
1000 to get mm, and so does this pipeline), while the downloaded ECMWF
open-data GRIB has `tp` in **kg/m² = mm**, cumulative since init, used as mm
directly. (GDAL labels that field "TPRATE [kg/(m^2*s)]"; it is a naming quirk,
the values are totals.) So a new cluster AIFS run must write `tp` in metres,
like the training stores. If it writes mm by mistake, the run is refused
(rainfall 1000× too large), and so is the opposite mistake (no rain anywhere).

The store is checked before use, and rejected with a clear message if its
grid doesn't match, required lead hours are missing, it has too few
members, its init date differs from `--date` (a wrong day's file), or its
rainfall totals are implausible (the usual sign of wrong units or
cumulative instead of 6-hourly values). Verified: for the 14 Nov 2025 store,
every AIFS feature this produces is identical to the features the models
were trained on for that date.

## Archiving forecasts over time

Pass the same `--archive-dir` on every run and forecasts accumulate as one
file per run, grouped by config:

```
archive/gefs_reduced/20260922_00.npz
archive/ncmrwf_full/20260922_00.npz
...
```

Load one run, or many (optionally filtered):

```python
import numpy as np
d = np.load('archive/ncmrwf_reduced/20260922_00.npz')
d['p_dry']   # (129,135)

from pipeline import archive
runs = archive.read_archive('archive/', config='ncmrwf_reduced')   # list of dicts, sorted by date
runs = archive.read_archive('archive/', date='20260922')           # every config for that date
```

A full run takes ~3-5 minutes (downloads: under a minute; feature extraction
is the rest — GRIB2/NetCDF decoding parallelized across 16 workers; Full
configs' extra atmospheric variables add well under a minute thanks to
disk-caching of already-decoded GEFS files shared with the rainfall step).

## What's in `models/`

- `<config>/xgb_dry.json`, `xgb_wet.json` — native XGBoost boosters
- `<config>/xgb_dry.mask.npz`, `xgb_wet.mask.npz` — which source features
  each model uses, and their names, in the exact order the booster expects
- `land_mask.npy` — (129,135) boolean India land mask on the target grid, shared across configs
- `climatology.npz` — daily (365,129,135) DRY/WET climatology fields, used
  only by Full configs (`logit_pi_dry`/`logit_pi_wet` features)

Weights were trained once (`python -m pipeline.train_weights <config>`) by
exactly reproducing each experiment's final-production recipe — pooled
2021-2024 training years, same XGBoost hyperparameters throughout, and (for
Reduced configs) the already-selected LASSO feature mask. Retraining needs
the private historical archive (`/net/monsoon/...`) and isn't part of this
repo; the point of shipping the weights is that nobody else needs it.
Validated against the originally-evaluated BSS on the 2020/2025 held-out
test years — all 4 configs x DRY/WET x 2 years (16 checks) matched exactly.

**Reduced** (`gefs_reduced`, `ncmrwf_reduced`): 8-9 LASSO-selected features
from the 49-feature layout (Exp10/Exp11/Exp14/Exp15).

**Full** (`gefs_full`, `ncmrwf_full`): a genuinely separate, older 42-feature
layout (Exp8/Exp9) — no LASSO reduction, no interaction terms, single `doy`
column instead of `doy_sin`/`doy_cos`, and 7 more atmospheric variables than
Reduced uses (RH850, RH700, wind speed@850, omega@850, Z850, Z500, plus the
Z300 Western-Disturbance index — a scalar area-mean over a box wider than
the India target grid, computed straight from the raw undecoded GEFS field).

## Package layout

```
pipeline/
  configs.py           registry of the 4 configs and their models/ paths
  grid.py               target 129x135 grid, bilinear regrid, land mask + climatology loaders
  download_gefs.py      GEFS byte-range downloader (NOAA S3, .idx-based)
  download_aifs.py      AIFS-ENS v2 byte-range downloader (ECMWF, .index-based)
  gefs_features.py      raw GEFS GRIB2 -> Reduced + Full features for one cycle
  aifs_features.py      AIFS (downloaded GRIB2 or local zarr) -> Reduced + Full features for one cycle
  download_ncmrwf.py    NCMRWF portal downloader (API key, one file per cycle)
  ncmrwf_features.py    raw NCMRWF NetCDF -> Reduced + Full rainfall features for one cycle
  inference.py          assemble features (Reduced or Full), run the saved XGBoost weights
  archive.py            write/read the .npz forecast record and archive folder
  runtime.py            CPU count available to this job (scheduler-aware)
  check.py              the --check setup report
  train_weights.py      (one-time, needs private archive; VERIFICATION_ROOT) produces models/<config>/
run_forecast.py         top-level CLI: download -> extract -> infer -> plot, any config
requirements.txt        pip install
environment.yml         conda install (brings its own Python)
notebooks/              operational_forecast.ipynb — interactive run via run_forecast.run()
tests/                  offline test suite (no network, no private data)
slurm/                  SLURM script for retraining (only if models/ is lost)
```

## Tests

```bash
python -m pytest tests
```

Runs offline in ~20 s, from any clone: the shipped weights load and predict
for all four configs, the climatology date rule (valid = init + 1, leap-year
clip), the `.npz` record round trip, command-line validation, and the NCMRWF
portal downloader against a local stand-in for the portal (nested folders,
both naming styles, wrong key, not-yet-uploaded, truncated download, error
page, missing key — and that the key never appears in an error message).
It never reads your real API key.

## Things to consider

- The models were trained for the post-monsoon period, from mid-October to
  late December. Forecasts outside this period should be treated as
  extrapolation.
- ECMWF retains only a rolling window of recent AIFS cycles, so older forecast
  dates may no longer be available from the open-data feed.
- The NCMRWF models were trained on the older 11-member GRIB product, while
  operational forecasts use the newer 23-member NetCDF product.
