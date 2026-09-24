"""Registry of deployable forecast configurations.

Each config pairs an NWP source (gefs | ncmrwf) with a feature set
(reduced | full) and points at its own model weights directory under
models/. AIFS-ENS v2 is the second ensemble in every config — it isn't a
per-config choice, so it has no entry here.

Adding a new config means: add the row below, and make sure
models/<name>/ has the matching xgb_reduced_or_full_{dry,wet}.json +
.mask.npz files (produced by train_weights.py for that source/feature_set).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(HERE, '..', 'models')

CONFIGS = {
    'gefs_reduced':   dict(source='gefs',   feature_set='reduced'),
    'gefs_full':      dict(source='gefs',   feature_set='full'),
    'ncmrwf_reduced': dict(source='ncmrwf', feature_set='reduced'),
    'ncmrwf_full':    dict(source='ncmrwf', feature_set='full'),
}


def models_dir(config_name: str) -> str:
    if config_name not in CONFIGS:
        raise ValueError(f'Unknown config {config_name!r}. Choices: {sorted(CONFIGS)}')
    return os.path.join(MODELS_ROOT, config_name)


def model_paths(config_name: str) -> dict:
    d = models_dir(config_name)
    return {
        'dry_json': os.path.join(d, 'xgb_dry.json'),
        'dry_mask': os.path.join(d, 'xgb_dry.mask.npz'),
        'wet_json': os.path.join(d, 'xgb_wet.json'),
        'wet_mask': os.path.join(d, 'xgb_wet.mask.npz'),
    }
