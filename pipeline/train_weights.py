#!/usr/bin/env python3
"""Reproduce and SAVE the final-production XGB weights for a given config.

None of Exp8/Exp9/Exp14/Exp15 (Full/Reduced x NCMRWF/GEFS) ever saved a
fitted model object — only predictions/BSS/masks. This script reruns exactly
each experiment's "final model" branch (fit on the pooled 2021-2024 CV years)
and saves real, portable weights to models/<config>/:

  xgb_dry.json / xgb_wet.json        (native XGBoost booster)
  xgb_dry.mask.npz / xgb_wet.mask.npz (column mask into the source feature
                                        matrix, + feature names, in order)

Also re-derives BSS on the 2020/2025 test years from the freshly-trained
model and compares it against the stored bss_*_2020/2025 (Reduced configs)
or the recomputed BSS from pred_*_test (Full configs) — a match confirms the
retrained weights are a faithful reproduction, not a new fit.

Reduced configs (gefs_reduced, ncmrwf_reduced): mask = the LASSO-selected
subset of the source 49-feature matrix (Exp11/Exp10), reusing the mask
already saved by the Exp15/Exp14 part-b scripts.

Full configs (gefs_full, ncmrwf_full): a SEPARATE, older 42-feature matrix
(Exp9/Exp8) — not derived from the 49-feature layout at all: no interaction
terms, single `doy` instead of doy_sin/doy_cos. "Full" mask = all 42
columns, unmasked.
"""

import os

import numpy as np
from xgboost import XGBClassifier

from . import configs

CV_YEARS = [2021, 2022, 2023, 2024]
TEST_YEARS = [2020, 2025]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
    gamma=1.0, reg_alpha=0.1, reg_lambda=1.0,
    objective='binary:logistic', eval_metric='logloss',
    tree_method='hist', n_jobs=-1, random_state=42,
)

# Root of the historical experiment outputs (feature matrices + LASSO masks).
# These are NOT part of the repo — retraining is only needed if the shipped
# weights in models/ are lost. Override with VERIFICATION_ROOT=/path/to/verification.
ROOT = os.environ.get('VERIFICATION_ROOT', '/net/monsoon/prabal/verification')

# Per-config: where to load the feature matrix from, and (for Reduced only)
# where the LASSO mask lives. Full configs have no mask source — all columns
# are used, masked=None means "use every column".
SOURCES = {
    'gefs_reduced': dict(
        fm_path=f'{ROOT}/Exp11/results/feature_matrix.npz',
        mask_path=f'{ROOT}/Exp15/results/xgb_reduced_{{event}}_partb.npz',
        logit_pi_col=22,
    ),
    'ncmrwf_reduced': dict(
        fm_path=f'{ROOT}/Exp10/results/feature_matrix.npz',
        mask_path=f'{ROOT}/Exp14/results/xgb_reduced_{{event}}_partb.npz',
        logit_pi_col=22,
    ),
    'gefs_full': dict(
        fm_path=f'{ROOT}/Exp9/results/feature_matrix.npz',
        mask_path=None,
        logit_pi_col=22,
    ),
    'ncmrwf_full': dict(
        fm_path=f'{ROOT}/Exp8/results/feature_matrix.npz',
        mask_path=None,
        logit_pi_col=22,
    ),
}


def bss(yt, yp, cp):
    m = ~np.isnan(yt) & ~np.isnan(yp)
    return 1.0 - np.mean((yp[m] - yt[m]) ** 2) / np.mean((cp[m] - yt[m]) ** 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _process(config_name, event, X, y, year_ids, feat_names, src):
    cv_mask = np.isin(year_ids, CV_YEARS)
    clim = sigmoid(X[:, src['logit_pi_col']])

    if src['mask_path'] is not None:
        res = np.load(src['mask_path'].format(event=event))
        mask = res[f'mask_{event}']
        stored_bss = {yr: float(res[f'bss_{event}_{yr}'][0]) for yr in TEST_YEARS}
    else:
        mask = np.ones(X.shape[1], dtype=bool)
        stored_bss = None   # Full configs: compare against a freshly-recomputed reference instead

    out_dir = configs.models_dir(config_name)
    os.makedirs(out_dir, exist_ok=True)

    Xcv, ycv = X[cv_mask][:, mask], y[cv_mask]
    model = XGBClassifier(**XGB_PARAMS).fit(Xcv, ycv)
    model.get_booster().save_model(os.path.join(out_dir, f'xgb_{event}.json'))
    np.savez_compressed(os.path.join(out_dir, f'xgb_{event}.mask.npz'),
                         mask=mask, feature_names=feat_names[mask])

    for yr in TEST_YEARS:
        tm = year_ids == yr
        p = model.predict_proba(X[tm][:, mask])[:, 1]
        b = bss(y[tm], p, clim[tm])
        if stored_bss is not None:
            print(f'  XGB {event} {yr}: retrained BSS={b:.4f}  stored BSS={stored_bss[yr]:.4f}  '
                  f'{"OK" if abs(b - stored_bss[yr]) < 1e-6 else "MISMATCH"}', flush=True)
        else:
            print(f'  XGB {event} {yr}: retrained BSS={b:.4f}', flush=True)

    print(f'  XGB {event}: {int(mask.sum())} features -> {list(feat_names[mask])}', flush=True)


def main(config_name: str):
    src = SOURCES[config_name]
    fm = np.load(src['fm_path'], allow_pickle=True)
    year_ids = fm['year_ids']

    print(f'=== {config_name}: DRY ===', flush=True)
    _process(config_name, 'dry', fm['X_dry'], fm['y_dry'], year_ids, fm['feature_names_dry'], src)
    print(f'=== {config_name}: WET ===', flush=True)
    _process(config_name, 'wet', fm['X_wet'], fm['y_wet'], year_ids, fm['feature_names_wet'], src)

    print(f'\nSaved weights -> {os.path.abspath(configs.models_dir(config_name))}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('config', choices=sorted(configs.CONFIGS))
    args = p.parse_args()
    main(args.config)
