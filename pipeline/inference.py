"""Combine Reduced-set features into model inputs and run inference.

Feature name -> column meaning (subset of the 49-feature Exp10/Exp11 layout
actually used by the reduced XGB models; see build_feature_matrix_exp{10,11}.py).
XGB's LASSO mask excludes all interaction columns (25-30) via apply_xgb_mask,
so no climatology term is needed here — only raw rain/AIFS/atmos features:

  {source}_mean_D3, {source}_mean_D4      : rain-source ensemble-mean rainfall, day 3 / day 4
  {source}_prob_dry_event/wet_event       : rain-source ensemble event fraction
  aifs_ens_mean_D3                        : AIFS ensemble-mean rainfall, day 3
  aifs_ens_spread_D1..D4                  : AIFS ensemble spread, days 1-4
  aifs_ens_prob_dry_event/wet_event       : AIFS ensemble event fraction
  pw_mean_roll                            : precipitable water, D1-D5 future mean (GEFS ens-mean product)
  cape_mean_roll                          : CAPE, D1-D5 future mean (GEFS ens-mean product)
  cape_spread_roll                        : CAPE, D1-D5 future mean (GEFS ens-spread product)

`source` is 'gefs' or 'ncmrwf' — which NWP model's rainfall drives the DRY/WET
event signal. Atmospheric features (pw_mean_roll, cape_*_roll) always come
from GEFS regardless of `source`, since that's what the training data used
for BOTH configs (NCMRWF+AIFS never had its own atmospheric fields — see
ncmrwf_features.py's docstring).
"""

import numpy as np
import xgboost as xgb


def build_feature_dict(source, rain_mos, rain_event_prob, gefs_pwat_cape, aifs_mos, aifs_event_prob):
    """Assemble every named feature the reduced XGB models need, as (129,135) arrays."""
    rain_mean, _rain_spread = rain_mos
    rain_prob_dry, rain_prob_wet = rain_event_prob
    pw_mean, cape_mean, cape_spread = gefs_pwat_cape
    aifs_mean, aifs_spread = aifs_mos
    aifs_prob_dry, aifs_prob_wet = aifs_event_prob

    return {
        f'{source}_mean_D3': rain_mean[2],
        f'{source}_mean_D4': rain_mean[3],
        f'{source}_prob_dry_event': rain_prob_dry,
        f'{source}_prob_wet_event': rain_prob_wet,
        'aifs_ens_mean_D3': aifs_mean[2],
        'aifs_ens_spread_D1': aifs_spread[0],
        'aifs_ens_spread_D2': aifs_spread[1],
        'aifs_ens_spread_D3': aifs_spread[2],
        'aifs_ens_spread_D4': aifs_spread[3],
        'aifs_ens_prob_dry_event': aifs_prob_dry,
        'aifs_ens_prob_wet_event': aifs_prob_wet,
        'pw_mean_roll': pw_mean,
        'cape_mean_roll': cape_mean,
        'cape_spread_roll': cape_spread,
    }


def build_feature_dict_full(source, rain_mos, rain_event_prob, aifs_mos, aifs_event_prob,
                             gefs_atmos_full, logit_pi_dry_today, logit_pi_wet_today, doy):
    """Assemble every named feature the Full (42-feature, Exp8/Exp9) XGB
    models need, as (129,135) arrays.

    `gefs_atmos_full` is gefs_features.compute_gefs_atmos_full()'s return
    dict — its 16 spatial keys (pw/rh850/ws850/rh700/omega850/z850/z500/cape
    x mean/spread) match these feature names directly; `z300_idx_mean` and
    `z300_idx_spread` are scalars there and get broadcast to the grid here.
    `logit_pi_{dry,wet}_today` are (129,135) climatology fields for the
    forecast's day-of-year (grid.load_climatology()[i][doy-1]), not scalars.
    """
    rain_mean, rain_spread = rain_mos
    rain_prob_dry, rain_prob_wet = rain_event_prob
    aifs_mean, aifs_spread = aifs_mos
    aifs_prob_dry, aifs_prob_wet = aifs_event_prob

    feat = {}
    for d in range(5):
        feat[f'{source}_mean_D{d+1}'] = rain_mean[d]
        feat[f'{source}_spread_D{d+1}'] = rain_spread[d]
        feat[f'aifs_ens_mean_D{d+1}'] = aifs_mean[d]
        feat[f'aifs_ens_spread_D{d+1}'] = aifs_spread[d]
    feat[f'{source}_prob_dry_event'] = rain_prob_dry
    feat[f'{source}_prob_wet_event'] = rain_prob_wet
    feat['aifs_ens_prob_dry_event'] = aifs_prob_dry
    feat['aifs_ens_prob_wet_event'] = aifs_prob_wet
    feat['logit_pi_dry'] = logit_pi_dry_today
    feat['logit_pi_wet'] = logit_pi_wet_today
    feat['doy'] = np.full_like(rain_prob_dry, doy, dtype=np.float32)

    for name, value in gefs_atmos_full.items():
        feat[name] = value if hasattr(value, 'shape') else np.full_like(rain_prob_dry, value, dtype=np.float32)

    return feat


def _stack_features(feat_dict, feature_names, land_mask):
    """Return X (n_land, n_feat) built in `feature_names` order, plus (rows, cols)."""
    rows, cols = np.where(land_mask)
    X = np.empty((len(rows), len(feature_names)), dtype=np.float32)
    for j, name in enumerate(feature_names):
        X[:, j] = feat_dict[name][rows, cols]
    return X, rows, cols


def predict_xgb(model_json_path, mask_npz_path, feat_dict, land_mask):
    w = np.load(mask_npz_path, allow_pickle=True)
    feature_names = [str(n) for n in w['feature_names']]
    X, rows, cols = _stack_features(feat_dict, feature_names, land_mask)
    booster = xgb.Booster()
    booster.load_model(model_json_path)
    p = booster.inplace_predict(X)
    out = np.full(land_mask.shape, np.nan, dtype=np.float32)
    out[rows, cols] = p
    return out
