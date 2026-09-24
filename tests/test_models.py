"""The shipped weights load and give sane probabilities for every config."""
import numpy as np
import pytest

from pipeline import configs, grid, inference


@pytest.mark.parametrize('config', sorted(configs.CONFIGS))
@pytest.mark.parametrize('event', ['dry', 'wet'])
def test_weights_predict(config, event):
    paths = configs.model_paths(config)
    names = [str(n) for n in np.load(paths[f'{event}_mask'], allow_pickle=True)['feature_names']]
    assert names, 'mask has no features'
    land = grid.load_land_mask()
    rng = np.random.default_rng(0)
    feats = {n: rng.normal(size=land.shape).astype(np.float32) for n in names}
    p = inference.predict_xgb(paths[f'{event}_json'], paths[f'{event}_mask'], feats, land)
    assert p.shape == (129, 135)
    assert np.all(np.isnan(p[~land])), 'sea must be NaN'
    assert np.all((p[land] >= 0) & (p[land] <= 1))


def test_unknown_config():
    with pytest.raises(ValueError):
        configs.models_dir('nope')
