import pytest
from lib.config import Config, MLConfig

def test_config_defaults():
    config = Config()
    assert config.nums_trials == 50
    assert config.n_exps == 10
    assert config.min_num_per_class == 10

def test_ml_config():
    ml_config = MLConfig()
    assert "xgboost" in ml_config.evaluators
    assert "adult" in ml_config.datasets
    assert "privsyn" in ml_config.algorithms 