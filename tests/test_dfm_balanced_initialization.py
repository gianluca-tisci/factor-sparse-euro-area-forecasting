"""The alternative initializer cuts the seed, not the observation likelihood."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.forecasting.dfm import MixedFrequencyEM
from src.forecasting.initialization import BalancedStartEM, balanced_start


def test_adopted_profile_preserves_historical_sensitivity_except_training_policy():
    original = yaml.safe_load(Path("config/forecast_analysis.yaml").read_text())
    experimental = yaml.safe_load(Path("config/forecast_analysis_dfm_balanced.yaml").read_text())
    assert (
        original["dfm_initialization"] == experimental["dfm_initialization"] == "balanced_t_minus_3"
    )
    assert original.pop("classification") == "BASELINE"
    assert experimental.pop("classification") == "EXPLORATORY"
    assert original.pop("training_outlier_policy") == "cleaned_inputs_and_labels"
    assert experimental.pop("training_outlier_policy") == "published_labels_and_own_lags"
    assert original == experimental


def test_seed_ignores_last_three_months_but_likelihood_uses_them():
    rng = np.random.default_rng(67)
    index = pd.period_range("2000-01", periods=72, freq="M")
    x = pd.DataFrame(rng.normal(size=(72, 8)), index=index)
    x.loc[index.month % 3 != 0, 7] = np.nan
    altered = x.copy()
    altered.iloc[-3:, :7] += 50
    original = MixedFrequencyEM(x, 7)
    other = MixedFrequencyEM(altered, 7)
    a, b = balanced_start(original), balanced_start(other)
    for field in ["loadings", "variances", "ar", "innovation"]:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert original.expectation(a).llf != other.expectation(a).llf
    experiment = BalancedStartEM(x, 7)
    np.testing.assert_array_equal(experiment.y, original.y)
    assert experiment.expectation(a).llf == original.expectation(a).llf
