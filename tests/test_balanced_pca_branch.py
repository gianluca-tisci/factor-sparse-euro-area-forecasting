"""Temporal isolation and VAR carry for the branch's PCA-only update."""

import numpy as np
import pandas as pd

from src.forecasting.experiment import fit_var
from src.forecasting.pca import balanced_pca, carried_common


def test_balanced_factors_ignore_recent_values_and_carry_var():
    rng = np.random.default_rng(48)
    index = pd.period_range("2000-01", periods=80, freq="M")
    panel = pd.DataFrame(rng.normal(size=(80, 8)), index=index)
    panel.iloc[15, 0] = np.nan  # internal cleaned outlier
    factors, loading, dates, _ = balanced_pca(panel, 3, 3)
    altered = panel.copy()
    altered.iloc[-3:] += 10000
    other_f, other_l, _, _ = balanced_pca(altered, 3, 3)
    np.testing.assert_array_equal(factors, other_f)
    np.testing.assert_array_equal(loading, other_l)
    assert dates[-1] == index[-1] - 3
    var, _ = fit_var(factors, 2)
    common = carried_common(factors, loading, var, dates, index)
    np.testing.assert_allclose(common.iloc[:-3], factors @ loading.T)
    ahead = var.forecast(factors[-var.k_ar :], 3) if var.k_ar else np.tile(var.intercept, (3, 1))
    np.testing.assert_allclose(common.iloc[-3:], ahead @ loading.T)
