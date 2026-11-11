"""Calendar periods, target publication, sparse selection and benchmark fairness."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.design import (
    bic_regression,
    dictionary,
    sparse_predict,
    target_month,
    training_sample,
)
from src.forecasting.experiment import clean_window


def test_required_missing_ar_lag_is_logged_not_imputed():
    x = pd.DataFrame({"M1": np.arange(40.0), "M2": np.arange(40.0) ** 2})
    y = np.sin(np.arange(40.0))
    row = pd.Series({"M1": np.nan, "M2": 2.0})
    prediction, detail = bic_regression(x, y, row, [["M1"], ["M1", "M2"]])
    assert np.isnan(prediction)
    assert detail["n_train"] == 40
    assert "unavailable" in detail["failure_reason"]


def example():
    index = pd.period_range("2000-01", periods=30, freq="M")
    panel = pd.DataFrame(
        {
            "fast": np.arange(30.0),
            "slow": np.arange(30.0) + 100,
            "quarterly": np.arange(30.0) + 200,
        },
        index=index,
    )
    panel.loc[index.month % 3 != 0, "quarterly"] = np.nan
    freq = pd.Series({"fast": "M", "slow": "M", "quarterly": "Q"})
    lags = pd.Series({"fast": 0, "slow": 2, "quarterly": 2})
    return panel, freq, lags


def test_quarterly_lags_are_released_calendar_quarters_not_monthly_shifts():
    panel, freq, lags = example()
    origin = pd.PeriodIndex(["2001-04", "2001-05", "2001-06"], freq="M")
    x, references = dictionary(panel, freq, lags, origin)
    assert str(references.loc["2001-04", "quarterly__Q1"]) == "2000-12"
    assert str(references.loc["2001-05", "quarterly__Q1"]) == "2001-03"
    assert str(references.loc["2001-06", "quarterly__Q1"]) == "2001-03"
    assert str(references.loc["2001-05", "quarterly__Q2"]) == "2000-12"
    assert x.loc["2001-05", "slow__M1"] == panel.loc["2001-03", "slow"]
    panel.loc["2001-03", "quarterly"] = np.nan
    missing, _ = dictionary(panel, freq, lags, origin)
    assert np.isnan(missing.loc["2001-05", "quarterly__Q1"])


@pytest.mark.parametrize(
    "lag,month", [(0, "2001-07"), (1, "2001-06"), (2, "2001-05"), (3, "2001-04")]
)
def test_first_unpublished_month_and_label_availability(lag, month):
    origin = pd.Period("2001-06", "M")
    assert str(target_month(origin, lag)) == month
    panel, _, _ = example()
    features = panel[["fast"]]
    x, y = training_sample(features, panel.fast, lag, origin)
    assert x.index[-1] == origin - 1
    assert len(y) == len(x)


def test_unpublished_and_future_values_cannot_change_current_preprocessing_or_features():
    panel, freq, lags = example()
    origin = pd.Period("2001-06", "M")
    altered = panel.copy()
    for name in panel:
        altered.loc[altered.index > origin - int(lags[name]), name] += 999999
    first = clean_window(panel, lags, origin)
    second = clean_window(altered, lags, origin)
    for a, b in zip(first, second, strict=True):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        else:
            pd.testing.assert_series_equal(a, b)
    a, _ = dictionary(first[1], freq, lags, first[1].index)
    b, _ = dictionary(second[1], freq, lags, second[1].index)
    pd.testing.assert_frame_equal(a, b)


def test_lasso_support_cap_and_training_only_standardization():
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(120, 30)), columns=[f"x{i}" for i in range(30)])
    y = 2 * x.x0 - x.x1 + rng.normal(scale=0.15, size=120)
    forecast = x.iloc[-1].copy()
    _, support, info = sparse_predict(x, y, forecast)
    _, other_support, other_info = sparse_predict(x, y, forecast * 1000)
    assert len(support) <= len(y) // 10
    assert {"x0", "x1"}.issubset(support.index)
    pd.testing.assert_series_equal(support, other_support)
    assert info == other_info


def test_zero_inflated_series_flags_only_the_genuine_spike():
    """D-72: a mostly-zero series keeps its ordinary moves; only the spike is flagged."""
    rng = np.random.default_rng(0)
    index = pd.period_range("2000-01", periods=200, freq="M")
    values = rng.normal(0, 1, 200)
    order = rng.permutation(200)
    values[order[:110]] = 0.0  # 55% exact zeros, as in early iron-ore prices
    values[order[110:130]] = rng.normal(0, 0.01, 20)  # small moves give a tiny positive IQR
    spike = order[-1]
    values[spike] = 50.0
    panel = pd.DataFrame({"administered_price": values}, index=index)
    lags = pd.Series({"administered_price": 0})
    published = panel.copy()
    old_iqr = published.quantile(0.75) - published.quantile(0.25)
    old = (published - published.median()).abs().gt(10 * old_iqr, axis="columns") & (old_iqr > 0)
    assert old.to_numpy().sum() > 20  # the all-values rule flagged ordinary moves
    _, _, _, _, _, outliers = clean_window(panel, lags, index[-1])
    flagged = outliers["administered_price"]
    assert flagged.sum() == 1
    assert flagged.iloc[spike]


def test_zero_iqr_preserves_nonzero_rounded_changes():
    index = pd.period_range("2000-01", periods=80, freq="M")
    values = np.zeros(80)
    values[-4:] = [0.1, -0.1, 0.1, -0.1]
    panel = pd.DataFrame({"rounded_change": values}, index=index)
    lags = pd.Series({"rounded_change": 0})
    published, cleaned, _, _, scale, outliers = clean_window(panel, lags, index[-1])
    pd.testing.assert_frame_equal(published, cleaned)
    assert scale.iloc[0] > 0
    assert not outliers.to_numpy().any()
