"""Projection preserves release masks; checkpoint reuse never invokes EM fitting."""

import numpy as np
import pandas as pd

from src.forecasting import fixed_external
from src.forecasting.dfm import MixedFrequencyEM
from src.forecasting.experiment import clean_window


def test_external_projection_is_train_only_and_preserves_missingness():
    rng = np.random.default_rng(2026)
    index = pd.period_range("2000-01", periods=80, freq="M")
    factors = rng.normal(size=(80, 3))
    values = pd.DataFrame(
        {"external": factors @ np.array([2, -1, 3]) + rng.normal(size=80)}, index=index
    )
    values.iloc[7] = np.nan
    lags = pd.Series({"external": 2})
    origin = index[59]
    altered = values.copy()
    altered.loc[altered.index > origin - 2] = 1e9
    first = clean_window(values, lags, origin)[1]
    second = clean_window(altered, lags, origin)[1]
    residuals = fixed_external.external_residuals(first, factors[: len(first)])
    pd.testing.assert_frame_equal(
        residuals, fixed_external.external_residuals(second, factors[: len(second)])
    )
    assert residuals.isna().equals(first.isna())
    valid = residuals.external.notna()
    design = np.column_stack([np.ones(len(first)), factors[: len(first)]])
    np.testing.assert_allclose(design[valid].T @ residuals.loc[valid, "external"], 0, atol=1e-10)


def test_two_origin_smoke_without_em(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("DFM refitting is forbidden")

    monkeypatch.setattr(MixedFrequencyEM, "fit", forbidden)
    monkeypatch.setattr(MixedFrequencyEM, "start", forbidden)
    monkeypatch.setattr(MixedFrequencyEM, "maximization", forbidden)
    monkeypatch.setattr(fixed_external, "OUTPUT", tmp_path)
    fixed_external.run(limit=2)
    forecasts = pd.read_csv(tmp_path / "forecasts.csv")
    assert len(forecasts) == 8
    assert forecasts.origin.nunique() == 2
    assert np.isfinite(forecasts.forecast).all()
    np.testing.assert_allclose(
        forecasts.forecast, forecasts.common_forecast + forecasts.sparse_correction
    )
