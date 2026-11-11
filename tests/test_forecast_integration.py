"""End-to-end leakage sentinel and PCA algebra on a small synthetic panel."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.forecasting import experiment
from src.forecasting.dfm import pca_fit
from src.forecasting.experiment import run_origin


def test_pca_reconstruction_and_available_row_projection():
    rng = np.random.default_rng(15)
    f = rng.normal(size=(72, 3))
    loadings = rng.normal(size=(8, 3))
    x = f @ loadings.T
    x[-1, :3] = np.nan
    estimated_f, estimated_loadings, _ = pca_fit(x, 3)
    common = estimated_f @ estimated_loadings.T
    np.testing.assert_allclose(common[np.isfinite(x)], x[np.isfinite(x)], atol=1e-10)
    np.testing.assert_allclose(common[-1], f[-1] @ loadings.T, atol=1e-10)
    residuals = np.nan_to_num(x - common)
    np.testing.assert_allclose((common + residuals)[np.isfinite(x)], x[np.isfinite(x)], atol=1e-12)


def test_whole_forecast_pipeline_ignores_every_unpublished_value():
    rng = np.random.default_rng(23)
    index = pd.period_range("2000-02", periods=100, freq="M")
    names = [f"m{i}" for i in range(6)] + ["q0", "q1"]
    panel = pd.DataFrame(rng.normal(size=(100, 8)), index=index, columns=names)
    panel.loc[index.month % 3 != 0, ["q0", "q1"]] = np.nan
    info = pd.DataFrame({"Frequency": ["M"] * 6 + ["Q"] * 2}, index=names)
    lags = pd.Series([0, 0, 0, 1, 2, 3, 2, 2], index=names)
    origin = index[-3]
    altered = panel.copy()
    for name in panel:
        altered.loc[index > origin - int(lags[name]), name] += 10000
    design = {
        "factor_model": {
            "selected_r": 3,
            "fit_monitoring": {
                "em_cap_cold_first_fit": 150,
                "em_cap_warm_fit": 100,
                "cold_check_every_n_dates": 12,
                "cold_check_rho_gap": 0.02,
                "flag_rho": 0.98,
            },
        },
        "factor_forecast": {"max_lag": 2},
    }
    settings = {"initial_factor_variance": 10.0}
    with threadpool_limits(limits=1):
        first = run_origin(panel, info, lags, origin, design, settings)
        second = run_origin(altered, info, lags, origin, design, settings)
    a, b = pd.DataFrame(first[0]), pd.DataFrame(second[0])
    pd.testing.assert_frame_equal(
        a[["target", "model", "forecast"]], b[["target", "model", "forecast"]]
    )
    pd.testing.assert_frame_equal(pd.DataFrame(first[1]), pd.DataFrame(second[1]))
    pd.testing.assert_frame_equal(pd.DataFrame(first[2]), pd.DataFrame(second[2]))
    assert len(a) == 6 * 10
    assert a.groupby("target").origin.nunique().eq(1).all()


def test_cleaned_training_labels_preserve_raw_scoring_and_all_models(monkeypatch):
    rng = np.random.default_rng(81)
    index = pd.period_range("2000-02", periods=100, freq="M")
    names = [f"m{i}" for i in range(6)] + ["q0", "q1"]
    panel = pd.DataFrame(rng.normal(size=(100, 8)), index=index, columns=names)
    panel.loc[index.month % 3 != 0, ["q0", "q1"]] = np.nan
    panel.loc[index[40], "m0"] = 1000.0
    origin = index[-3]
    panel.loc[origin + 1, "m0"] = 1234.0
    info = pd.DataFrame({"Frequency": ["M"] * 6 + ["Q"] * 2}, index=names)
    lags = pd.Series([0, 0, 0, 1, 2, 3, 2, 2], index=names)
    design = {
        "factor_model": {
            "selected_r": 3,
            "fit_monitoring": {
                "em_cap_cold_first_fit": 150,
                "em_cap_warm_fit": 100,
                "cold_check_every_n_dates": 12,
                "cold_check_rho_gap": 0.02,
                "flag_rho": 0.98,
            },
        },
        "factor_forecast": {"max_lag": 2},
    }
    original_training_sample = experiment.training_sample
    checked = []

    def inspect_labels(features, response, target_lag, cutoff):
        if response.name == "m0":
            assert np.isnan(response.loc[index[40]])
            checked.append(True)
        return original_training_sample(features, response, target_lag, cutoff)

    monkeypatch.setattr(experiment, "training_sample", inspect_labels)
    settings = {
        "initial_factor_variance": 10.0,
        "training_outlier_policy": "cleaned_inputs_and_labels",
    }
    with threadpool_limits(limits=1):
        result = experiment.run_origin(panel, info, lags, origin, design, settings)
    forecasts = pd.DataFrame(result[0])
    assert checked
    assert forecasts.loc[forecasts.target == "m0", "actual"].eq(1234.0).all()
    assert set(forecasts.model) == {
        "AR",
        "PCR",
        "AR_PCR",
        "PCA_VAR",
        "sparse_only",
        "PCA_VAR_LASSO",
        "MF_factor_only",
        "MF_factor_own_AR",
        "FarmPredict_M",
        "FarmPredict_MQ",
    }
