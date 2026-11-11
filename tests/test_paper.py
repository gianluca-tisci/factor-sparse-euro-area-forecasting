"""Paper scope, common calendars, imported release masks and checkpoint provenance."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.ea_md_qd import available
from src.data.external import IDS, load_external
from src.forecasting import experiment
from src.forecasting.design import dictionary, target_month
from src.forecasting.paper import TARGETS, evaluate_paper, specifications


def test_exact_paper_panels_and_inherited_methods():
    for panel, count in [("M", 47), ("MQ", 63), ("ME", 71), ("MQE", 87)]:
        design, settings, _, values, info, lags, _ = experiment.prepare(
            "config/forecast_paper.yaml", panel
        )
        assert len(values.columns) == count
        assert list(values) == list(info.index) == list(lags.index)
        assert settings["targets"] == TARGETS
        assert settings["dfm_initialization"] == "balanced_t_minus_3"
        assert settings["training_outlier_policy"] == "cleaned_inputs_and_labels"
        assert design["factor_model"]["selected_r"] == 3
        assert list(info[info.Frequency == "Q"].index) == settings["quarterly_series"]
        assert set(IDS).issubset(values) == ("E" in panel)
    assert len(specifications()) == 24
    assert len(set(specifications())) == 24
    assert ("MQ", "PCR") not in specifications()
    assert [(p, m) for p, m in specifications() if m == "AR"] == [("shared", "AR")]


def test_external_release_masks_and_future_sentinel():
    panel, manifest = load_external()
    lags = pd.Series({k: v["release_lag_months"] for k, v in manifest["series"].items()})
    origin = pd.Period("2015-01", "M")
    published = available(panel, lags, origin)
    altered = panel.copy()
    for name in panel:
        altered.loc[altered.index > origin - int(lags[name]), name] = 1e9
        assert published[name].dropna().index.max() == origin - int(lags[name])
    pd.testing.assert_frame_equal(published, available(altered, lags, origin))
    _, references = dictionary(published, pd.Series("M", index=panel.columns), lags, [origin])
    for name in panel:
        assert references.loc[origin, name + "__M1"] == origin - int(lags[name])
    assert target_month(origin, 2) == pd.Period("2014-12", "M")


def fake_forecasts():
    rows = []
    for target in TARGETS:
        for origin in ["2019-12", "2020-01", "2021-01", "2026-01"]:
            for i, (panel, model) in enumerate(specifications()):
                rows.append(
                    {
                        "target": target,
                        "origin": origin,
                        "target_month": origin,
                        "information_set": panel,
                        "model": model,
                        "actual": 0.0,
                        "forecast": float(i + 1),
                        "error": -float(i + 1),
                        "valid": True,
                        "flagged": False,
                    }
                )
    return pd.DataFrame(rows)


def test_scoring_intersections_na_failures_and_reference_year():
    forecasts = fake_forecasts()
    mask = (
        (forecasts.target == TARGETS[0])
        & (forecasts.origin == "2019-12")
        & (forecasts.model == "FarmPredict")
        & (forecasts.information_set == "MQE")
    )
    forecasts.loc[mask, "valid"] = False
    scores, pairs, coverage, interaction = evaluate_paper(forecasts)
    main = scores[(scores.target == TARGETS[0]) & (scores["sample"] == "primary")]
    assert main.n.eq(1).all()
    np.testing.assert_allclose(main.RMSE**2, main.MSE)
    assert not ((scores.model == "PCR") & scores.information_set.isin(["MQ", "MQE"])).any()
    assert (pairs.n > 1).any()  # Pairwise sample kept separate from global ranking.
    assert coverage.n_failed.max() == 1
    assert len(interaction) > 0
    assert (
        pairs.loc[pairs["sample"] == "primary", "pvalue_two_sided"].isna().all()
    )  # Constant synthetic loss differences.


def test_resume_rejects_changed_source_and_reuses_complete_checkpoint(tmp_path, monkeypatch):
    # Reuse the real panel/setup, substitute just the expensive fit.
    calls = []

    def fit(panel, info, lags, origin, design, settings, previous, ordinal):
        calls.append(origin)
        from src.forecasting.dfm import Parameters

        n = len(panel.columns)
        parameters = Parameters(np.ones((n, 3)), np.ones(n), np.zeros((3, 6)), np.eye(3))
        return (
            [],
            [],
            [],
            {"seconds": 0.0, "iterations": 0, "rho": 0.0, "flagged": False},
            (parameters, pd.Series(1.0, index=panel.columns), pd.Series(0.0, index=panel.columns)),
        )

    monkeypatch.setattr(experiment, "run_origin", fit)
    experiment.run("config/forecast_paper.yaml", tmp_path, limit=1, information_set="M")
    experiment.run(
        "config/forecast_paper.yaml", tmp_path, limit=1, information_set="M", resume=True
    )
    assert len(calls) == 1
    path = tmp_path / "metadata.json"
    meta = json.loads(path.read_text())
    meta["source_hashes"]["src/data/external.py"] = "wrong"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenance mismatch"):
        experiment.run(
            "config/forecast_paper.yaml", tmp_path, limit=1, information_set="M", resume=True
        )


def test_paper_target_subset_reuses_one_fit_and_preserves_upstream_forecasts(monkeypatch):
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(2026)
    index = pd.period_range("2000-02", periods=85, freq="M")
    names = [f"m{i}" for i in range(6)] + ["q0"]
    panel = pd.DataFrame(rng.normal(size=(85, 7)), index=index, columns=names)
    panel.loc[index.month % 3 != 0, "q0"] = np.nan
    info = pd.DataFrame({"Frequency": ["M"] * 6 + ["Q"]}, index=names)
    lags = pd.Series([0, 0, 0, 1, 2, 3, 2], index=names)
    design = {
        "factor_model": {
            "selected_r": 3,
            "fit_monitoring": {
                "em_cap_cold_first_fit": 100,
                "em_cap_warm_fit": 100,
                "cold_check_every_n_dates": 12,
                "cold_check_rho_gap": 0.02,
                "flag_rho": 0.98,
            },
        },
        "factor_forecast": {"max_lag": 2},
    }
    settings = {
        "initial_factor_variance": 10.0,
        "dfm_initialization": "balanced_t_minus_3",
        "training_outlier_policy": "cleaned_inputs_and_labels",
    }
    origin = index[-3]
    with threadpool_limits(limits=1):
        original = experiment.run_origin(panel, info, lags, origin, design, settings)
        calls = []
        fit = experiment.BalancedStartEM.fit

        def counted(self, *args, **kwargs):
            calls.append(1)
            return fit(self, *args, **kwargs)

        monkeypatch.setattr(experiment.BalancedStartEM, "fit", counted)
        paper_settings = {
            **settings,
            "information_set": "MQ",
            "targets": ["m0", "m4"],
            "models": ["MF_factor_only", "MF_factor_own_AR", "sparse_only", "FarmPredict_MQ"],
        }
        paper = experiment.run_origin(panel, info, lags, origin, design, paper_settings)
        assert len(calls) == 1
        altered = panel.copy()
        for name in panel:
            altered.loc[index > origin - int(lags[name]), name] = 1e8
        future = experiment.run_origin(altered, info, lags, origin, design, paper_settings)
    full = pd.DataFrame(original[0])
    subset = full[
        full.target.isin(paper_settings["targets"]) & full.model.isin(paper_settings["models"])
    ]
    actual = pd.DataFrame(paper[0])
    keys = ["target", "model"]
    np.testing.assert_allclose(
        subset.sort_values(keys).forecast, actual.sort_values(keys).forecast, rtol=1e-12
    )
    np.testing.assert_allclose(actual.forecast, pd.DataFrame(future[0]).forecast, rtol=1e-12)
    assert paper[3]["pca_var_order"] is None


def test_calendar_hac_preserves_contiguous_results_and_does_not_compress_gaps():
    from scipy.stats import norm

    from src.forecasting.report import loss_test

    losses = np.array([1.0, 4.0, -2.0, 3.0, 8.0, -1.0])
    dates = pd.period_range("2019-10", periods=6, freq="M")
    np.testing.assert_allclose(loss_test(losses), loss_test(losses, dates))
    gapped = pd.PeriodIndex(
        ["2019-10", "2019-11", "2019-12", "2021-01", "2021-02", "2021-03"], freq="M"
    )
    statistic, pvalue, bandwidth = loss_test(losses, gapped)
    centered = losses - losses.mean()
    variance = centered @ centered / len(losses)
    for lag in range(1, bandwidth + 1):
        products = sum(
            centered[i] * centered[j]
            for i in range(6)
            for j in range(i)
            if (gapped[i] - gapped[j]).n == lag
        )
        variance += 2 * (1 - lag / (bandwidth + 1)) * products / len(losses)
    expected = losses.mean() / np.sqrt(variance / len(losses))
    assert statistic == pytest.approx(expected)
    assert pvalue == pytest.approx(2 * norm.sf(abs(expected)))
    assert statistic != pytest.approx(loss_test(losses)[0])


def test_failed_origin_is_recorded_and_retried(tmp_path, monkeypatch):
    attempts = []

    def failed_fit(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("synthetic numerical failure")

    monkeypatch.setattr(experiment, "run_origin", failed_fit)
    experiment.run("config/forecast_paper.yaml", tmp_path, limit=1, information_set="M")
    frame = pd.read_csv(tmp_path / "2015-01" / "forecasts.csv")
    assert len(frame) == 36
    assert frame.forecast.isna().all()
    assert frame.failure_reason.str.contains("synthetic numerical failure").all()
    assert not (tmp_path / "2015-01" / "parameters.npz").exists()
    experiment.run(
        "config/forecast_paper.yaml", tmp_path, limit=1, information_set="M", resume=True
    )
    assert len(attempts) == 2


def test_all_failed_scores_remain_missing_not_zero():
    forecasts = fake_forecasts()
    forecasts["valid"] = False
    scores, pairs, coverage, _ = evaluate_paper(forecasts)
    assert scores.MSE.isna().all()
    assert scores.n.eq(0).all()
    assert pairs.pvalue_two_sided.isna().all()
    assert coverage.n_common.eq(0).all()
