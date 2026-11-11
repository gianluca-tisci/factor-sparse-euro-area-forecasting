"""The canonical main design must agree with the adopted execution profile."""

from pathlib import Path

import pytest
import yaml

from src.forecasting.experiment import MODELS, load_settings, starting_point


def test_paper_profile_starts_every_date_cold():
    _, settings, _ = load_settings("config/forecast_paper.yaml")
    assert settings["reestimation_start"] == "cold"  # D-71
    assert settings["classification"] == "BASELINE"


def test_cold_reestimation_discards_the_previous_fit():
    chain = object()
    assert starting_point({}, chain) is chain
    assert starting_point({"reestimation_start": "warm"}, chain) is chain
    assert starting_point({"reestimation_start": "cold"}, chain) is None
    with pytest.raises(ValueError):
        starting_point({"reestimation_start": "tepid"}, chain)


def test_main_design_matches_execution_profile():
    design = yaml.safe_load(Path("config/research_design.yaml").read_text())
    profile = yaml.safe_load(Path("config/forecast_analysis.yaml").read_text())
    assert design["research"]["forecast"]["horizons_months"] == [profile["horizon"]] == [1]
    assert design["research"]["forecast"]["first_origin"] == profile["test_start"] == "2015-01"
    assert design["research"]["forecast"]["information_timing"] == "release_calendar"
    assert design["factor_model"]["reestimation_start"] == profile["reestimation_start"] == "warm"
    assert design["factor_model"]["cold_start_method"] == profile["dfm_initialization"]
    assert set(design["factor_model"]["quarterly_panel"]) == set(profile["quarterly_series"])
    assert len(profile["quarterly_series"]) == 16
    assert design["preprocessing"]["training_outlier_policy"] == profile["training_outlier_policy"]
    assert profile["training_outlier_policy"] == "cleaned_inputs_and_labels"
    assert design["sparse_model"]["monthly_lags"] == 4
    assert design["sparse_model"]["quarterly_lags"] == 2
    assert design["sparse_model"]["cap_basis"] == "effective_complete_training_rows"
    assert design["benchmarks"]["pca_var_min_lag"] == 0
    assert set(
        design["benchmarks"]["mandatory"] + design["benchmarks"]["strategy_A_models"]
    ) == set(MODELS)
    assert len(MODELS) == 10
