"""MSE ratios are cumulative sums on matched dates, not ratios of point errors."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.experiment import MODELS
from src.forecasting.report import evaluate, selection_tables


def test_declared_missing_cell_uses_common_dates_for_every_model():
    rows = []
    for date in ["2015-01", "2015-02", "2015-03"]:
        for model in MODELS:
            missing = date == "2015-02" and model == "AR"
            rows.append(
                {
                    "origin": date,
                    "target_month": date,
                    "target": "m0",
                    "model": model,
                    "error": np.nan if missing else 1.0,
                    "forecast": np.nan if missing else 0.0,
                    "flagged": False,
                    "status": "unavailable" if missing else "ok",
                }
            )
    data = pd.DataFrame(rows)
    metrics, _, curves = evaluate(data)
    assert metrics.n.eq(2).all()
    assert "2015-02" not in set(curves.origin)
    data.loc[data.error.isna(), "status"] = "ok"
    with pytest.raises(ValueError, match="Undocumented"):
        evaluate(data)


def test_cumulative_mse_and_zero_selection_denominators():
    rows = []
    for t, month in enumerate(["2015-01", "2015-02", "2015-03"]):
        for model in MODELS:
            error = t + 1 if model == "FarmPredict_MQ" else 2
            rows.append(
                {
                    "origin": month,
                    "target_month": month,
                    "target": "m0",
                    "model": model,
                    "error": error,
                    "forecast": -error,
                    "flagged": False,
                }
            )
    forecasts = pd.DataFrame(rows)
    _, pairs, cumulative = evaluate(forecasts)
    curve = cumulative[(cumulative["sample"] == "all") & (cumulative.benchmark == "AR")]
    np.testing.assert_allclose(curve.cumulative_MSE_ratio, [1 / 4, 5 / 8, 14 / 12])
    assert not pairs[(pairs["sample"] == "all") & (pairs.benchmark == "AR")].wins_MSE.item()
    selected = pd.DataFrame(
        [
            {
                "origin": "2015-01",
                "target": "m0",
                "model": "FarmPredict_MQ",
                "predictor": "m0",
                "term": "m0__M1",
                "frequency": "M",
                "released_position": 1,
                "coefficient": 0.2,
            },
            {
                "origin": "2015-01",
                "target": "m0",
                "model": "FarmPredict_MQ",
                "predictor": "m0",
                "term": "m0__M2",
                "frequency": "M",
                "released_position": 2,
                "coefficient": 0.3,
            },
        ]
    )
    variables, _ = selection_tables(selected, forecasts, ["m0"], ["q0"])
    assert variables.loc[variables.predictor == "m0", "selection_frequency"].item() == 1 / 3
    assert variables.loc[variables.predictor == "q0", "selection_frequency"].item() == 0
