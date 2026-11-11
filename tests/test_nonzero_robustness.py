"""Current-run fixed MQ checkpoints and release-mask invariance."""

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts import fixed_factor_mqe_nonzero as runner
from src.forecasting.dfm import MixedFrequencyEM
from src.forecasting.experiment import prepare


def test_current_mq_checkpoint_reuse_without_em_and_without_unreleased_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A fixed-factor ablation cannot refit the DFM")

    for method in ["fit", "start", "maximization"]:
        monkeypatch.setattr(MixedFrequencyEM, method, forbidden)
    meta = json.loads((runner.SOURCE / "MQ/metadata.json").read_text())
    runner.verify_saved_files(meta)
    _, settings, _, panel, info, lags, _ = prepare("config/forecast_paper.yaml", "MQE")
    baseline = pd.read_csv(runner.SOURCE / "forecasts.csv")
    baseline = baseline[baseline.information_set.eq("MQ") & baseline.model.eq("DFM")]
    origin = pd.Period("2022-04", "M")
    with threadpool_limits(limits=1):
        rows = runner.origin_forecasts(
            panel, info, lags, origin, meta["panel_members"], settings, baseline
        )
        altered = panel.copy()
        for name in panel.columns.difference(["HICPOV_EA", "HICPNG_EA", "HICPNEF_EA", "IPMN_EA"]):
            altered.loc[altered.index > origin - lags[name], name] = 1e9
        again = runner.origin_forecasts(
            altered, info, lags, origin, meta["panel_members"], settings, baseline
        )
    for first, second in zip(rows, again, strict=True):
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second))
    forecasts = pd.DataFrame(rows[0])
    np.testing.assert_allclose(
        forecasts.forecast, forecasts.common_forecast + forecasts.sparse_correction
    )
