"""Extend D-70 to saved MQ factors; external series enter only as residuals."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.forecasting.design import dictionary, sparse_predict, target_month, training_sample
from src.forecasting.dfm import Fit, MixedFrequencyEM, Parameters
from src.forecasting.experiment import clean_window, prepare
from src.forecasting.fixed_external import external_residuals
from src.forecasting.paper import TARGETS

SOURCE = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/nonzero_robustness/mqe_ablation")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_saved_files(meta):
    """Allow only verified LF/CRLF differences from the saved Linux run."""
    newline_only = []
    for name, expected in {**meta["input_hashes"], **meta["source_hashes"]}.items():
        value = Path(name).read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            if hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest() != expected:
                raise ValueError(f"Saved run content changed: {name}")
            newline_only.append(name)
    return newline_only


def origin_forecasts(panel, info, lags, origin, members, settings, baseline):
    """One fixed-parameter E-step: preserve MQ monthly/quarterly measurement model."""
    _, cleaned, _, mean, scale, _ = clean_window(panel, lags, origin)
    nm = int(info.loc[members, "Frequency"].eq("M").sum())
    assert nm == 47 and len(members) == 63
    with np.load(SOURCE / "MQ" / str(origin) / "parameters.npz") as saved:
        np.testing.assert_allclose(mean[members], saved["mean"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scale[members], saved["scale"], rtol=1e-12, atol=1e-12)
        z = (cleaned[members] - saved["mean"]) / saved["scale"]
        params = Parameters(
            *(saved[k].copy() for k in ["loadings", "variances", "ar", "innovation"])
        )
    model = MixedFrequencyEM(z, nm, 3, settings["initial_factor_variance"])
    posterior = model.expectation(params)
    fit = Fit(model, params, posterior, 0, True, [])
    residuals = cleaned[members] - (fit.common() * scale[members] + mean[members])
    external = panel.columns.difference(members, sort=False)
    assert len(external) == 24
    factors = posterior.mean.reshape(-1, model.r)[4:]
    residuals = pd.concat([residuals, external_residuals(cleaned[external], factors)], axis=1)
    features, _ = dictionary(residuals, info.Frequency, lags, z.index)
    records, selections, tuning = [], [], []
    for target in TARGETS:
        saved = baseline[(baseline.origin == str(origin)) & (baseline.target == target)].iloc[0]
        date = target_month(origin, lags[target])
        reconstructed = fit.common_at(date, target) * scale[target] + mean[target]
        np.testing.assert_allclose(reconstructed, saved.forecast, rtol=1e-9, atol=1e-9)
        assert str(date) == saved.target_month
        np.testing.assert_allclose(panel.loc[date, target], saved.actual, rtol=1e-12)
        x, y = training_sample(features, residuals[target], lags[target], origin)
        correction, selected, detail = sparse_predict(x, y, features.loc[origin])
        prediction = float(saved.forecast + correction)
        records.append(
            {
                "origin": str(origin),
                "target": target,
                "target_month": str(date),
                "model": "Fixed_DFM_MQ_external_sparse",
                "actual": saved.actual,
                "common_forecast": saved.forecast,
                "sparse_correction": correction,
                "forecast": prediction,
                "error": float(saved.actual - prediction),
            }
        )
        tuning.append({"origin": str(origin), "target": target, **detail})
        for term, coefficient in selected.items():
            selections.append(
                {
                    "origin": str(origin),
                    "target": target,
                    "term": term,
                    "predictor": term.rsplit("__", 1)[0],
                    "coefficient": coefficient,
                }
            )
    return records, selections, tuning


def main():
    if OUT.exists():
        raise FileExistsError("Use a new run directory; saved results are immutable")
    meta = json.loads((SOURCE / "MQ/metadata.json").read_text())
    newline_only = verify_saved_files(meta)
    _, settings, _, panel, info, lags, _ = prepare("config/forecast_paper.yaml", "MQE")
    assert meta["completion"] == "COMPLETE"
    assert meta["profile"]["reestimation_start"] == "cold"
    baseline = pd.read_csv(SOURCE / "forecasts.csv")
    baseline = baseline[baseline.information_set.eq("MQ") & baseline.model.eq("DFM")]
    paths = [
        SOURCE / "forecasts.csv",
        SOURCE / "MQ/metadata.json",
        Path("scripts/fixed_factor_mqe_nonzero.py"),
    ]
    paths += [SOURCE / "MQ" / origin / "parameters.npz" for origin in meta["origins"]]
    paths += [Path(p) for p in {**meta["input_hashes"], **meta["source_hashes"]}]
    hashes = {p.as_posix(): digest(p) for p in paths}
    all_rows = [[], [], []]
    with threadpool_limits(limits=1):
        for i, origin in enumerate(meta["origins"]):
            rows = origin_forecasts(
                panel, info, lags, pd.Period(origin, "M"), meta["panel_members"], settings, baseline
            )
            for total, current in zip(all_rows, rows, strict=True):
                total.extend(current)
            print(f"[{i + 1}/{len(meta['origins'])}] {origin}: MQ fixed sparse stage", flush=True)
    for path, expected in hashes.items():
        assert digest(path) == expected, path
    assert len(all_rows[0]) == 4 * len(meta["origins"])
    OUT.mkdir(parents=True)
    for name, rows in zip(["forecasts", "selected_predictors", "tuning"], all_rows, strict=True):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "classification": "ALTERNATIVE",
                "base_factors": "MQ",
                "sparse_dictionary": "M+Q+E residuals",
                "completed_origins": len(meta["origins"]),
                "dfm_refits": 0,
                "source_hashes": hashes,
                "verified_newline_only_differences": newline_only,
                "projection": "Released cleaned E projected with intercept on fixed MQ factors at each origin; missingness retained",
                "profile": "config/forecast_paper.yaml",
                "raw_external_predictors": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
