"""Fixed DFM-M ablation: reuse checkpoints, project E, run existing sparse stage."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import _style
from src.forecasting.design import dictionary, sparse_predict, target_month, training_sample
from src.forecasting.dfm import Fit, MixedFrequencyEM, Parameters
from src.forecasting.experiment import clean_window, prepare
from src.forecasting.paper import TARGETS

MODEL = "Fixed_DFM_M_external_sparse"
# D-72 branch: the ablation reads the rerun with the nonzero-IQR outlier rule.
SOURCE = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUTPUT = Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr")


def external_residuals(cleaned, factors):
    """OLS projection with intercept, per series, on released/cleaned rows only.

    Factors are smoothed ONLY within the outer origin, following D-41. Missing
    observations stay missing. No interpolation or additional lag menu is used.
    """
    x = np.column_stack([np.ones(len(factors)), factors])
    residuals = cleaned.copy()
    for name in cleaned:
        valid = cleaned[name].notna().to_numpy()
        coefficients, _, rank, _ = np.linalg.lstsq(x[valid], cleaned.loc[valid, name], rcond=None)
        if rank != x.shape[1]:
            raise ValueError(f"External projection lacks rank: {name}")
        residuals.loc[valid, name] -= x[valid] @ coefficients
    return residuals


def origin_forecasts(panel, info, lags, origin, monthly, settings, baseline):
    """Exactly one fixed-parameter E-step and one shared residual dictionary."""
    _, cleaned, _, mean, scale, _ = clean_window(panel, lags, origin)
    with np.load(SOURCE / "M" / str(origin) / "parameters.npz") as saved:
        np.testing.assert_allclose(mean[monthly], saved["mean"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scale[monthly], saved["scale"], rtol=1e-12, atol=1e-12)
        z = (cleaned[monthly] - saved["mean"]) / saved["scale"]
        params = Parameters(
            *(saved[k].copy() for k in ["loadings", "variances", "ar", "innovation"])
        )
    model = MixedFrequencyEM(z, len(monthly), 3, settings["initial_factor_variance"])
    posterior = model.expectation(params)  # Never call fit(), initialize(), or any M-step.
    fit = Fit(model, params, posterior, 0, True, [])
    common = fit.common() * scale[monthly] + mean[monthly]
    residuals = cleaned[monthly] - common
    external = panel.columns.difference(monthly, sort=False)
    factors = posterior.mean.reshape(-1, model.r)[4:]
    residuals = pd.concat([residuals, external_residuals(cleaned[external], factors)], axis=1)
    features, _ = dictionary(residuals, info.Frequency, lags, z.index)
    records, selections, tuning = [], [], []
    for target in TARGETS:
        saved = baseline[(baseline.origin == str(origin)) & (baseline.target == target)].iloc[0]
        date = target_month(origin, lags[target])
        reconstructed = fit.common_at(date, target) * scale[target] + mean[target]
        np.testing.assert_allclose(reconstructed, saved.forecast, rtol=1e-9, atol=1e-9)
        if str(date) != saved.target_month or panel.loc[date, target] != saved.actual:
            # CSV floating-point round trips need a numerical comparison.
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
                "model": MODEL,
                "actual": saved.actual,
                "common_forecast": saved.forecast,
                "sparse_correction": correction,
                "forecast": prediction,
                "error": float(saved.actual - prediction),
            }
        )
        tuning.append(dict(origin=str(origin), target=target, **detail))
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


def report(output, forecasts, selected, external):
    """Retain the original primary global matched dates, excluding target year 2020."""
    coverage = pd.read_csv(SOURCE / "coverage.csv")
    original = pd.read_csv(SOURCE / "forecasts.csv")
    scores, improvements, frequencies = [], [], []
    specifications = [
        ("M", "DFM", "DFM-M"),
        ("M", "FarmPredict", "FarmPredict-M"),
        ("ME", "FarmPredict", "FarmPredict-ME"),
        ("ME", "LASSO", "LASSO-ME"),
    ]
    for target in TARGETS:
        dates = coverage[(coverage.target == target) & (coverage["sample"] == "primary")]
        dates = set(dates.iloc[0].common_dates.split(";"))
        new = forecasts[(forecasts.target == target) & forecasts.origin.isin(dates)]
        if set(new.origin) != dates:
            raise ValueError(f"Incomplete original matched evaluation dates: {target}")
        losses = {}
        for panel, model, label in specifications:
            rows = original[
                (original.target == target)
                & original.origin.isin(dates)
                & (original.information_set == panel)
                & (original.model == model)
            ]
            assert set(rows.origin) == dates and len(rows) == len(new)
            aligned = rows.set_index("origin").sort_index()
            check = new.set_index("origin").sort_index()
            np.testing.assert_allclose(aligned.actual, check.actual, rtol=1e-12)
            assert aligned.target_month.equals(check.target_month)
            losses[label] = float(np.mean(rows.error**2))
        losses[MODEL] = float(np.mean(new.error**2))
        for label, mse in losses.items():
            scores.append(
                {"target": target, "model": label, "n": len(new), "MSE": mse, "RMSE": np.sqrt(mse)}
            )
        for label in ["FarmPredict-M", "FarmPredict-ME"]:
            improvements.append(
                {
                    "target": target,
                    "benchmark": label,
                    "MSE_improvement_pct": 100 * (1 - losses[MODEL] / losses[label]),
                    "RMSE_improvement_pct": 100 * (1 - np.sqrt(losses[MODEL] / losses[label])),
                }
            )
        subset = selected[(selected.target == target) & selected.origin.isin(dates)]
        counts = subset.drop_duplicates(["origin", "predictor"]).predictor.value_counts()
        for name in external:
            frequencies.append(
                {
                    "target": target,
                    "predictor": name,
                    "selected_origins": int(counts.get(name, 0)),
                    "n_origins": len(new),
                    "selection_pct": 100 * counts.get(name, 0) / len(new),
                }
            )
    tables = {
        "comparison": pd.DataFrame(scores),
        "improvement": pd.DataFrame(improvements),
        "external_selection": pd.DataFrame(frequencies),
    }
    old_out = _style.OUT
    _style.OUT = output
    try:
        for number, (name, frame) in enumerate(tables.items(), 80):
            frame.to_csv(output / f"{name}.csv", index=False)
            display = frame
            if name == "external_selection":
                display = frame.pivot(
                    index="predictor", columns="target", values="selection_pct"
                ).reset_index()
            _style.save_table(
                display,
                f"tab_{number}_{name}",
                index=False,
                notes="Original matched primary dates; target months through December 2025, "
                "excluding 2020. Fixed saved DFM-M; original modified-BIC LASSO. "
                "Selection counts any released lag once per origin. Source: saved "
                "paper experiment and fixed-DFM external sparse ablation.",
            )
    finally:
        _style.OUT = old_out


def run(limit=None):
    started = time.perf_counter()
    _, settings, hashes, panel, info, lags, _ = prepare("config/forecast_paper.yaml", "ME")
    meta = json.loads((SOURCE / "M" / "metadata.json").read_text())
    if hashes != meta["input_hashes"]:
        raise ValueError("Input hashes differ from saved DFM experiment")
    for path, digest in meta["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Saved experiment implementation changed: {path}")
    monthly = meta["panel_members"]
    if len(monthly) != 47 or len(panel.columns.difference(monthly)) != 24:
        raise ValueError("Expected 47 M and 24 E series")
    origins = meta["origins"][:limit]
    for origin in origins:
        if not (SOURCE / "M" / origin / "parameters.npz").exists():
            raise FileNotFoundError(f"Missing DFM-M checkpoint at {origin}")
    baseline = pd.read_csv(SOURCE / "forecasts.csv")
    baseline = baseline[(baseline.information_set == "M") & (baseline.model == "DFM")]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / f"{name}.csv" for name in ["forecasts", "selected_predictors", "tuning"]]
    frames = [pd.read_csv(path) if path.exists() else pd.DataFrame() for path in paths]
    done = set(frames[0].origin) if len(frames[0]) else set()
    with threadpool_limits(limits=1):
        for ordinal, origin in enumerate(origins):
            if origin in done:
                assert set(frames[0].loc[frames[0].origin == origin, "target"]) == set(TARGETS)
                continue
            rows = origin_forecasts(
                panel, info, lags, pd.Period(origin, "M"), monthly, settings, baseline
            )
            for i, records in enumerate(rows):
                frames[i] = pd.concat([frames[i], pd.DataFrame(records)], ignore_index=True)
            print(f"[{ordinal + 1}/{len(origins)}] {origin}: sparse stage complete", flush=True)
    for frame, path in zip(frames, paths, strict=True):
        frame.to_csv(path, index=False)
    packages = {p: importlib.metadata.version(p) for p in meta["packages"]}
    manifest = {
        "classification": "ALTERNATIVE",
        "completed_origins": len(frames[0]) // 4,
        "dfm_refits": 0,
        "source": str(SOURCE),
        "profile": "config/forecast_paper.yaml",
        "input_hashes": hashes,
        "python": platform.python_version(),
        "packages": packages,
        "original_packages": meta["packages"],
        "source_forecasts_sha256": hashlib.sha256(
            (SOURCE / "forecasts.csv").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
        "projection": "Per-origin OLS with intercept on released cleaned E and fixed M factors; "
        "missing values preserved; outer-window smoothing (D-41).",
        "checkpoint_hashes": {
            o: hashlib.sha256((SOURCE / "M" / o / "parameters.npz").read_bytes()).hexdigest()
            for o in origins
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if limit is None:
        report(OUTPUT, frames[0], frames[1], panel.columns.difference(monthly, sort=False))
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    SOURCE, OUTPUT = args.source, args.output
    print(run(args.limit))
