"""Paper scope and matched-calendar reporting over the existing estimator runner."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import _style
from src.forecasting.experiment import run
from src.forecasting.report import loss_test

PANELS = ["M", "MQ", "ME", "MQE"]
MODELS = ["RW", "AR", "PCR", "PCA_VAR", "PCA_VAR_LASSO", "DFM", "DFM_AR", "LASSO", "FarmPredict"]
ALIASES = {
    "MF_factor_only": "DFM",
    "MF_factor_own_AR": "DFM_AR",
    "sparse_only": "LASSO",
    "FarmPredict_MQ": "FarmPredict",
}
TARGETS = ["HICPOV_EA", "HICPNG_EA", "HICPNEF_EA", "IPMN_EA"]


def specifications():
    return [("shared", m) for m in ["RW", "AR"]] + [
        (p, m)
        for p in PANELS
        for m in MODELS[2:]
        if p in ["M", "ME"] or m not in ["PCR", "PCA_VAR", "PCA_VAR_LASSO"]
    ]


def run_paper(output, limit=None, end=None, resume=False, profile="config/forecast_paper.yaml"):
    for panel in PANELS:
        run(
            profile,
            str(Path(output) / panel),
            limit=limit,
            end=end,
            resume=resume,
            information_set=panel,
        )
    return report_paper(output)


def evaluate_paper(forecasts):
    """Global intersections for rankings; separately matched pairs for contrasts."""
    scores, pairs, coverage, interactions = [], [], [], []
    expected = [f"{p}:{m}" for p, m in specifications()]
    for target in TARGETS:
        data = forecasts[forecasts.target == target].copy()
        data["spec"] = data.information_set + ":" + data.model
        if data.duplicated(["origin", "spec"]).any():
            raise ValueError("Duplicate forecast specifications")
        if (data.groupby("origin").actual.nunique() > 1).any() or (
            data.groupby("origin").target_month.nunique() > 1
        ).any():
            raise ValueError("Inconsistent target values or reference months")
        for sample in ["primary", "including_2020", "broader", "unflagged_primary"]:
            subset = data.copy()
            if sample != "broader":
                subset = subset[subset.target_month <= "2025-12"]
            if sample in ["primary", "unflagged_primary"]:
                subset = subset[~subset.target_month.str.startswith("2020-")]
            if sample == "unflagged_primary":
                subset.loc[subset.flagged, "valid"] = False
            subset.loc[~subset.valid, ["error", "forecast"]] = np.nan
            errors = (
                subset.pivot(index="origin", columns="spec", values="error")
                .reindex(columns=expected)
                .sort_index()
            )
            predictions = (
                subset.pivot(index="origin", columns="spec", values="forecast")
                .reindex(columns=expected)
                .sort_index()
            )
            common = errors.dropna()
            for spec in expected:
                panel, model = spec.split(":")
                coverage.append(
                    {
                        "target": target,
                        "sample": sample,
                        "information_set": panel,
                        "model": model,
                        "n_eligible": len(errors),
                        "n_valid": int(errors[spec].notna().sum()),
                        "n_failed": int(errors[spec].isna().sum()),
                        "n_common": len(common),
                        "common_dates": ";".join(common.index),
                        "restrictive_intersection": len(common) < len(errors),
                    }
                )
                mse = (common[spec] ** 2).mean()
                scores.append(
                    {
                        "target": target,
                        "sample": sample,
                        "information_set": panel,
                        "model": model,
                        "n": len(common),
                        "MSE": mse,
                        "RMSE": np.sqrt(mse),
                        "relative_MSE_AR": mse / (common["shared:AR"] ** 2).mean(),
                        "relative_MSE_RW": mse / (common["shared:RW"] ** 2).mean(),
                    }
                )
            contrasts = []
            for panel in PANELS:
                supported = [m for p, m in specifications() if p == panel]
                for model in supported:
                    contrasts.extend(
                        [
                            (f"{panel}:{model}", f"shared:{base}", False, "model")
                            for base in ["AR", "RW"]
                        ]
                    )
                for a, b, nested in [
                    ("FarmPredict", "DFM", True),
                    ("FarmPredict", "LASSO", False),
                    ("DFM_AR", "DFM", True),
                    ("PCA_VAR_LASSO", "PCA_VAR", True),
                ]:
                    if a in supported and b in supported:
                        contrasts.append((f"{panel}:{a}", f"{panel}:{b}", nested, "model"))
            contrasts.append(("shared:AR", "shared:RW", False, "model"))
            for model in MODELS[2:]:
                for a, b in [("MQ", "M"), ("ME", "M"), ("MQE", "MQ"), ("MQE", "ME")]:
                    if f"{a}:{model}" in expected and f"{b}:{model}" in expected:
                        contrasts.append((f"{a}:{model}", f"{b}:{model}", False, "information_set"))
                cols = [f"{p}:{model}" for p in PANELS]
                if set(cols).issubset(expected):
                    matched = errors[cols].dropna() ** 2
                    value = (
                        (matched[cols[3]] - matched[cols[1]])
                        - (matched[cols[2]] - matched[cols[0]])
                    ).mean()
                    interactions.append(
                        {
                            "target": target,
                            "sample": sample,
                            "model": model,
                            "n": len(matched),
                            "MSE_interaction": value,
                            "dates": ";".join(matched.index),
                        }
                    )
            for augmented, base, nested, kind in contrasts:
                matched = errors[[augmented, base]].dropna()
                n = len(matched)
                mse = (matched**2).mean()
                loss = matched[base] ** 2 - matched[augmented] ** 2
                if nested:
                    loss += (
                        predictions.loc[matched.index, base]
                        - predictions.loc[matched.index, augmented]
                    ) ** 2
                stat, pvalue, band = loss_test(loss, dates=matched.index)
                pairs.append(
                    {
                        "target": target,
                        "sample": sample,
                        "kind": kind,
                        "augmented": augmented,
                        "baseline": base,
                        "n": n,
                        "MSE_difference": mse[augmented] - mse[base],
                        "MSE_ratio": mse[augmented] / mse[base],
                        "RMSE_augmented": np.sqrt(mse[augmented]),
                        "RMSE_baseline": np.sqrt(mse[base]),
                        "test": "Clark_West" if nested else "Diebold_Mariano",
                        "statistic": stat,
                        "pvalue_two_sided": pvalue,
                        "HAC_bandwidth": band,
                        "test_status": "computed_calendar_HAC"
                        if np.isfinite(pvalue)
                        else "not_computed_insufficient_or_degenerate_losses",
                        "dates": ";".join(matched.index),
                    }
                )
    return tuple(pd.DataFrame(x) for x in [scores, pairs, coverage, interactions])


def report_paper(output):
    output = Path(output)
    rows, selected, tuning, fits, metadata = [], [], [], [], {}
    for panel in PANELS:
        root = output / panel
        meta = json.loads((root / "metadata.json").read_text())
        if meta["completion"] != "COMPLETE":
            raise ValueError(f"Incomplete panel: {panel}")
        metadata[panel] = meta
        for origin in meta["origins"]:
            folder = root / origin
            frame = pd.read_csv(folder / "forecasts.csv")
            frame["model"] = frame.model.replace(ALIASES)
            frame["information_set"] = np.where(frame.model.isin(["RW", "AR"]), "shared", panel)
            frame["valid"] = np.isfinite(frame.forecast) & np.isfinite(frame.actual)
            if "failure_reason" not in frame:
                frame["failure_reason"] = ""
            rows.append(frame)
            for name, dest in [("selected", selected), ("tuning", tuning)]:
                f = pd.read_csv(folder / f"{name}.csv")
                f["model"] = f.model.replace(ALIASES)
                f["information_set"] = np.where(f.model.isin(["RW", "AR"]), "shared", panel)
                dest.append(f)
            fits.append(
                dict(information_set=panel, **json.loads((folder / "fit.json").read_text()))
            )
    forecasts = pd.concat(rows, ignore_index=True)
    expected = set(specifications())
    for _, group in forecasts.groupby(["origin", "target"]):
        if set(zip(group.information_set, group.model)) != expected:
            raise ValueError("Missing or extra model-panel cells")
    forecasts["validity_status"] = np.where(forecasts.valid, "valid", "failed")
    forecasts["failure_reason"] = forecasts.failure_reason.fillna("")
    forecasts.to_csv(output / "forecasts.csv", index=False)
    chosen = pd.concat(selected, ignore_index=True)
    if not chosen.empty:
        origins = pd.PeriodIndex(chosen.origin, freq="M")
        delays = np.array(
            [
                metadata[row.information_set]["release_lags"][row.predictor]
                for row in chosen.itertuples()
            ],
            dtype=int,
        )
        latest = origins - delays
        position = chosen.released_position.astype(int).to_numpy() - 1
        offsets = np.where(chosen.frequency.eq("Q"), latest.month % 3 + 3 * position, position)
        references = latest - offsets
        if ((references + delays) > origins).any():
            raise AssertionError("Unreleased selected predictor")
        chosen["reference_month"] = references.astype(str)
        chosen["assumed_available_month"] = (references + delays).astype(str)
        chosen["estimate_cutoff"] = chosen.origin
    chosen.to_csv(output / "selected_predictors.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(output / "tuning.csv", index=False)
    availability = []
    for fit in fits:
        for name, values in fit.pop("data_availability", {}).items():
            availability.append(
                {
                    "origin": fit["origin"],
                    "information_set": fit["information_set"],
                    "series": name,
                    **values,
                }
            )
    pd.DataFrame(availability).to_csv(output / "data_availability.csv", index=False)
    pd.DataFrame(fits).to_csv(output / "fit_monitoring.csv", index=False)
    scores, pairs, coverage, interactions = evaluate_paper(forecasts)
    for frame, name in [
        (scores, "scores"),
        (pairs, "paired_comparisons"),
        (coverage, "coverage"),
        (interactions, "information_interactions"),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    scores[scores["sample"] != "primary"].to_csv(output / "robustness.csv", index=False)
    pairs.to_csv(output / "statistical_tests.csv", index=False)
    counts = forecasts[forecasts.valid].groupby(["target", "information_set", "model"]).size()
    frequencies = (
        chosen.drop_duplicates(["origin", "target", "information_set", "model", "predictor"])
        .groupby(["target", "information_set", "model", "predictor"])
        .size()
        .rename("n_selected")
        .reset_index()
    )
    # Include never-selected predictors with zero counts, including zero-support fits.
    full_index = pd.MultiIndex.from_tuples(
        [
            (target, panel, model, predictor)
            for target in TARGETS
            for panel in PANELS
            for model in ["FarmPredict", "LASSO", "PCA_VAR_LASSO"]
            if (panel, model) in specifications()
            for predictor in metadata[panel]["panel_members"]
        ],
        names=["target", "information_set", "model", "predictor"],
    )
    frequencies = (
        frequencies.set_index(list(full_index.names))
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    frequencies["n_origins"] = [
        counts.get((r.target, r.information_set, r.model), 0) for r in frequencies.itertuples()
    ]
    frequencies["selection_frequency"] = frequencies.n_selected / frequencies.n_origins.replace(
        0, np.nan
    )
    frequencies.to_csv(output / "selection_frequencies.csv", index=False)
    for sample in scores["sample"].unique():
        destination = output / "samples" / sample
        destination.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            group = scores[(scores.target == target) & (scores["sample"] == sample)]
            for metric in ["MSE", "RMSE", "n", "relative_MSE_AR", "relative_MSE_RW"]:
                table = group.pivot(
                    index="model", columns="information_set", values=metric
                ).reindex(index=MODELS, columns=PANELS)
                for baseline in ["RW", "AR"]:
                    table.loc[baseline] = group.loc[group.model == baseline, metric].iloc[0]
                table.to_csv(destination / f"{metric}_{target}.csv", na_rep="NA")
    original = _style.OUT
    _style.OUT = output / "exhibits"
    try:
        for t, target in enumerate(TARGETS):
            main = scores[(scores.target == target) & (scores["sample"] == "primary")]
            for metric in ["MSE", "RMSE"]:
                matrix = main.pivot(
                    index="model", columns="information_set", values=metric
                ).reindex(index=MODELS, columns=PANELS)
                for model in ["RW", "AR"]:
                    matrix.loc[model] = main.loc[main.model == model, metric].iloc[0]
                matrix.to_csv(output / f"{metric.lower()}_{target}.csv", na_rep="NA")
                _style.save_table(
                    matrix.fillna("NA"),
                    f"tab_{70 + t * 2 + (metric == 'RMSE')}_{metric.lower()}_{target}",
                    notes="Proposed paper experiment; common dates across 24 specifications. Target months through 2025-12 excluding 2020. NA denotes non-applicability or unavailable score; see coverage.csv. Smoke runs are not empirical evidence. Source: EA-MD-QD and archived external providers.",
                    float_format="%.5g",
                )
            figure, ax = plt.subplots(figsize=(9, 5))
            table = main.pivot(
                index="model", columns="information_set", values="relative_MSE_AR"
            ).reindex(index=MODELS[2:], columns=PANELS)
            table.plot.bar(ax=ax, color=_style.PALETTE_MANY[:4])
            ax.axhline(1, color="black", linestyle=":")
            ax.set_title(target + " — MSE relative to AR")
            ax.set_ylabel("Relative MSE")
            figure.text(
                0.01,
                0.01,
                "Notes: common dates; 2020 excluded; proposed experiment. Source: EA-MD-QD and external snapshot.",
                fontsize=7,
            )
            figure.tight_layout(rect=(0, 0.05, 1, 1))
            _style.save_fig(figure, f"fig_{70 + t}_paper_{target}", formats=["png", "pdf"])
            plt.close(figure)
        _style.record_number("PaperDistinctSpecifications", len(specifications()))
        _style.record_number("PaperForecastRows", len(forecasts))
    finally:
        _style.OUT = original
    manifest = {
        "report_source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("src/forecasting/paper.py"),
                Path("src/forecasting/report.py"),
                Path("src/_style.py"),
            ]
        },
        "status": "SMOKE_OR_LIMITED"
        if len(metadata["M"]["origins"]) < 100
        else "COMPUTED_NOT_TEAM_APPROVED",
        "panels": metadata,
        "forecast_rows": len(forecasts),
        "specifications": specifications(),
        "limitations": [
            "Revised vintages and assumed external publication delays.",
            "Nested-model tests with generated factors are descriptive; multiple comparisons are not adjusted.",
            "HAC uses actual calendar-month distances (user-approved adaptation); bandwidth and Bartlett weights are inherited.",
        ],
        "outputs": sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"forecast_rows": len(forecasts), "matrices": 8, "output": str(output)}
