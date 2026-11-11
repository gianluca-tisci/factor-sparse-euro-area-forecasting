"""Common-date MSE comparisons, FMM cumulative ratios and selected residuals."""

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src import _style
from src.forecasting.experiment import MODELS


def loss_test(differential, dates=None):
    """Two-sided normal HAC loss test; h=1, explicit Newey--West bandwidth.

    Optional dates match covariance products by calendar-month distance, preserving
    gaps without compressing them. The default historical row-based test is unchanged.
    """
    d = np.asarray(differential, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), 0
    bandwidth = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    centered = d - d.mean()
    variance = centered @ centered / n
    calendar = None
    if dates is not None:
        index = pd.PeriodIndex(dates, freq="M")
        if len(index) != n or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("HAC dates must be unique, ordered and aligned with losses")
        calendar = pd.Series(centered, index=index).reindex(
            pd.period_range(index[0], index[-1], freq="M")
        )
    for lag in range(1, bandwidth + 1):
        product = (
            centered[lag:] @ centered[:-lag]
            if calendar is None
            else (calendar * calendar.shift(lag)).sum()
        )
        variance += 2 * (1 - lag / (bandwidth + 1)) * product / n
    if variance <= 0:
        return float("nan"), float("nan"), bandwidth
    statistic = float(d.mean() / np.sqrt(variance / n))
    return statistic, float(2 * norm.sf(abs(statistic))), bandwidth


def evaluate(forecasts):
    """Pair all models on exactly the same origins, with optional scoring exclusions."""
    summaries, comparisons, cumulative = [], [], []
    for sample in ["all", "without_2020", "unflagged_without_2020"]:
        subset = forecasts.copy()
        if sample != "all":
            subset = subset[~subset.target_month.str.startswith("2020-")]
        if sample.startswith("unflagged"):
            subset = subset[~subset.flagged]
        for target, data in subset.groupby("target", sort=False):
            errors = data.pivot(index="origin", columns="model", values="error").sort_index()
            predictions = data.pivot(
                index="origin", columns="model", values="forecast"
            ).sort_index()
            if set(errors.columns) != set(MODELS):
                raise ValueError(f"Unequal model scoring dates for {target}")
            if errors.isna().any().any():
                failed = data[data.error.isna()]
                if "status" not in failed or not failed.status.eq("unavailable").all():
                    raise ValueError(f"Undocumented missing forecasts for {target}")
                common = errors.dropna().index
                errors, predictions = errors.loc[common], predictions.loc[common]
            if len(errors) < 2:
                raise ValueError(f"Insufficient common scoring dates for {target}")
            squared = errors**2
            mse = squared.mean()
            best = mse.min()
            for model in MODELS:
                summaries.append(
                    {
                        "sample": sample,
                        "target": target,
                        "model": model,
                        "n": len(errors),
                        "MSE": mse[model],
                        "RMSE": np.sqrt(mse[model]),
                        "MAE": errors[model].abs().mean(),
                        "relative_MSE_AR": mse[model] / mse["AR"],
                        "best_MSE": bool(np.isclose(mse[model], best, rtol=1e-10, atol=0)),
                    }
                )
            for competitor in MODELS:
                if competitor == "FarmPredict_MQ":
                    continue
                ratio = squared.FarmPredict_MQ.cumsum() / squared[competitor].cumsum().replace(
                    0, np.nan
                )
                for origin, value in ratio.items():
                    cumulative.append(
                        {
                            "sample": sample,
                            "target": target,
                            "origin": origin,
                            "benchmark": competitor,
                            "cumulative_MSE_ratio": value,
                        }
                    )
                nested = competitor in ["MF_factor_only", "MF_factor_own_AR", "FarmPredict_M"]
                difference = squared[competitor] - squared.FarmPredict_MQ
                if nested:
                    difference += (predictions[competitor] - predictions.FarmPredict_MQ) ** 2
                statistic, pvalue, bandwidth = loss_test(difference)
                comparisons.append(
                    {
                        "sample": sample,
                        "target": target,
                        "benchmark": competitor,
                        "n": len(errors),
                        "MSE_ratio": mse.FarmPredict_MQ / mse[competitor],
                        "wins_MSE": bool(mse.FarmPredict_MQ < mse[competitor]),
                        "test": "Clark_West" if nested else "Diebold_Mariano",
                        "statistic": statistic,
                        "pvalue_two_sided": pvalue,
                        "pvalue_one_sided_improvement": norm.sf(statistic),
                        "HAC_bandwidth": bandwidth,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons), pd.DataFrame(cumulative)


def selection_tables(selected, forecasts, monthly, quarterly):
    """Zero selections count in denominators; union of lags for variable frequencies."""
    counts = forecasts[forecasts.model == "FarmPredict_MQ"].groupby("target").size()
    chosen = selected[selected.model == "FarmPredict_MQ"].copy()
    variable_counts = (
        chosen.drop_duplicates(["origin", "target", "predictor"])
        .groupby(["target", "predictor"])
        .size()
    )
    index = pd.MultiIndex.from_product(
        [monthly, monthly + quarterly], names=["target", "predictor"]
    )
    frequency = (
        variable_counts.reindex(index, fill_value=0).rename("origins_selected").reset_index()
    )
    frequency["n_origins"] = frequency.target.map(counts)
    frequency["selection_frequency"] = frequency.origins_selected / frequency.n_origins
    frequency["frequency"] = np.where(frequency.predictor.isin(quarterly), "Q", "M")
    frequency["own_residual"] = frequency.target == frequency.predictor
    terms = (
        chosen.groupby(["target", "term", "predictor", "frequency", "released_position"])
        .agg(
            origins_selected=("origin", "nunique"),
            mean_coefficient_when_selected=("coefficient", "mean"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
        )
        .reset_index()
    )
    terms["n_origins"] = terms.target.map(counts)
    terms["selection_frequency"] = terms.origins_selected / terms.n_origins
    return frequency, terms


def report(output):
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["completion"] != "COMPLETE":
        raise ValueError(
            "Incomplete run; refusing to label partial forecasts as a final comparison"
        )
    folders = [output / origin for origin in metadata["origins"]]
    forecasts = pd.concat([pd.read_csv(p / "forecasts.csv") for p in folders], ignore_index=True)
    unavailable = forecasts[forecasts.error.isna()].copy()
    unavailable.to_csv(output / "unavailable_forecasts.csv", index=False)
    selected = pd.concat([pd.read_csv(p / "selected.csv") for p in folders], ignore_index=True)
    tuning = pd.concat([pd.read_csv(p / "tuning.csv") for p in folders], ignore_index=True)
    fits = pd.DataFrame([json.loads((p / "fit.json").read_text(encoding="utf-8")) for p in folders])
    # Store assumed source availability explicitly; these are monthly rules,
    # not invented historical first-release calendar dates.
    origins = pd.PeriodIndex(selected.origin, freq="M")
    delays = selected.predictor.map(metadata["release_lags"]).astype(int).to_numpy()
    latest = origins - delays
    position = selected.released_position.astype(int).to_numpy() - 1
    offsets = np.where(selected.frequency.eq("Q"), latest.month % 3 + 3 * position, position)
    references = latest - offsets
    if ((references + delays) > origins).any():
        raise AssertionError("A selected predictor was not released at its forecast origin")
    selected["reference_month"] = references.astype(str)
    selected["assumed_available_month"] = (references + delays).astype(str)
    selected["estimate_cutoff"] = selected.origin
    forecasts.to_csv(output / "forecasts.csv", index=False)
    selected.to_csv(output / "selected_terms_all_models.csv", index=False)
    selected[selected.model == "FarmPredict_MQ"].to_csv(
        output / "selected_residuals_by_origin.csv", index=False
    )
    tuning.to_csv(output / "tuning.csv", index=False)
    fits.to_csv(output / "fit_monitoring.csv", index=False)
    metrics, pairs, cumulative = evaluate(forecasts)
    metrics.to_csv(output / "metrics.csv", index=False)
    pairs.to_csv(output / "comparisons.csv", index=False)
    cumulative.to_csv(output / "cumulative_MSE_ratios.csv", index=False)
    names = list(forecasts.target.drop_duplicates())
    quarters = metadata["profile"]["quarterly_series"]
    frequency, terms = selection_tables(selected, forecasts, names, quarters)
    frequency.to_csv(output / "selected_residual_frequencies.csv", index=False)
    terms.to_csv(output / "selected_lag_frequencies.csv", index=False)
    main = metrics[metrics["sample"] == "without_2020"]
    matrix = main.pivot(index="target", columns="model", values="MSE")
    ranking = pd.DataFrame(
        {
            "best_model": matrix.idxmin(axis=1),
            "FarmPredict_MQ_MSE": matrix.FarmPredict_MQ,
            "ratio_AR": matrix.FarmPredict_MQ / matrix.AR,
            "ratio_MF_factor_only": matrix.FarmPredict_MQ / matrix.MF_factor_only,
            "wins_all_models": matrix.FarmPredict_MQ.eq(matrix.min(axis=1)),
        }
    )
    best_other = matrix.drop(columns="FarmPredict_MQ").min(axis=1)
    tied = np.isclose(matrix.FarmPredict_MQ, best_other, rtol=1e-10, atol=0)
    ranking["wins_all_models"] = (matrix.FarmPredict_MQ < best_other) & ~tied
    ranking["best_or_tied"] = matrix.FarmPredict_MQ.eq(matrix.min(axis=1)) | tied
    ranking["ties_with"] = [
        "; ".join(
            model
            for model in matrix.columns
            if model != "FarmPredict_MQ"
            and np.isclose(
                matrix.loc[target, model], matrix.loc[target, "FarmPredict_MQ"], rtol=1e-10, atol=0
            )
        )
        for target in matrix.index
    ]
    ranking = ranking.sort_values("ratio_AR")
    ranking.to_csv(output / "winners.csv")
    # All exhibits use the project style; separate namespaces prevent smoke outputs
    # from replacing full-run figures or the paper's numbers.
    original_out = _style.OUT
    # Publish the final default run through the canonical outputs manifest.
    # Development/custom runs retain a separate namespace.
    canonical = output.resolve() == Path("outputs/forecasts/analysis_final").resolve()
    _style.OUT = Path("outputs") if canonical else output / "exhibits"
    notes = (
        "Expanding window; one-step to each target's first unpublished month; "
        "EA-MD-QD 09.2026 revised vintage; r=3, VAR(2); modified-BIC LASSO. "
        "Test origins from January 2015; target months in 2020 excluded. "
        "Wins denote lower sample MSE, not statistical significance. "
        "Source: EA-MD-QD; authors' code. EXPLORATORY corrected estimator."
    )
    try:
        _style.save_table(
            ranking.drop(columns=["ties_with", "best_or_tied"]),
            "tab_35_forecast_winners",
            notes=notes,
            used_in="analysis",
            float_format="%.4g",
        )
        _style.record_number("AnalysisForecastOrigins", len(folders))
        _style.record_number("AnalysisFarmPredictWinners", int(ranking.wins_all_models.sum()))
        _style.record_number("AnalysisTargets", len(names))
        _style.record_number("AnalysisModels", len(MODELS))
        _style.record_number("AnalysisForecastRows", len(forecasts))
        _style.record_number("AnalysisFlaggedOrigins", int(fits.flagged.sum()))
        _style.record_number("AnalysisBeatsAR", int((ranking.ratio_AR < 1).sum()))
        all_ar = pairs[(pairs["sample"] == "all") & (pairs.benchmark == "AR")]
        _style.record_number("AnalysisBeatsARAll", int(all_ar.wins_MSE.sum()))
        comparators = ["AR", "PCR", "AR_PCR", "sparse_only", "MF_factor_only"]
        colors = _style.PALETTE_MANY
        plot_data = cumulative[cumulative["sample"] == "without_2020"]
        for page in range(4):
            fig, axes = plt.subplots(4, 3, figsize=(14, 12), squeeze=False)
            for ax, name in zip(axes.ravel(), names[page * 12 : (page + 1) * 12], strict=False):
                target = plot_data[plot_data.target == name]
                for benchmark, color in zip(comparators, colors, strict=True):
                    line = target[target.benchmark == benchmark]
                    ax.plot(
                        pd.to_datetime(line.origin),
                        line.cumulative_MSE_ratio,
                        label={
                            "AR_PCR": "AR-PCR",
                            "sparse_only": "LASSO",
                            "MF_factor_only": "MF factor-only",
                        }.get(benchmark, benchmark),
                        color=color,
                        linewidth=1,
                    )
                ax.axhline(1, color="black", linestyle=":", linewidth=0.7)
                ax.set_yscale("log")
                ax.set_title(name, fontsize=9)
                ax.tick_params(labelsize=7)
            for ax in axes.ravel()[len(names[page * 12 : (page + 1) * 12]) :]:
                ax.set_visible(False)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=9
            )
            fig.suptitle("Cumulative MSE: FarmPredict-MQ / benchmark", y=0.995)
            fig.text(
                0.02, -0.015, "Notes: logarithmic vertical axes. " + notes, fontsize=8, wrap=True
            )
            fig.tight_layout(rect=(0, 0.02, 1, 0.96))
            _style.save_fig(
                fig, f"fig_{35 + page}_cumulative_mse_{page + 1}", formats=["png", "pdf"]
            )
            plt.close(fig)
        heat = frequency.pivot(index="target", columns="predictor", values="selection_frequency")
        heat = heat.reindex(index=names, columns=names + quarters)
        fig, ax = plt.subplots(figsize=(17, 11))
        image = ax.imshow(heat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.axvline(46.5, color="black", linewidth=1)
        ax.set_title("LASSO residual selection frequencies")
        ax.set_xlabel("Predictor residual (47 monthly | 16 quarterly)")
        ax.set_ylabel("Monthly target")
        fig.colorbar(image, ax=ax, label="Share of forecast origins; union over lags")
        fig.text(
            0.02,
            -0.025,
            "Notes: zero-selection origins count in denominators. "
            + notes.replace(
                "target months in 2020 excluded.",
                "selection frequencies include every origin, including 2020.",
            ),
            fontsize=8,
            wrap=True,
        )
        fig.tight_layout()
        _style.save_fig(fig, "fig_39_residual_selection", formats=["png", "pdf"])
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.scatter(ranking.ratio_AR, np.arange(len(ranking)), color=_style.PALETTE[0], s=22)
        ax.set_yticks(np.arange(len(ranking)), ranking.index)
        ax.set_xscale("log")
        ax.axvline(1, color="black", linestyle=":")
        ax.invert_yaxis()
        ax.set_xlabel("FarmPredict-MQ MSE / AR MSE (below 1: lower MSE)")
        ax.set_title("Forecast gains by target")
        ax.tick_params(axis="y", labelsize=7)
        fig.text(
            0.02, -0.025, "Notes: logarithmic horizontal axis. " + notes, fontsize=8, wrap=True
        )
        fig.tight_layout()
        _style.save_fig(fig, "fig_40_forecast_ranking", formats=["png", "pdf"])
        plt.close(fig)
    finally:
        _style.OUT = original_out
    summary = {
        "origins": len(folders),
        "targets": len(names),
        "models": len(MODELS),
        "forecast_rows": len(forecasts),
        "unavailable_forecast_rows": len(unavailable),
        "flagged_origins": int(fits.flagged.sum()),
        "wins_all_without_2020": ranking.index[ranking.wins_all_models].tolist(),
        "beats_AR_without_2020": ranking.index[ranking.ratio_AR < 1].tolist(),
        "beats_AR_all": all_ar.loc[all_ar.wins_MSE, "target"].tolist(),
        "exhibits": "outputs" if canonical else str(output / "exhibits"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = Path(__file__).resolve().relative_to(Path.cwd())
    report_key = str(report_path)
    metadata.setdefault(
        "forecast_source_hashes",
        {
            key: value
            for key, value in metadata["source_hashes"].items()
            if Path(key).name != "report.py"
        },
    )
    metadata.setdefault("original_report_source_hash", metadata["source_hashes"][report_key])
    metadata["source_hashes"][report_key] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    metadata["supplemental_source_hashes"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path("src/data/ea_md_qd.py"),
            Path("src/preprocessing/transform.py"),
            Path("src/_style.py"),
            Path("run_all.py"),
            Path("uv.lock"),
        ]
    }
    metadata["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata["uncommitted_implementation"] = True
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary
