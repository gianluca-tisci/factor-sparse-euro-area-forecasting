"""Explain saved energy-loss reversals without transferring MQ selection claims."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_mq_energy_diagnostics import markdown

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/nonzero_robustness/energy_reversal")


def main():
    if OUT.exists():
        raise FileExistsError("Use a new report directory; do not overwrite results")
    paths = [
        ROOT / name
        for name in ["forecasts.csv", "coverage.csv", "tuning.csv", "selected_predictors.csv"]
    ]
    fixed_paths = {
        "ME": Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr/forecasts.csv"),
        "MQE": Path("outputs/reports/nonzero_robustness/mqe_ablation/forecasts.csv"),
    }
    paths += list(fixed_paths.values()) + [
        Path("scripts/report_nonzero_energy_reversal.py"),
        Path("scripts/report_mq_energy_diagnostics.py"),
    ]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    forecasts = pd.read_csv(ROOT / "forecasts.csv")
    tuning = pd.read_csv(ROOT / "tuning.csv")
    selected = pd.read_csv(ROOT / "selected_predictors.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    dates = coverage[
        coverage.target.eq("HICPNG_EA") & coverage["sample"].eq("primary")
    ].common_dates.unique()
    assert len(dates) == 1
    dates = dates[0].split(";")
    monthly, summaries, contributions, selections = [], [], [], []
    for panel, path in fixed_paths.items():
        frame = forecasts[
            forecasts.target.eq("HICPNG_EA")
            & forecasts.information_set.eq(panel)
            & forecasts.origin.isin(dates)
        ]
        lass = frame[frame.model.eq("LASSO")].set_index("origin").loc[dates]
        rows = pd.DataFrame(
            {
                "target_month": lass.target_month,
                "actual": lass.actual,
                "LASSO": lass.forecast,
                "LASSO_squared_error": lass.error.pow(2),
            }
        )
        for model in ["DFM", "FarmPredict"]:
            other = frame[frame.model.eq(model)].set_index("origin").loc[dates]
            np.testing.assert_allclose(other.actual, lass.actual)
            rows[model] = other.forecast
            rows[model + "_squared_error"] = other.error.pow(2)
        fixed = pd.read_csv(path)
        fixed = (
            fixed[fixed.target.eq("HICPNG_EA") & fixed.origin.isin(dates)]
            .set_index("origin")
            .loc[dates]
        )
        np.testing.assert_allclose(fixed.actual, lass.actual)
        rows["Fixed_factor"] = fixed.forecast
        rows["Fixed_factor_squared_error"] = fixed.error.pow(2)
        rows["Fixed_common_forecast"] = fixed.common_forecast
        rows["Fixed_sparse_correction"] = fixed.sparse_correction
        rows["Original_sparse_correction"] = rows.FarmPredict - rows.DFM
        tuned = (
            tuning[
                tuning.target.eq("HICPNG_EA")
                & tuning.information_set.eq(panel)
                & tuning.model.eq("LASSO")
            ]
            .set_index("origin")
            .loc[dates]
        )
        slopes = selected[
            selected.target.eq("HICPNG_EA")
            & selected.information_set.eq(panel)
            & selected.model.eq("LASSO")
            & selected.origin.isin(dates)
        ]
        count = slopes.groupby("origin").size().reindex(dates, fill_value=0)
        np.testing.assert_allclose(count, tuned.selected)
        rows["selected_slopes"] = count
        rows["selected_terms"] = (
            slopes.groupby("origin").term.agg(";".join).reindex(dates, fill_value="")
        )
        rows["panel"] = panel
        rows.index.name = "origin"
        for model in ["FarmPredict", "Fixed_factor"]:
            rows[model + "_excess_squared_error_vs_LASSO"] = (
                rows[model + "_squared_error"] - rows.LASSO_squared_error
            )
            for phase, start, end in [
                ("pre_2022_no_2020", "2015-01", "2021-12"),
                ("crisis_2022_2023", "2022-01", "2023-12"),
                ("post_2023", "2024-01", "2025-12"),
            ]:
                block = rows[rows.target_month.between(start, end)]
                a, b = block[model + "_squared_error"].mean(), block.LASSO_squared_error.mean()
                summaries.append(
                    {
                        "panel": panel,
                        "phase": phase,
                        "model": model,
                        "n": len(block),
                        "MSE": a,
                        "LASSO_MSE": b,
                        "reduction_vs_LASSO_percent": 100 * (1 - a / b),
                        "LASSO_win_months": int(
                            (block[model + "_squared_error"] > block.LASSO_squared_error).sum()
                        ),
                        "intercept_only_LASSO_months": int(block.selected_slopes.eq(0).sum()),
                    }
                )
            crisis = rows[rows.target_month.between("2022-01", "2023-12")]
            for label, block in [
                ("negative_actual", crisis[crisis.actual.lt(0)]),
                ("nonnegative_actual", crisis[crisis.actual.ge(0)]),
            ]:
                contributions.append(
                    {
                        "panel": panel,
                        "model": model,
                        "actual_sign": label,
                        "n": len(block),
                        "LASSO_win_months": int(
                            (block[model + "_squared_error"] > block.LASSO_squared_error).sum()
                        ),
                        "excess_SSE_vs_LASSO": block[
                            model + "_excess_squared_error_vs_LASSO"
                        ].sum(),
                    }
                )
        monthly.append(rows.reset_index())
        selections.append(slopes[slopes.origin.between("2022-01", "2023-12")])
    all_rows = pd.concat(monthly, ignore_index=True)
    periods = pd.DataFrame(summaries)
    OUT.mkdir(parents=True)
    all_rows.to_csv(OUT / "monthly_errors_and_selections.csv", index=False)
    periods.to_csv(OUT / "period_MSE.csv", index=False)
    pd.DataFrame(contributions).to_csv(OUT / "crisis_actual_sign.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(
        OUT / "crisis_selected_predictors.csv", index=False
    )
    episodes = all_rows[
        all_rows.origin.isin(
            ["2022-04", "2022-07", "2022-08", "2022-10", "2022-11", "2022-12", "2023-03"]
        )
    ]
    columns = [
        "panel",
        "origin",
        "actual",
        "LASSO",
        "DFM",
        "FarmPredict",
        "Fixed_factor",
        "selected_slopes",
        "selected_terms",
    ]
    report = [
        "# Why direct LASSO catches up in energy inflation",
        "Saved primary ME/MQE forecasts; 2020 scoring months excluded. Descriptive post-hoc diagnosis, not a causal test or new validation sample.",
        "## Period MSE and win counts",
        markdown(periods.round(6)),
        "## Reversal episodes and the October counterexample",
        markdown(episodes[columns].round(6)),
        (
            "Check the saved per-origin predictor lists for the revised nonzero-IQR run rather than "
            "transferring individual selections from the historical run. Direct LASSO has nonzero "
            "slopes throughout 2022-2023 in ME/MQE, as verified against saved tuning counts. "
            "Selection alone does not establish contribution direction or causal importance; "
            "that requires predictor values, centering and coefficient products or an ablation."
        ),
        (
            "LASSO forecasts are often much less positive than the expanded-panel DFM forecasts. "
            "When realized inflation becomes negative, a smaller positive forecast can make a much smaller "
            "squared error without predicting the correct sign. July and August 2022 also reward less positive "
            "forecasts despite slightly positive realizations. It is inaccurate to call every such forecast "
            "zero, an unconditional training mean or evidence of stronger shrinkage: ME/MQE retain selected "
            "signals, and near-zero predictions can also reflect predictor realizations and offsetting terms."
        ),
        (
            "October 2022 is a counterexample for the original expanded-panel FarmPredict comparison: "
            "the large positive realization is closer to its high forecast than to direct LASSO. "
            "The report retains every scored month, including losses. A few large favorable squared-error "
            "differences can outweigh more numerous small losses; cumulative rankings depend on error magnitudes, "
            "not just win counts. The crisis sign table separates negative and nonnegative realized changes."
        ),
        (
            "Fixed-factor ablation changes the common forecast relative to original ME/MQE, but still leaves "
            "a largely factor-driven prediction. A LASSO residual correction does not shrink the entire common "
            "forecast. Both the original correction (FarmPredict minus DFM) and the fixed correction are "
            "exported monthly, so this observation is directly inspectable."
        ),
        (
            "Outside-crisis results show whether relative losses concentrate in the excluded episode, "
            "not why it occurred. Crisis data remain in later training samples. Inspect external_crisis/comparisons.csv "
            "in the parent report for the primary, broad and narrow exclusions. No LaTeX changes or refits."
        ),
    ]
    (OUT / "README.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (OUT / "audit.json").write_text(
        json.dumps({"sources": hashes, "model_refits": 0}, indent=2), encoding="utf-8"
    )
    print(periods.to_string(index=False))


if __name__ == "__main__":
    main()
