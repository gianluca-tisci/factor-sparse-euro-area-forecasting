"""Record M/MQ contrasts and energy episodes from the saved cold run only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/mq_energy_diagnostics")


def markdown(df):
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    rows += [
        "| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        ROOT / f for f in ["forecasts.csv", "scores.csv", "paired_comparisons.csv", "coverage.csv"]
    ]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    pairs = pd.read_csv(ROOT / "paired_comparisons.csv")
    contrasts = pairs[
        pairs["sample"].eq("primary")
        & pairs.augmented.str.startswith("MQ:")
        & pairs.baseline.str.startswith("M:")
    ].copy()
    assert len(contrasts) == 16
    contrasts["MSE_reduction_percent_M_to_MQ"] = 100 * (1 - contrasts.MSE_ratio)
    contrasts.to_csv(OUT / "m_vs_mq.csv", index=False)
    display = contrasts[
        ["target", "augmented", "n", "MSE_reduction_percent_M_to_MQ", "pvalue_two_sided"]
    ].copy()
    display.columns = ["Target", "MQ model", "N", "MSE reduction (%)", "DM p-value"]
    display["MSE reduction (%)"] = display["MSE reduction (%)"].map(lambda x: f"{x:.3f}")
    display["DM p-value"] = display["DM p-value"].map(lambda x: f"{x:.4f}")
    f = pd.read_csv(ROOT / "forecasts.csv", low_memory=False)
    coverage = pd.read_csv(ROOT / "coverage.csv")
    dates = coverage[
        (coverage.target == "HICPNG_EA") & coverage["sample"].eq("primary")
    ].common_dates
    assert dates.nunique() == 1
    dates = dates.iloc[0].split(";")
    selected = f[
        f.target.eq("HICPNG_EA")
        & f.origin.isin(dates)
        & f.information_set.isin(["MQ", "shared"])
        & f.model.isin(["FarmPredict", "DFM", "LASSO", "AR"])
    ].copy()
    assert selected.valid.all() and not selected.target_month.str.startswith("2020-").any()
    assert selected.target_month.eq(selected.origin).all()
    error = selected.pivot(index="origin", columns="model", values="error").loc[dates]
    squared = error**2
    periods = []
    for start, end in [("2015-01", "2021-12"), ("2022-01", "2023-12"), ("2024-01", "2025-12")]:
        block = squared.loc[start:end]
        mean = block.mean()
        periods.append(
            {
                "period": start + " to " + end,
                "n": len(block),
                **{model: mean[model] for model in mean.index},
                "FP_reduction_vs_LASSO_percent": 100 * (1 - mean.FarmPredict / mean.LASSO),
            }
        )
    period_frame = pd.DataFrame(periods)
    assert period_frame.n.sum() == len(dates) == 120
    period_frame.to_csv(OUT / "energy_period_MSE.csv", index=False)
    monthly = squared.copy()
    monthly["FP_minus_LASSO_squared_loss"] = squared.FarmPredict - squared.LASSO
    monthly["FP_over_LASSO_cumulative_ratio"] = (
        squared.FarmPredict.cumsum() / squared.LASSO.cumsum()
    )
    monthly.to_csv(OUT / "energy_monthly_losses.csv")
    episodes = selected[selected.origin.isin(["2022-04", "2022-11", "2022-12", "2023-03"])]
    episode_frame = episodes.pivot(
        index=["origin", "target_month", "actual"], columns="model", values="forecast"
    ).reset_index()
    episode_frame.to_csv(OUT / "energy_inversion_forecasts.csv", index=False)
    paragraphs = [
        "# M/MQ and energy forecast diagnostics",
        (
            "Saved all-cold paper run; primary scoring dates through 2025-12; 2020 target months excluded. "
            "No estimators, settings, saved forecasts or LaTeX files are changed. This is a research review note, not paper prose."
        ),
        "## Quarterly information: matched-model comparisons",
        (
            "Positive reductions favor MQ. Compare the same model on the same dates, rather than only comparing the winners "
            "of different model menus. P-values are the saved two-sided Diebold-Mariano values, unadjusted for multiple comparisons."
        ),
        markdown(display),
        (
            "Quarterly information has selective predictive value. Manufacturing improves in the DFM family; "
            "inflation does not improve systematically. LASSO improves descriptively for energy and core, without nominal "
            "5% evidence. Manufacturing DFM and FarmPredict-DFM improve by similar amounts. This is consistent with a factor "
            "channel, but does not identify it: M to MQ changes both the factor panel and the sparse dictionary. "
            "A fixed-factor quarterly-dictionary ablation would be needed to separate those channels."
        ),
        (
            "The best monthly-only manufacturing method remains FarmPredict (PCA+VAR+LASSO). "
            "The MQ improvement of DFM does not eliminate that performance gap. The PCA variant is not estimated in MQ, "
            "so these results do not identify the effect of adding quarterly information to PCA-based FarmPredict."
        ),
        "## Energy: periods and inversion episodes (MQ)",
        markdown(period_frame.round(6)),
        markdown(episode_frame.round(6)),
        (
            "The cumulative FarmPredict/LASSO ranking reverses mainly during 2022-2023. "
            "In the listed inversion episodes, the DFM-based forecast remains positive while the realized monthly "
            "energy-price change is negative. Direct LASSO forecasts are less positive or negative. Large squared "
            "errors dominate earlier smaller gains. Results in 2024-2025 are nearly tied, so the high final cumulative "
            "ratio does not mean LASSO continuously improves at the end of the sample. FarmPredict and DFM forecasts "
            "remain very close: the sparse residual correction does not materially repair these factor forecast errors."
        ),
        (
            "These observations are descriptive and do not establish which selected predictors cause the difference. "
            "That question requires a separate selection/ablation analysis. No causal claim about an economic shock "
            "or a general rejection of quarterly information follows from these diagnostics."
        ),
        "Source: outputs/forecasts/paper_experiment_nonzero_iqr; generated by scripts/report_mq_energy_diagnostics.py.",
    ]
    (OUT / "README.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.as_posix()]
    assert np.isfinite(error.to_numpy()).all()
    (OUT / "provenance.json").write_text(
        json.dumps({"sources": hashes}, indent=2), encoding="utf-8"
    )
    _style.record_output(
        "mq_energy_diagnostics", [str(OUT / "README.md")], "Research notes; no LaTeX changes"
    )
    print("Verified 16 M/MQ contrasts and 120 matched energy dates; generated diagnostics.")


if __name__ == "__main__":
    main()
