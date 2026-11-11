"""Exhibits for the two-question Results outline; saved nonzero-IQR forecasts only."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/results_reorganized")
NAMES = {
    "HICPOV_EA": "Headline",
    "HICPNG_EA": "Energy",
    "HICPNEF_EA": "Core",
    "IPMN_EA": "Manufacturing",
}
LABELS = {
    "AR": "AR",
    "LASSO": "LASSO",
    "PCR": "PCR",
    "PCA_VAR": "PCA+VAR",
    "DFM": "DFM",
    "DFM_AR": "DFM+AR",
    "PCA_VAR_LASSO": "FP-PCA",
    "FarmPredict": "FP-DFM",
}


def main():
    if OUT.exists():
        raise FileExistsError("Use a new export directory")
    sources = [ROOT / n for n in ["scores.csv", "forecasts.csv", "coverage.csv", "robustness.csv"]]
    narrow = Path("outputs/reports/nonzero_robustness/inflation_crisis/scores.csv")
    sources += [narrow, Path("scripts/reorganized_results_exhibits.py"), Path("src/_style.py")]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    scores = pd.read_csv(sources[0])
    scores = scores[scores["sample"].eq("primary")]
    forecasts, coverage = pd.read_csv(sources[1]), pd.read_csv(sources[2])
    old = _style.OUT
    _style.OUT = OUT
    try:
        rows = []
        us = {"AR": 1, "LASSO": 0.9217, "PCR": 1.0249, "PCA_VAR_LASSO": 0.9080}
        for model, label in LABELS.items():
            row = {"Method": label}
            for target, title in NAMES.items():
                panel = "shared" if model == "AR" else "M"
                a = scores[
                    scores.target.eq(target)
                    & scores.information_set.eq(panel)
                    & scores.model.eq(model)
                ].iloc[0]
                row[title] = f"{a.relative_MSE_AR:.3f}"
            row["US IP (FMM)"] = f"{us[model]:.4f}" if model in us else "---"
            rows.append(row)
        rows.append(
            {
                "Method": "Original AR-PCR",
                **{t: "---" for t in NAMES.values()},
                "US IP (FMM)": "0.9215",
            }
        )
        _style.save_table(
            pd.DataFrame(rows),
            "tab_100_monthly_ladder",
            index=False,
            notes="M only, primary nonzero-IQR sample without 2020. MSE divided by AR, lower is better. FP-DFM and FP-PCA are FarmPredict variants. The US column records Fan et al. (2023), not an identical euro-area design; its FarmPredict uses PCA. Their AR-PCR is distinct from our DFM+AR. Source: saved forecasts and FMM Section 4.2.",
        )
        winners = []
        for target, title in NAMES.items():
            a = scores[scores.target.eq(target)].sort_values("MSE").iloc[0]
            winners.append(
                {
                    "Target": title,
                    "Set": a.information_set,
                    "Method": LABELS[a.model],
                    "MSE / AR": f"{a.relative_MSE_AR:.3f}",
                }
            )
        _style.save_table(
            pd.DataFrame(winners),
            "tab_99_overall_winners",
            index=False,
            notes="Lowest primary MSE across all estimated models and information sets, including both FarmPredict variants. Shared univariate forecasts count once. Source: saved nonzero-IQR scores; rankings are descriptive.",
        )
        robust = []
        broad = pd.read_csv(sources[3])
        for sample in ["including_2020", "broader"]:
            for target, title in NAMES.items():
                a = (
                    broad[broad.target.eq(target) & broad["sample"].eq(sample)]
                    .sort_values("MSE")
                    .iloc[0]
                )
                robust.append(
                    {
                        "Sample": sample.replace("_", " "),
                        "Target": title,
                        "Scope": "all sets",
                        "Best": LABELS[a.model] + "-" + a.information_set,
                        "MSE / AR": f"{a.relative_MSE_AR:.3f}",
                    }
                )
        n = pd.read_csv(narrow)
        for target, title in list(NAMES.items())[:3]:
            a = (
                n[
                    n.target.eq(target)
                    & n["sample"].eq("outside_war_phase")
                    & n.panel.isin(["M", "MQ", "shared"])
                ]
                .sort_values("MSE")
                .iloc[0]
            )
            robust.append(
                {
                    "Sample": "exclude Feb 2022--Dec 2023",
                    "Target": title,
                    "Scope": "M/MQ",
                    "Best": LABELS[a.model] + "-" + a.panel,
                    "MSE / AR": f"{a.relative_MSE_AR:.3f}",
                }
            )
        _style.save_table(
            pd.DataFrame(robust),
            "tab_101_robustness_summary",
            index=False,
            notes="Including 2020 and broader are the saved run's sample definitions; the narrower crisis exclusion retains the original matched primary dates and also excludes 2020. Narrow-window winners shown for M/MQ inflation only. No refitting. Source: saved robustness and crisis-sensitivity scores.",
        )
        checks = []
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        styles = [
            ("DFM", _style.PALETTE_MANY[0], "-"),
            ("PCA_VAR", _style.PALETTE_MANY[0], "--"),
            ("DFM_AR", _style.PALETTE_MANY[1], "-"),
            ("FarmPredict", _style.PALETTE_MANY[2], "-"),
            ("PCA_VAR_LASSO", _style.PALETTE_MANY[2], "--"),
            ("LASSO", _style.PALETTE_MANY[3], "-"),
        ]
        for ax, (target, title) in zip(axes.flat, NAMES.items(), strict=True):
            dates = coverage[
                coverage.target.eq(target) & coverage["sample"].eq("primary")
            ].common_dates.unique()
            assert len(dates) == 1
            dates = dates[0].split(";")
            a = forecasts[forecasts.target.eq(target) & forecasts.origin.isin(dates)].copy()
            a["spec"] = a.information_set + ":" + a.model
            errors = a.pivot(index="origin", columns="spec", values="error").loc[dates]
            for model, color, style in styles:
                ratio = errors["M:" + model].pow(2).cumsum() / errors["shared:AR"].pow(2).cumsum()
                gold = scores[
                    scores.target.eq(target)
                    & scores.information_set.eq("M")
                    & scores.model.eq(model)
                ].relative_MSE_AR.item()
                np.testing.assert_allclose(ratio.iloc[-1], gold, rtol=1e-10)
                checks.append({"target": target, "model": model, "endpoint": gold})
                time = pd.PeriodIndex(ratio.index, freq="M").to_timestamp()
                curve = pd.Series(ratio.to_numpy(), index=time).reindex(
                    pd.date_range(time.min(), time.max(), freq="MS")
                )
                ax.plot(
                    curve.index,
                    curve,
                    color=color,
                    linestyle=style,
                    label=LABELS[model],
                    linewidth=1.2,
                )
            ax.axvspan(
                pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), color="gray", alpha=0.13
            )
            ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_ylabel("Cumulative MSE / AR")
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        _style.save_fig(
            fig, "fig_195_monthly_ladder", formats=["png", "pdf"], used_in="Results 4.1 M ladder"
        )
        plt.close(fig)
        pd.DataFrame(checks).to_csv(OUT / "endpoint_checks.csv", index=False)
        for name, digest in hashes.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        (OUT / "audit.json").write_text(
            json.dumps(
                {"sources": hashes, "model_refits": 0, "endpoint_checks": len(checks)}, indent=2
            ),
            encoding="utf-8",
        )
    finally:
        _style.OUT = old
    _style.record_output(
        "results_reorganized",
        [str(OUT / "audit.json")],
        "New Results structure; current nonzero-IQR outputs",
    )


if __name__ == "__main__":
    main()
