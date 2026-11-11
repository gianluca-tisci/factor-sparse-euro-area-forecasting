"""Generate the nonzero-IQR Results draft exhibits; never refit models."""

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
OUT = Path("outputs/reports/results_section_nonzero")
NAMES = {
    "HICPOV_EA": "Headline",
    "HICPNG_EA": "Energy",
    "HICPNEF_EA": "Core",
    "IPMN_EA": "Manufacturing",
}
LABELS = {
    "DFM": "DFM",
    "DFM_AR": "DFM-AR",
    "FarmPredict": "FP-DFM",
    "LASSO": "LASSO",
    "PCR": "PCR",
    "PCA_VAR_LASSO": "FP-PCA",
    "AR": "AR",
    "PCA_VAR": "PCA-VAR",
}


def main():
    if OUT.exists():
        raise FileExistsError("Use a new draft export; do not overwrite existing exhibits")
    sources = [ROOT / n for n in ["forecasts.csv", "coverage.csv", "scores.csv"]]
    sources += [
        Path("outputs/reports/industrial_nonzero/winners.csv"),
        Path("outputs/reports/nonzero_robustness/external_crisis/comparisons.csv"),
        Path("outputs/reports/nonzero_robustness/endpoint_checks.csv"),
        Path("outputs/reports/nonzero_robustness/energy_lasso_selections/README.md"),
        Path("scripts/results_section_nonzero.py"),
        Path("src/_style.py"),
    ]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    scores = pd.read_csv(ROOT / "scores.csv")
    scores = scores[scores["sample"].eq("primary")]
    forecasts = pd.read_csv(ROOT / "forecasts.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    winners, checks = [], []
    old = _style.OUT
    _style.OUT = OUT
    try:

        def number(name, value):
            _style.record_number("Results" + name, f"{value:.2f}")

        def mse(target, panel, model):
            return scores[
                scores.target.eq(target) & scores.information_set.eq(panel) & scores.model.eq(model)
            ].MSE.item()

        for target, title in NAMES.items():
            for panel in ["M", "MQ"]:
                rank = scores[
                    scores.target.eq(target) & scores.information_set.isin([panel, "shared"])
                ].sort_values("MSE")
                a, b = rank.iloc[0], rank.iloc[1]
                winners.append(
                    {
                        "Target": title,
                        "Set": panel,
                        "Best": LABELS[a.model],
                        "MSE": f"{a.MSE:.6f}",
                        "Second": LABELS[b.model],
                        "Gain (\\%)": f"{100 * (1 - a.MSE / b.MSE):.2f}",
                    }
                )
        notes = "Current D-72 nonzero-IQR cold run; matched target dates through December 2025, excluding 2020. Gain is the winner's MSE reduction against the second-best model; descriptive, not a significance test. FP-DFM and FP-PCA denote the two FarmPredict variants. Source: saved paper experiment."
        _style.save_table(
            pd.DataFrame(winners), "tab_95_results_m_mq", index=False, escape=False, notes=notes
        )
        industrial = pd.read_csv(sources[3])
        rows = []
        for target, title in {
            "IPCAG_EA": "Capital goods",
            "IPCOG_EA": "Consumer goods",
            "IPDCOG_EA": "Durables",
            "IPNDCOG_EA": "Non-durables",
            "IPING_EA": "Intermediate goods",
            "IPNRG_EA": "Energy",
        }.items():
            row = {"Production": title}
            for panel in ["M", "MQ"]:
                r = industrial[industrial.target.eq(target) & industrial.panel.eq(panel)].iloc[0]
                row[panel + " best"] = LABELS[r.winner]
                row[panel + " gain (\\%)"] = f"{r.reduction_percent:.2f}"
            rows.append(row)
        _style.save_table(
            pd.DataFrame(rows),
            "tab_96_results_industry",
            index=False,
            escape=False,
            notes=(
                "Six industrial-production aggregates, chosen before the run; information sets M "
                "and MQ. Gain: percentage reduction in MSE of the most accurate model relative to "
                "the second most accurate in the same set. The aggregates overlap, so they are not "
                "independent checks. Target months to December 2025, without 2020. "
                "\\textit{Source:} Own calculations."
            ),
        )
        fixed = pd.read_csv(sources[5])
        rows = []
        for target, title in list(NAMES.items())[:3]:
            for base, panel in [("M", "ME"), ("MQ", "MQE")]:
                value = fixed[fixed.target.eq(target) & fixed.panel.eq(panel)].ablation_MSE.iloc[0]
                rows.append(
                    {
                        "Target": title,
                        "Sets": base + "/" + panel,
                        "DFM without E": f"{mse(target, base, 'DFM'):.6f}",
                        "DFM, E in factors": f"{mse(target, panel, 'DFM'):.6f}",
                        "FP-DFM, E in factors": f"{mse(target, panel, 'FarmPredict'):.6f}",
                        "Fixed-factor": f"{value:.6f}",
                        "LASSO with E": f"{mse(target, panel, 'LASSO'):.6f}",
                    }
                )
        _style.save_table(
            pd.DataFrame(rows),
            "tab_97_results_external",
            index=False,
            notes="Current D-72 nonzero-IQR primary scores without 2020. Fixed-factor: FP-DFM with the M/MQ factors kept and E added only to the sparse step; E in factors: the model re-estimated on ME/MQE. Candidate predictors and effective regression rows can differ. Source: saved paper and fixed-factor experiments.",
        )
        for target, prefix in [
            ("IPMN_EA", "Manufacturing"),
            ("HICPNG_EA", "Energy"),
            ("HICPNEF_EA", "Core"),
        ]:
            number(
                prefix + "QuarterlyGain",
                100 * (1 - mse(target, "MQ", "DFM") / mse(target, "M", "DFM")),
            )
        target = "IPMN_EA"
        for model, suffix in [("PCA_VAR", "Factor"), ("LASSO", "Lasso"), ("AR", "Ar")]:
            number(
                "ManufacturingGain" + suffix,
                100
                * (
                    1
                    - mse(target, "M", "PCA_VAR_LASSO")
                    / mse(target, "shared" if model == "AR" else "M", model)
                ),
            )
        for target, prefix in [("HICPNEF_EA", "Core"), ("HICPNG_EA", "Energy")]:
            number(
                prefix + "ExternalChange",
                100 * (mse(target, "ME", "DFM") / mse(target, "M", "DFM") - 1),
            )
        # Published ratios are documented constants, not new empirical estimates.
        number("OriginalGainAr", 100 * (1 - 0.9080))
        number("OriginalGainSparse", 100 * (1 - 0.9080 / 0.9217))
        comparisons = pd.read_csv(sources[4])
        for panel in ["ME", "MQE"]:
            block = comparisons[
                comparisons.target.eq("HICPNG_EA")
                & comparisons["sample"].eq("outside_energy_crisis")
                & comparisons.panel.eq(panel)
                & comparisons.numerator.eq(panel + ":Fixed_factor")
            ]
            for bench, suffix in [
                (panel + ":LASSO", "Lasso"),
                (("M" if panel == "ME" else "MQ") + ":DFM", "Dfm"),
            ]:
                number(
                    "EnergyFixed" + panel + suffix,
                    block[block.benchmark.eq(bench)].reduction_percent.item(),
                )
        for group, targets in enumerate([["HICPOV_EA", "HICPNG_EA"], ["HICPNEF_EA", "IPMN_EA"]]):
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            for i, target in enumerate(targets):
                dates = coverage[
                    coverage.target.eq(target) & coverage["sample"].eq("primary")
                ].common_dates.unique()
                assert len(dates) == 1
                dates = dates[0].split(";")
                data = forecasts[forecasts.target.eq(target) & forecasts.origin.isin(dates)]
                data = data.assign(spec=data.information_set + ":" + data.model)
                errors = data.pivot(index="origin", columns="spec", values="error").loc[dates]
                for j, panel in enumerate(["M", "MQ"]):
                    ax = axes[i, j]
                    specs = [(panel, m) for m in ["DFM", "DFM_AR", "FarmPredict", "LASSO"]]
                    if panel == "M":
                        specs.append(("M", "PCR"))
                    if panel == "M" or target == "IPMN_EA":
                        specs.append(("M", "PCA_VAR_LASSO"))
                    for (info, model), color in zip(
                        specs, _style.PALETTE_MANY + [_style.PALETTE[0]], strict=False
                    ):
                        ratio = (
                            errors[info + ":" + model].pow(2).cumsum()
                            / errors["shared:AR"].pow(2).cumsum()
                        )
                        expected = mse(target, info, model) / mse(target, "shared", "AR")
                        np.testing.assert_allclose(ratio.iloc[-1], expected, rtol=1e-10)
                        checks.append(
                            {
                                "target": target,
                                "panel": panel,
                                "model": model,
                                "source_panel": info,
                                "endpoint": expected,
                            }
                        )
                        time = pd.PeriodIndex(ratio.index, freq="M").to_timestamp()
                        curve = pd.Series(ratio.to_numpy(), index=time).reindex(
                            pd.date_range(time.min(), time.max(), freq="MS")
                        )
                        ax.plot(
                            curve.index,
                            curve,
                            color=color,
                            label=LABELS[model] + (" (M)" if info != panel else ""),
                            linestyle="--" if model == "PCA_VAR_LASSO" else "-",
                            linewidth=1.1,
                        )
                    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
                    ax.set_yscale("log")
                    ax.set_title(NAMES[target] + " -- " + panel)
                    ax.set_ylabel("Cumulative MSE / AR")
                    ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            _style.save_fig(
                fig,
                f"fig_{185 + group}_results_m_mq",
                formats=["pdf", "png"],
                used_in="Current D-72 nonzero-IQR Results draft",
            )
            plt.close(fig)
        pd.DataFrame(checks).to_csv(OUT / "endpoint_checks.csv", index=False)
        for p, h in hashes.items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        (OUT / "audit.json").write_text(
            json.dumps(
                {
                    "sources": hashes,
                    "run": "current D-72 nonzero-IQR",
                    "model_refits": 0,
                    "endpoint_checks": len(checks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        _style.OUT = old
    _style.record_output(
        "results_section_nonzero",
        [str(OUT / "audit.json"), str(OUT / "numbers.tex")],
        "Current D-72 nonzero-IQR Results rewrite; explicit provenance",
    )


if __name__ == "__main__":
    main()
