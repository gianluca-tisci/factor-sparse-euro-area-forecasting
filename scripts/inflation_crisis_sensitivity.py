"""Score saved inflation forecasts outside fixed crisis windows; do not refit."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.report_m_mq_winners import LABELS
from scripts.report_mq_energy_diagnostics import markdown
from src import _style

CONFIG = Path("config/evaluation_inflation_crisis.yaml")
OUT = Path("outputs/reports/inflation_crisis_sensitivity")
NAMES = {"HICPOV_EA": "Headline", "HICPNG_EA": "Energy", "HICPNEF_EA": "Core"}


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    root = Path(cfg["source_run"])
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [CONFIG] + [root / f for f in ["forecasts.csv", "coverage.csv", "scores.csv"]]
    paths += [root / p / "metadata.json" for p in cfg["information_sets"]]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for panel in cfg["information_sets"]:
        meta = json.loads((root / panel / "metadata.json").read_text())
        assert meta["completion"] == "COMPLETE"
        assert meta["profile"]["reestimation_start"] == "cold"
    forecasts = pd.read_csv(root / "forecasts.csv", low_memory=False)
    coverage = pd.read_csv(root / "coverage.csv")
    gold = pd.read_csv(root / "scores.csv")
    metrics, winners, cumulative, endpoints, cards = [], [], [], [], []
    for i, target in enumerate(cfg["targets"]):
        data = forecasts[forecasts.target == target].copy()
        data["spec"] = data.information_set + ":" + data.model
        assert not data.duplicated(["origin", "spec"]).any()
        cov = coverage[(coverage.target == target) & coverage["sample"].eq("primary")]
        assert cov.common_dates.nunique() == 1
        dates = cov.common_dates.iloc[0].split(";")
        data = data[data.origin.isin(dates)]
        assert data.valid.all() and data.target_month.eq(data.origin).all()
        assert data.target_month.le(cfg["score_through"]).all()
        assert (
            not data.target_month.str[:4].astype(int).isin(cfg["always_exclude_target_years"]).any()
        )
        error = data.pivot(index="origin", columns="spec", values="error").loc[dates]
        assert np.isfinite(error.to_numpy()).all()
        baseline = gold[(gold.target == target) & gold["sample"].eq("primary")].copy()
        baseline["spec"] = baseline.information_set + ":" + baseline.model
        baseline = baseline.set_index("spec")
        np.testing.assert_allclose((error**2).mean().loc[baseline.index], baseline.MSE, rtol=1e-10)
        samples = {"primary": error}
        for sample, window in cfg["exclusions"].items():
            mask = ~error.index.to_series().between(window["start"], window["end"])
            samples[sample] = error.loc[mask]
            assert samples[sample].index.isin(error.index).all()
            assert (
                not samples[sample].index.to_series().between(window["start"], window["end"]).any()
            )
        for sample, errors in samples.items():
            squared = errors**2
            mse = squared.mean()
            assert len(errors) >= 24
            for spec, value in mse.items():
                panel, model = spec.split(":")
                metrics.append(
                    {
                        "sample": sample,
                        "target": target,
                        "panel": panel,
                        "model": model,
                        "n": len(errors),
                        "MSE": value,
                        "relative_MSE_AR": value / mse["shared:AR"],
                    }
                )
            for panel in cfg["information_sets"]:
                columns = [c for c in mse.index if c.startswith((panel + ":", "shared:"))]
                ranked = mse[columns].sort_values()
                winners.append(
                    {
                        "sample": sample,
                        "target": target,
                        "panel": panel,
                        "n": len(errors),
                        "winner": ranked.index[0].split(":")[1],
                        "winner_MSE": ranked.iloc[0],
                        "runner_up": ranked.index[1].split(":")[1],
                        "runner_up_MSE": ranked.iloc[1],
                        "reduction_percent": 100 * (1 - ranked.iloc[0] / ranked.iloc[1]),
                    }
                )
        sample = cfg["primary_sensitivity"]
        errors = samples[sample]
        sums = (errors**2).cumsum()
        for j, panel in enumerate(cfg["information_sets"]):
            models = ["FarmPredict", "PCA_VAR_LASSO"] if panel in ["M", "ME"] else ["FarmPredict"]
            fig, axes = plt.subplots(
                1, len(models), figsize=(13 if len(models) == 2 else 8, 5.5), squeeze=False
            )
            for ax, model in zip(axes.flat, models, strict=True):
                numerator = panel + ":" + model
                factor = "PCA_VAR" if model == "PCA_VAR_LASSO" else "DFM"
                benchmarks = ["AR", "LASSO", factor] + (
                    ["PCR", "DFM_AR"] if panel in ["M", "ME"] else ["DFM_AR"]
                )
                available = [
                    c for c in sums if (c.startswith((panel + ":", "shared:"))) and c != numerator
                ]
                for denominator in available:
                    ratio = sums[numerator] / sums[denominator]
                    endpoint = (errors[numerator] ** 2).mean() / (errors[denominator] ** 2).mean()
                    np.testing.assert_allclose(ratio.iloc[-1], endpoint, rtol=1e-12)
                    endpoints.append(
                        {
                            "target": target,
                            "panel": panel,
                            "model": model,
                            "benchmark": denominator,
                            "n": len(errors),
                            "final_ratio": endpoint,
                        }
                    )
                    for month, value in ratio.items():
                        cumulative.append(
                            {
                                "target": target,
                                "panel": panel,
                                "model": model,
                                "benchmark": denominator,
                                "target_month": month,
                                "cumulative_MSE_ratio": value,
                            }
                        )
                for benchmark, color in zip(benchmarks, _style.PALETTE_MANY, strict=False):
                    denominator = "shared:AR" if benchmark == "AR" else panel + ":" + benchmark
                    curve = sums[numerator] / sums[denominator]
                    curve.index = pd.PeriodIndex(curve.index, freq="M").to_timestamp()
                    curve = curve.reindex(
                        pd.date_range(curve.index.min(), curve.index.max(), freq="MS")
                    )
                    ax.plot(curve.index, curve, color=color, label=LABELS[benchmark], linewidth=1.3)
                ax.axhline(1, color="#555555", linestyle="--", linewidth=0.8)
                ax.set_yscale("log")
                ax.set_title(LABELS[model])
                ax.set_ylabel("Cumulative MSE ratio (log scale)")
                ax.set_xlabel("Target month")
                ax.grid(axis="y", alpha=0.2)
                ax.legend(fontsize=8, loc="best")
            fig.suptitle(f"{NAMES[target]} ({target}) — {panel} — outside energy crisis")
            fig.text(
                0.02,
                0.015,
                f"Notes: {len(errors)} common scoring dates; 2020 and 2021-2023 target months excluded; through 2025-12.\n"
                "Saved cold-run forecasts; training unchanged. Ratio = FarmPredict / benchmark; below 1 favors FarmPredict.\n"
                "Gaps are not reset or connected. Source: saved EA-MD-QD / external-series paper experiment.",
                fontsize=7,
            )
            fig.tight_layout(rect=(0, 0.14, 1, 0.94))
            figure_id = f"fig_{115 + 4 * i + j}_outside_crisis_{target.lower()}_{panel.lower()}"
            paths_fig = _style.save_fig(
                fig,
                figure_id,
                formats=["png", "pdf"],
                used_in="EXPLORATORY D-73; scoring outside 2021-2023; no 2020",
            )
            plt.close(fig)
            cards.append((target, panel, paths_fig))
    metrics = pd.DataFrame(metrics)
    winners = pd.DataFrame(winners)
    metrics.to_csv(OUT / "scores.csv", index=False)
    winners.to_csv(OUT / "winners.csv", index=False)
    pd.DataFrame(cumulative).to_csv(OUT / "cumulative_ratios.csv", index=False)
    pd.DataFrame(endpoints).to_csv(OUT / "endpoint_checks.csv", index=False)
    base = metrics[metrics["sample"].eq("primary")].drop(columns="sample")
    changes = metrics[~metrics["sample"].eq("primary")].merge(
        base, on=["target", "panel", "model"], suffixes=("", "_primary"), validate="many_to_one"
    )
    changes["MSE_change_percent"] = 100 * (changes.MSE / changes.MSE_primary - 1)
    changes.to_csv(OUT / "changes_from_primary.csv", index=False)
    text = [
        "# Inflation outside the energy-crisis episode (D-73)",
        (
            "EXPLORATORY evaluation-only sensitivity. Three inflation targets in M, MQ, ME and MQE, separately. "
            "The primary experiment remains the reference. 2020 target months are excluded in every sample."
        ),
        "## Fixed exclusion windows",
        (
            "Main sensitivity: January 2021-December 2023, a broad calendar-year envelope for the 2021-2023 "
            "energy crisis described in Commission COM(2025)72. Boundary sensitivity: February 2022-December 2023, "
            "covering the war escalation and subsequent adjustment. These are operational bounds, not a universal "
            "official break date. They were fixed before computing this sensitivity and applied uniformly."
        ),
        "Sources: " + " ; ".join(cfg["sources"]),
        "## Winners by information set and sample",
        markdown(winners.round(6)),
        (
            "FarmPredict includes both saved identifiers FarmPredict (DFM+LASSO) and PCA_VAR_LASSO (PCA+VAR+LASSO). "
            "The PCA variant is estimated only in M and ME. Margins are descriptive, not significance tests. "
            "MSE levels across different scoring samples also reflect changed target volatility, not just model quality."
        ),
        (
            "No models are refitted or retuned. Exclusions affect scoring only; crisis data remain in later expanding "
            "training samples. This is not a counterfactual economy without the crisis. The check was motivated by "
            "observed primary errors and is not an independent validation sample. Cumulated losses resume after each "
            "gap using the pre-gap accumulated values; they are not restarted."
        ),
        "Reproduce: python -m scripts.inflation_crisis_sensitivity. No LaTeX or estimator configuration changes.",
    ]
    (OUT / "README.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    links = [
        f'<h2>{NAMES[t]} — {p}</h2><a href="../../figures/{Path(files[1]).name}">PDF</a><br><img src="../../figures/{Path(files[0]).name}" width="1100">'
        for t, p, files in cards
    ]
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Inflation crisis sensitivity</title><body style="font-family:system-ui;margin:2rem"><h1>Inflation outside 2021-2023</h1><p>2020 also excluded; saved forecasts, training unchanged. Below 1 favors the named FarmPredict variant.</p>'
        + "".join(links)
        + "</body></html>",
        encoding="utf-8",
    )
    for path in paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.as_posix()]
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "sources": hashes,
                "configuration": cfg,
                "endpoint_checks": len(endpoints),
                "figures": len(cards),
                "estimators_refitted": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "inflation_crisis_sensitivity",
        [str(OUT / "README.md"), str(OUT / "index.html")],
        "EXPLORATORY D-73; three inflation targets",
    )
    print(winners.to_string(index=False))
    print(
        f"Generated {len(cards)} figures, verified {len(endpoints)} endpoints; source forecasts unchanged."
    )


if __name__ == "__main__":
    main()
