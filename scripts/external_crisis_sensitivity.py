"""Evaluate saved ME/MQE forecasts and fixed-factor ablations outside the crisis."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.report_mq_energy_diagnostics import markdown
from src import _style

CONFIG = Path("config/evaluation_external_crisis.yaml")
NAMES = {"HICPOV_EA": "Headline", "HICPNG_EA": "Energy", "HICPNEF_EA": "Core"}


def scoring_samples(months, exclusions):
    """Select by target month, preserving the original calendar and ordering."""
    return {
        "primary": np.ones(len(months), dtype=bool),
        **{
            name: ~months.between(window["start"], window["end"]).to_numpy()
            for name, window in exclusions.items()
        },
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    root, out = Path(cfg["source_run"]), Path(cfg["output"])
    if out.exists():
        raise FileExistsError("Use a new evaluation directory; never overwrite results")
    paths = [
        CONFIG,
        Path("scripts/external_crisis_sensitivity.py"),
        Path("src/_style.py"),
        Path("scripts/report_mq_energy_diagnostics.py"),
    ]
    paths += [root / name for name in ["forecasts.csv", "coverage.csv", "scores.csv"]]
    paths += [
        Path(folder) / name
        for folder in cfg["ablation_runs"].values()
        for name in ["forecasts.csv", "manifest.json"]
    ]
    paths += [root / panel / "metadata.json" for panel in ["M", "MQ", "ME", "MQE"]]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original = pd.read_csv(root / "forecasts.csv")
    coverage = pd.read_csv(root / "coverage.csv")
    gold = pd.read_csv(root / "scores.csv")
    metrics, comparisons, winners, curves, endpoints, cards, selected_dates = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, target in enumerate(cfg["targets"]):
        cov = coverage[coverage.target.eq(target) & coverage["sample"].eq("primary")]
        assert cov.common_dates.nunique() == 1
        dates = cov.common_dates.iloc[0].split(";")
        data = original[original.target.eq(target) & original.origin.isin(dates)].copy()
        assert data.valid.all()
        data["spec"] = data.information_set + ":" + data.model
        assert not data.duplicated(["origin", "spec"]).any()
        pivot = data.pivot(index="origin", columns="spec", values="error").loc[dates]
        reference = data[data.model.eq("AR")].set_index("origin").loc[dates]
        assert (
            not reference.target_month.str[:4]
            .astype(int)
            .isin(cfg["always_exclude_target_years"])
            .any()
        )
        assert reference.target_month.le(cfg["score_through"]).all()
        for panel, folder in cfg["ablation_runs"].items():
            fixed = pd.read_csv(Path(folder) / "forecasts.csv")
            fixed = (
                fixed[fixed.target.eq(target) & fixed.origin.isin(dates)]
                .set_index("origin")
                .loc[dates]
            )
            assert len(fixed) == len(dates) and fixed.index.is_unique
            assert fixed.target_month.equals(reference.target_month)
            np.testing.assert_allclose(fixed.actual, reference.actual, rtol=1e-12)
            np.testing.assert_allclose(fixed.actual - fixed.forecast, fixed.error)
            base = "M" if panel == "ME" else "MQ"
            base_rows = (
                data[data.information_set.eq(base) & data.model.eq("DFM")]
                .set_index("origin")
                .loc[dates]
            )
            np.testing.assert_allclose(fixed.common_forecast, base_rows.forecast, rtol=1e-12)
            pivot[panel + ":Fixed_factor"] = fixed.error
        assert np.isfinite(pivot.to_numpy()).all()
        baseline = gold[gold.target.eq(target) & gold["sample"].eq("primary")].copy()
        baseline["spec"] = baseline.information_set + ":" + baseline.model
        np.testing.assert_allclose(pivot.pow(2).mean()[baseline.spec], baseline.MSE, rtol=1e-10)
        masks = scoring_samples(reference.target_month, cfg["exclusions"])
        for sample, mask in masks.items():
            errors = pivot.loc[mask]
            assert (
                len(errors)
                == {"primary": 120, "outside_energy_crisis": 84, "outside_war_phase": 97}[sample]
            )
            for origin in errors.index:
                selected_dates.append(
                    {
                        "target": target,
                        "sample": sample,
                        "origin": origin,
                        "target_month": reference.loc[origin, "target_month"],
                    }
                )
            mse = errors.pow(2).mean()
            sums = errors.pow(2).cumsum()
            for j, panel in enumerate(cfg["ablation_runs"]):
                base = "M" if panel == "ME" else "MQ"
                available = [c for c in mse.index if c.startswith((panel + ":", "shared:"))]
                extra = [base + ":DFM", base + ":FarmPredict"]
                for spec in available + extra:
                    metrics.append(
                        {
                            "target": target,
                            "sample": sample,
                            "panel": panel,
                            "specification": spec,
                            "n": len(errors),
                            "MSE": mse[spec],
                        }
                    )
                for menu in ["original", "including_fixed_factor"]:
                    columns = [
                        c
                        for c in available
                        if menu != "original" or not c.endswith(":Fixed_factor")
                    ]
                    ranked = mse[columns].sort_values()
                    winners.append(
                        {
                            "target": target,
                            "sample": sample,
                            "panel": panel,
                            "menu": menu,
                            "winner": ranked.index[0],
                            "MSE": ranked.iloc[0],
                            "n": len(errors),
                            "runner_up": ranked.index[1],
                            "runner_up_MSE": ranked.iloc[1],
                            "reduction_vs_second_percent": 100
                            * (1 - ranked.iloc[0] / ranked.iloc[1]),
                        }
                    )
                for numerator in [panel + ":Fixed_factor", panel + ":FarmPredict"]:
                    for denominator in available + extra:
                        if denominator == numerator:
                            continue
                        ratio = sums[numerator] / sums[denominator]
                        expected = mse[numerator] / mse[denominator]
                        np.testing.assert_allclose(ratio.iloc[-1], expected, rtol=1e-12)
                        row = {
                            "target": target,
                            "sample": sample,
                            "panel": panel,
                            "numerator": numerator,
                            "benchmark": denominator,
                            "n": len(errors),
                            "MSE": mse[numerator],
                            "benchmark_MSE": mse[denominator],
                            "final_ratio": expected,
                            "reduction_percent": 100 * (1 - expected),
                        }
                        comparisons.append(row)
                        endpoints.append(row)
                        for origin, value in ratio.items():
                            curves.append(
                                {
                                    "target": target,
                                    "sample": sample,
                                    "panel": panel,
                                    "numerator": numerator,
                                    "benchmark": denominator,
                                    "origin": origin,
                                    "cumulative_MSE_ratio": value,
                                }
                            )
                if sample == "primary":
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
                for ax, numerator in zip(
                    axes, [panel + ":Fixed_factor", panel + ":FarmPredict"], strict=True
                ):
                    comparators = [
                        base + ":DFM",
                        base + ":FarmPredict",
                        panel + (":FarmPredict" if numerator.endswith(":Fixed_factor") else ":DFM"),
                        panel + ":LASSO",
                        "shared:AR",
                    ]
                    for comparator, color in zip(comparators, _style.PALETTE_MANY, strict=True):
                        ratio = sums[numerator] / sums[comparator]
                        time = pd.PeriodIndex(ratio.index, freq="M").to_timestamp()
                        curve = pd.Series(ratio.to_numpy(), index=time).reindex(
                            pd.date_range(time.min(), time.max(), freq="MS")
                        )
                        ax.plot(curve.index, curve, color=color, linewidth=1.3, label=comparator)
                    ax.axhline(1, color="#555555", linestyle="--", linewidth=0.8)
                    ax.set_yscale("log")
                    ax.set_title(
                        "Fixed-factor ablation"
                        if numerator.endswith(":Fixed_factor")
                        else "Original expanded-panel FarmPredict"
                    )
                    ax.set_ylabel("Cumulative MSE ratio (log scale)")
                    ax.set_xlabel("Forecast origin")
                    ax.legend(fontsize=8)
                window = cfg["exclusions"][sample]
                fig.suptitle(
                    f"{NAMES[target]} ({target}) — {panel} — exclude {window['start']} to {window['end']}"
                )
                fig.text(
                    0.02,
                    0.015,
                    f"Notes: {len(errors)} matched dates; 2020 and the stated crisis window excluded; target months through 2025-12.\n"
                    "Ratio = named panel forecast / benchmark; below 1 favors the numerator. FarmPredict here means DFM + LASSO.\n"
                    f"Ablation fixes {base} factors and adds E residuals to the sparse stage. Training and saved forecasts unchanged.\n"
                    "Gaps do not reset accumulation or connect curves. Source: saved cold paper run and fixed-factor ablations.",
                    fontsize=7,
                )
                fig.tight_layout(rect=(0, 0.15, 1, 0.94))
                k = list(cfg["exclusions"]).index(sample)
                figure_id = f"fig_{135 + 6 * k + 2 * i + j}_external_crisis_{target.lower()}_{panel.lower()}_{sample}"
                paths_fig = _style.save_fig(
                    fig,
                    figure_id,
                    formats=["png", "pdf"],
                    used_in="EXPLORATORY external crisis scoring sensitivity",
                )
                plt.close(fig)
                cards.append((target, panel, sample, paths_fig))
    out.mkdir(parents=True)
    tables = {
        "scores": metrics,
        "comparisons": comparisons,
        "winners": winners,
        "cumulative_ratios": curves,
        "endpoint_checks": endpoints,
        "scoring_dates": selected_dates,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    comparison = pd.DataFrame(comparisons)
    key = comparison[
        comparison.numerator.str.endswith(":Fixed_factor")
        & comparison.benchmark.str.endswith(":LASSO")
    ]
    report = [
        "# External variables outside the crisis: fixed versus expanded factors",
        (
            "EXPLORATORY evaluation-only sensitivity for headline, energy and core in ME/MQE. "
            "Same broad exclusion January 2021-December 2023 and narrow exclusion February 2022-December 2023 "
            "as codex/inflation-crisis-sensitivity. Every sample also excludes 2020 target months. "
            "Primary evaluation remains the reference; no changes to training or model fitting."
        ),
        "## Fixed-factor ablation versus direct external-panel LASSO",
        markdown(key.round(6)),
        (
            "Positive reductions favor the ablation. MSE changes between samples also reflect different "
            "target volatility; relative performance on matched dates is the relevant comparison. "
            "Original and expanded model menus, including PCA + VAR + LASSO in ME, are ranked in winners.csv. "
            "FarmPredict is a family including both PCA and DFM variants; the two plotted numerators concern only DFM variants."
        ),
        (
            "The exclusions change scoring only. Crisis observations remain in subsequent expanding training windows. "
            "This is not a counterfactual economy without crisis, or an independent validation sample. "
            "The broad window is an operational envelope, not a uniform official break date for every target. "
            "No significance or causal claim follows from descriptive MSE differences."
        ),
        (
            "Both panels use the same dates. The fixed-factor case preserves the original M or MQ common forecast "
            "and projects E on its saved factors. It changes sparse candidates and potentially complete training rows. "
            "All plotted endpoints are checked against independently computed full retained-sample MSE ratios."
        ),
        "Reproduce: python -m scripts.external_crisis_sensitivity. No model refits or LaTeX edits.",
    ]
    (out / "README.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    html = [
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>External crisis sensitivity</title>'
            '<body style="font-family:system-ui;margin:2rem"><h1>ME/MQE outside crisis</h1>'
            "<p>Three inflation targets, fixed-factor ablation and original DFM-based FarmPredict. "
            "2020 excluded throughout. Training unchanged. Below one favors the named numerator.</p>"
            '<p><a href="README.md">Results</a> | <a href="comparisons.csv">MSE comparisons</a></p>'
        )
    ]
    for target, panel, sample, files in cards:
        png, pdf = ["../../figures/" + Path(p).name for p in files]
        html.append(
            f'<h2>{NAMES[target]} — {panel} — {sample}</h2><a href="{pdf}">PDF</a><br><img src="{png}" width="1150">'
        )
    (out / "index.html").write_text("".join(html) + "</body></html>", encoding="utf-8")
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (out / "audit.json").write_text(
        json.dumps(
            {
                "configuration": cfg,
                "sources": hashes,
                "model_refits": 0,
                "figures": len(cards),
                "endpoint_checks": len(endpoints),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "external_crisis_sensitivity",
        [str(out / "README.md"), str(out / "index.html")],
        "EXPLORATORY ME/MQE; fixed and original factors; no 2020",
    )
    print(key.to_string(index=False))
    print(f"Generated {len(cards)} figures; checked {len(endpoints)} endpoints.")


if __name__ == "__main__":
    main()
