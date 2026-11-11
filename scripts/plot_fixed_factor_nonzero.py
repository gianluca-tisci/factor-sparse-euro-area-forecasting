"""Plot ME/MQE fixed-factor ablations against saved primary comparators."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.report_m_mq_winners import TARGETS
from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/nonzero_robustness")


def main():
    if (OUT / "index.html").exists():
        raise FileExistsError("Refuse to overwrite a saved plot report")
    ablations = {
        "ME": Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr"),
        "MQE": OUT / "mqe_ablation",
    }
    sources = [ROOT / name for name in ["forecasts.csv", "coverage.csv", "scores.csv"]]
    sources += [Path("scripts/plot_fixed_factor_nonzero.py"), Path("src/_style.py")]
    sources += [
        path / name for path in ablations.values() for name in ["forecasts.csv", "manifest.json"]
    ]
    hashes = {
        str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    original = pd.read_csv(ROOT / "forecasts.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    scores = pd.read_csv(ROOT / "scores.csv")
    ratios, checks, links, losses = [], [], [], []
    for i, (target, title) in enumerate(TARGETS.items()):
        dates = coverage[coverage.target.eq(target) & coverage["sample"].eq("primary")]
        assert dates.common_dates.nunique() == 1
        dates = dates.common_dates.iloc[0].split(";")
        for j, (panel, directory) in enumerate(ablations.items()):
            base = "M" if panel == "ME" else "MQ"
            fixed = pd.read_csv(directory / "forecasts.csv")
            fixed = (
                fixed[fixed.target.eq(target) & fixed.origin.isin(dates)]
                .set_index("origin")
                .loc[dates]
            )
            assert len(fixed) == len(dates) and fixed.index.is_unique
            assert not fixed.target_month.str.startswith("2020-").any()
            assert fixed.target_month.le("2025-12").all()
            np.testing.assert_allclose(fixed.actual - fixed.forecast, fixed.error)
            np.testing.assert_allclose(
                fixed.common_forecast + fixed.sparse_correction, fixed.forecast
            )
            specs = [
                (base, "DFM", f"DFM-{base}"),
                (base, "FarmPredict", f"FarmPredict-{base}"),
                (panel, "FarmPredict", f"FarmPredict-{panel}"),
                (panel, "LASSO", f"LASSO-{panel}"),
                ("shared", "AR", "AR"),
            ]
            fig, ax = plt.subplots(figsize=(9, 6))
            for (info, model, label), color in zip(specs, _style.PALETTE_MANY, strict=True):
                data = original[
                    original.target.eq(target)
                    & original.information_set.eq(info)
                    & original.model.eq(model)
                    & original.origin.isin(dates)
                ]
                assert not data.origin.duplicated().any()
                data = data.set_index("origin").loc[dates]
                assert data.valid.all() and data.target_month.equals(fixed.target_month)
                np.testing.assert_allclose(data.actual, fixed.actual, rtol=1e-12)
                if model == "DFM":
                    np.testing.assert_allclose(data.forecast, fixed.common_forecast, rtol=1e-12)
                a, b = fixed.error.pow(2), data.error.pow(2)
                primary = scores[
                    scores.target.eq(target)
                    & scores.information_set.eq(info)
                    & scores.model.eq(model)
                    & scores["sample"].eq("primary")
                ].iloc[0]
                np.testing.assert_allclose(b.mean(), primary.MSE, rtol=1e-10)
                curve = a.cumsum() / b.cumsum()
                expected = a.mean() / primary.MSE
                np.testing.assert_allclose(curve.iloc[-1], expected, rtol=1e-10)
                checks.append(
                    {
                        "target": target,
                        "panel": panel,
                        "benchmark": label,
                        "n": len(dates),
                        "ablation_MSE": a.mean(),
                        "benchmark_MSE": b.mean(),
                        "final_ratio": curve.iloc[-1],
                        "reduction_percent": 100 * (1 - expected),
                    }
                )
                for origin in dates:
                    ratios.append(
                        {
                            "target": target,
                            "panel": panel,
                            "benchmark": label,
                            "origin": origin,
                            "target_month": fixed.loc[origin, "target_month"],
                            "cumulative_MSE_ratio": curve.loc[origin],
                        }
                    )
                time = pd.PeriodIndex(dates, freq="M").to_timestamp()
                plot = pd.Series(curve.to_numpy(), index=time).reindex(
                    pd.date_range(time.min(), time.max(), freq="MS")
                )
                ax.plot(plot.index, plot, label=label, color=color, linewidth=1.3)
            for origin, value in fixed.iterrows():
                losses.append(
                    {
                        "target": target,
                        "panel": panel,
                        "origin": origin,
                        "target_month": value.target_month,
                        "squared_error": value.error**2,
                    }
                )
            ax.axhline(1, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_yscale("log")
            ax.set_ylabel("Cumulative MSE ratio (log scale)")
            ax.set_xlabel("Forecast origin")
            ax.set_title(f"{title} ({target}) — fixed {base} factors + sparse {panel}")
            ax.legend(fontsize=8)
            fig.text(
                0.02,
                0.015,
                f"Notes: {len(dates)} matched primary dates; target months through 2025-12; 2020 excluded.\n"
                f"Numerator: fixed DFM-{base} + sparse({'+'.join(panel)} residuals); denominator: named benchmark.\n"
                "Below 1 favors the ablation. E is projected on fixed factors; no EM refits. Gaps do not reset accumulation.\n"
                "Source: saved cold paper experiment, D-70 ME ablation and MQE extension.",
                fontsize=7,
            )
            fig.tight_layout(rect=(0, 0.15, 1, 1))
            paths = _style.save_fig(
                fig,
                f"fig_{127 + 2 * i + j}_fixed_factors_{target.lower()}_{panel.lower()}",
                formats=["png", "pdf"],
                used_in="External fixed-factor ablation; no 2020",
            )
            plt.close(fig)
            links.append((title, panel, paths))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ratios).to_csv(OUT / "cumulative_ratios.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "endpoint_checks.csv", index=False)
    pd.DataFrame(losses).to_csv(OUT / "ablation_squared_errors.csv", index=False)
    for name, value in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "sample": "primary_without_2020",
                "dfm_refits": 0,
                "numerator": "Fixed DFM-M or MQ plus residual sparse stage with E",
                "panels": ["ME", "MQE"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cards = []
    for title, panel, paths in links:
        png, pdf = ["../../figures/" + Path(p).name for p in paths]
        cards.append(
            f'<h2>{title} — {panel}</h2><a href="{pdf}">PDF</a><br><img src="{png}" width="1000">'
        )
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-factor external ablations</title>'
        '<body style="font-family:system-ui;margin:2rem"><h1>Fixed-factor ablations: ME and MQE</h1>'
        "<p>Four original targets; 2020 excluded. Ratio below one favors the fixed-factor ablation. "
        "ME keeps M factors; MQE keeps MQ factors. External predictors are residualized, not raw. "
        "FarmPredict labels in these graphs refer to the DFM variant. Logarithmic vertical axis.</p>"
        '<p><a href="endpoint_checks.csv">MSE comparisons</a> | <a href="cumulative_ratios.csv">All ratios</a></p>'
        + "".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )
    _style.record_output(
        "fixed_factor_external_cumulative",
        [str(OUT / "index.html"), str(OUT / "endpoint_checks.csv")],
        "ME/MQE fixed-factor ALTERNATIVE; no LaTeX",
    )
    print(f"Generated {len(links)} figures and verified {len(checks)} endpoint ratios.")


if __name__ == "__main__":
    main()
