"""Plot selected rows of the saved ladder comparisons without changing estimates."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm

from scripts.results_ladder_exhibits import NAMES, RUN, primary_frame, years
from src import _style

OUT = Path("outputs/reports/ladder_steps_display")
PANELS = [
    ("FarmPredict", "DFM", "(a) Sparse step on DFM factors", "FP-DFM vs DFM"),
    ("PCA_VAR_LASSO", "PCA_VAR", "(b) Sparse step on PCA factors", "FP-PCA vs PCA-VAR"),
    ("DFM_AR", "DFM", "(c) Own lag on DFM factors", "DFM-AR vs DFM"),
]


def main():
    sources = [RUN / name for name in ["forecasts.csv", "coverage.csv", "statistical_tests.csv"]]
    sources += [Path(__file__), Path("scripts/results_ladder_exhibits.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    forecasts = pd.read_csv(RUN / "forecasts.csv")
    coverage = pd.read_csv(RUN / "coverage.csv")
    tests = pd.read_csv(RUN / "statistical_tests.csv")
    tests = tests[tests["sample"].eq("primary")].set_index(["target", "augmented", "baseline"])
    positions, centers, boundaries = {}, {}, []
    cursor = 0
    for target in NAMES:
        sets = ["M", "MQ"] if target == "IPMN_EA" else ["M", "MQ", "ME"]
        first = cursor
        for info in sets:
            positions[(target, info)] = -cursor
            cursor += 1
        centers[target] = -(first + cursor - 1) / 2
        if target != list(NAMES)[-1]:
            boundaries.append(-cursor)
        cursor += 1
    assert len(positions) == 11
    assert not any(
        info == "MQE" or (target == "IPMN_EA" and info == "ME") for target, info in positions
    )
    frames = {target: primary_frame(forecasts, coverage, target) for target in NAMES}
    fig, axes = plt.subplots(1, 3, figsize=(_style.SIZES["full"][0], 3.4), sharex=True, sharey=True)
    checks = []
    for ax, (augmented, baseline, title, pair) in zip(axes, PANELS, strict=True):
        for (target, info), y0 in positions.items():
            if augmented == "PCA_VAR_LASSO" and info == "MQ":
                continue
            frame = frames[target]
            a, b = f"{info}:{augmented}", f"{info}:{baseline}"
            ratio = frame["error"][a].pow(2).sum() / frame["error"][b].pow(2).sum()
            outside = (years(frame.index) < 2021) | (years(frame.index) > 2023)
            ratio_out = (
                frame["error"][a][outside].pow(2).sum() / frame["error"][b][outside].pow(2).sum()
            )
            saved = tests.loc[(target, a, b)]
            np.testing.assert_allclose(ratio, saved.MSE_ratio, rtol=1e-9)
            assert saved.test == "Clark_West"
            p = float(norm.sf(saved.statistic))
            color = _style.PALETTE[1 if info == "ME" else 0]
            gain, gain_out = 100 * (1 - ratio), 100 * (1 - ratio_out)
            ax.plot([gain, gain_out], [y0, y0], color=color, linewidth=0.6, alpha=0.5, zorder=2)
            ax.scatter(gain_out, y0, s=14, marker="x", color=color, linewidths=0.8, zorder=3)
            ax.scatter(
                gain,
                y0,
                s=22,
                marker="o",
                color=color,
                zorder=4,
                facecolors=color if p < 0.05 else "white",
                linewidths=0.9,
            )
            checks.append(
                {
                    "target": target,
                    "set": info,
                    "augmented": augmented,
                    "baseline": baseline,
                    "gain_percent": gain,
                    "outside_gain_percent": gain_out,
                    "one_sided_p": p,
                }
            )
        ax.axvline(0, color="#555555", linewidth=0.6)
        ax.set_title(f"{title}\n{pair}", fontsize=7)
        ax.set_xlabel("MSE reduction (%); right is better")
        ax.grid(axis="x", color="#E3E3E3", linewidth=0.5)
        ax.grid(axis="y", visible=False)
        for y in boundaries:
            ax.axhline(y, color="#CCCCCC", linewidth=0.5)
    assert len(checks) == 29
    axes[0].set_xlim(-9, 16)
    axes[0].set_yticks(list(positions.values()), [info for _, info in positions])
    for target, center in centers.items():
        axes[0].annotate(
            NAMES[target],
            xy=(-0.2, center),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="center",
            fontweight="bold",
            fontsize=7,
        )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=_style.PALETTE[0],
            label="M, MQ (no external series)",
        ),
        Line2D(
            [], [], marker="o", linestyle="", color=_style.PALETTE[1], label="ME (inflation only)"
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="#555555",
            markerfacecolor="white",
            label="hollow: not significant at 5%",
        ),
        Line2D([], [], marker="x", linestyle="", color="#555555", label="without 2021–2023"),
    ]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0.06, 0.07, 1, 1), w_pad=1.0)
    old = _style.OUT
    try:
        _style.OUT = OUT
        _style.save_fig(
            fig,
            "fig_198_ladder_steps",
            formats=["pdf", "png"],
            used_in="Results: MQE omitted; manufacturing M/MQ only",
        )
    finally:
        _style.OUT = old
        plt.close(fig)
    pd.DataFrame(checks).to_csv(OUT / "plotted_values.csv", index=False)
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "sources": hashes,
                "displayed_rows": len(positions),
                "checked_comparisons": len(checks),
                "model_refits": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "ladder_steps_display",
        [str(OUT / "figures/fig_198_ladder_steps.pdf"), str(OUT / "audit.json")],
        "Figure display restricted to M/MQ/ME for inflation, M/MQ for manufacturing",
    )
    print(f"Saved figure: {len(positions)} rows, {len(checks)} checked comparisons")


if __name__ == "__main__":
    main()
