"""Draft results figures for the paper (D-68 run with the D-72 outlier rule).

fig_90_sparse_step       Question 1: change in MSE when the sparse step is added, as a dot
                         plot (FarmPredict vs DFM, PCA-VAR-LASSO vs PCA-VAR, FarmPredict vs LASSO).
fig_91_information_sets  Question 2: change in MSE when quarterly (Q) or external (E) series are
                         added to the monthly panel, by model and target.
fig_92_mse_heatmap       The full MSE matrix of each target, models grouped by family.

Run from the repo root:  uv run python -m scripts.results_figures
Reads only saved outputs (mse_<target>.csv, statistical_tests.csv); estimates nothing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import norm

from src._style import SIZES, save_fig, stars

RUN = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
TARGETS = {
    "HICPOV_EA": "Headline HICP",
    "HICPNG_EA": "Energy HICP",
    "HICPNEF_EA": "Core HICP",
    "IPMN_EA": "Manufacturing",
}
SHORT = {"HICPOV_EA": "Headline", "HICPNG_EA": "Energy", "HICPNEF_EA": "Core", "IPMN_EA": "Manuf."}
SETS = ["M", "MQ", "ME", "MQE"]
NAMES = {
    "RW": "RW",
    "AR": "AR",
    "LASSO": "LASSO",
    "DFM": "DFM",
    "DFM_AR": "DFM-AR",
    "FarmPredict": "FP-DFM",
    "PCR": "PCR",
    "PCA_VAR": "PCA-VAR",
    "PCA_VAR_LASSO": "FP-PCA",
}
# Rows after which a family ends (white gap): benchmarks | sparse | DFM family | PCA family.
FAMILY_ENDS = {"AR", "LASSO", "FarmPredict"}
BETTER, WORSE = "#2166AC", "#B2182B"  # Tol BuRd poles: lower and higher MSE
# Tol BuRd, a colorblind-safe diverging scheme: blue = lower MSE, red = higher.
DIVERGING = LinearSegmentedColormap.from_list(
    "tol_burd",
    ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D"]
    + ["#B2182B"],
)
# Light single-hue ramp for MSE levels, light enough for dark text in every cell.
SEQUENTIAL = LinearSegmentedColormap.from_list("levels", ["#F7FBFF", "#C6DBEF", "#6BAED6"])
MISSING = "#EEEEEE"
INK = "#222222"
MUTED = "#777777"
CLIP = 25  # % change at which the diverging colors saturate; values are printed in full
SIGNIFICANT = 0.10


def mse(target):
    return pd.read_csv(RUN / f"mse_{target}.csv", index_col=0)


def tests():
    t = pd.read_csv(RUN / "statistical_tests.csv")
    return t[t["sample"] == "primary"].set_index(["target", "augmented", "baseline"])


def tested_change(t, target, model, base, big, small, one_sided):
    """% change in MSE of model@big against base@small and its p-value, checked against the CSVs."""
    m = mse(target)
    a, b = m.loc[model, big], m.loc[base, small]
    if pd.isna(a) or pd.isna(b):
        return np.nan, np.nan
    row = t.loc[(target, f"{big}:{model}", f"{small}:{base}")]
    np.testing.assert_allclose(row.MSE_ratio, a / b, rtol=1e-9)
    # Clark-West is one-sided (H1: the nesting model has the lower MSE); DM is two-sided.
    p = norm.sf(row.statistic) if one_sided else row.pvalue_two_sided
    return 100 * (a / b - 1), p


def signed(value, p):
    return f"{value:+.1f}{stars(p)}".replace("-", "−")


def ink_for(rgba):
    """Dark text on light cells, white on dark ones."""
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.45 else INK


def heat(ax, values, labels, rows, cols, cmap, norm_, bold=None, gaps=()):
    """Annotated heatmap; NaN cells are grey with a dash; white gaps after the rows in `gaps`."""
    cmap = cmap.copy()
    cmap.set_bad(MISSING)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm_, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#999999", fontsize=7)
                continue
            text = ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7)
            text.set_color(ink_for(cmap(norm_(values[i, j]))))
            if bold is not None and bold[i, j]:
                text.set_fontweight("bold")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, len(cols)):
        ax.axvline(x, color="white", linewidth=1.5)
    for k, row in enumerate(rows):
        ax.axhline(k + 0.5, color="white", linewidth=4 if row in gaps else 1.5)
    return image


def sparse_step_figure():
    """Question 1 as a dot plot: does adding the sparse step lower the MSE?"""
    t = tests()
    panels = [
        ("(a) Sparse step on DFM factors", "FarmPredict vs DFM", "FarmPredict", "DFM", True),
        (
            "(b) Sparse step on PCA factors",
            "PCA-VAR-LASSO vs PCA-VAR",
            "PCA_VAR_LASSO",
            "PCA_VAR",
            True,
        ),
        ("(c) Hybrid vs sparse only", "FarmPredict vs LASSO", "FarmPredict", "LASSO", False),
    ]
    # One row per target and information set, targets separated by a gap.
    y, labels, groups = [], [], []
    for i, target in enumerate(TARGETS):
        top = i * (len(SETS) + 1.2)
        groups.append((target, top))
        for j, s in enumerate(SETS):
            y.append(top + j)
            labels.append(s)
    fig, axes = plt.subplots(1, 3, figsize=(SIZES["full"][0], 3.5), sharey=True)
    for ax, (title, subtitle, model, base, one_sided) in zip(axes, panels, strict=True):
        k = 0
        values = []
        for target in TARGETS:
            for s in SETS:
                value, p = tested_change(t, target, model, base, s, s, one_sided)
                values.append(value)
                if np.isnan(value):
                    ax.text(
                        0,
                        y[k],
                        "not estimated",
                        ha="center",
                        va="center",
                        color=MUTED,
                        fontsize=6,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                        zorder=4,
                    )
                else:
                    color = BETTER if value < 0 else WORSE
                    ax.plot([0, value], [y[k], y[k]], color=color, linewidth=1.2, alpha=0.5)
                    face = color if p < SIGNIFICANT else "white"
                    ax.plot(
                        value,
                        y[k],
                        "o",
                        markersize=5,
                        markerfacecolor=face,
                        markeredgecolor=color,
                        markeredgewidth=1.1,
                        zorder=3,
                    )
                k += 1
        span = np.nanmax(np.abs(values)) * 1.15
        ax.set_xlim(-span, span)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{title}\n", loc="left")
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=7)
        # Palatino has no arrow glyphs, so the arrows are drawn.
        for start, end, text, color, align in [
            (0.12, 0.01, "lower MSE", BETTER, "left"),
            (0.88, 0.99, "higher MSE", WORSE, "right"),
        ]:
            ax.annotate(
                text,
                xy=(end, -0.11),
                xytext=(start, -0.11),
                xycoords="axes fraction",
                color=color,
                fontsize=6.5,
                ha=align,
                va="center",
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
            )
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel("Change in MSE (%)", labelpad=12)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    for target, top in groups:
        axes[0].text(
            -0.36,
            top + 1.5,
            SHORT[target],
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontweight="bold",
        )
    for ax in axes:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    legend = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK,
            markerfacecolor=INK,
            markersize=5,
            label="significant at 10%",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=INK,
            markerfacecolor="white",
            markersize=5,
            label="not significant",
        ),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0.04, 0.03, 1, 1))
    save_fig(fig, "fig_90_sparse_step", used_in="draft: results, question 1")
    plt.close(fig)


def information_figure():
    """Question 2: change in MSE when Q or E is added to the monthly panel M."""
    models = ["LASSO", "DFM", "DFM_AR", "FarmPredict", "PCR", "PCA_VAR", "PCA_VAR_LASSO"]
    panels = [
        ("(a) Adding quarterly series", "MQ vs M", "MQ"),
        ("(b) Adding external series", "ME vs M", "ME"),
    ]
    norm_ = TwoSlopeNorm(vcenter=0, vmin=-CLIP, vmax=CLIP)
    fig, axes = plt.subplots(
        1, 2, figsize=(SIZES["full"][0], 2.9), sharey=True, width_ratios=[4, 3]
    )
    for ax, (title, subtitle, big) in zip(axes, panels, strict=True):
        targets = [target for target in TARGETS if big == "MQ" or target != "IPMN_EA"]
        values = np.full((len(models), len(targets)), np.nan)
        labels = [[""] * len(targets) for _ in models]
        for i, model in enumerate(models):
            for j, target in enumerate(targets):
                scores = mse(target)
                augmented, baseline = scores.loc[model, big], scores.loc[model, "M"]
                value = 100 * (augmented / baseline - 1)
                values[i, j] = value
                if not np.isnan(value):
                    labels[i][j] = f"{value:+.1f}".replace("-", "−")
        names = [NAMES[x] for x in models]
        image = heat(
            ax,
            values,
            labels,
            names,
            [SHORT[target] for target in targets],
            DIVERGING,
            norm_,
            gaps={"LASSO", "FP-DFM"},
        )
        ax.set_title(f"{title}\n\n", loc="left")
        ax.text(0, 1.13, subtitle, transform=ax.transAxes, color=MUTED, fontsize=7)
    bar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.06, pad=0.04, aspect=50, extend="both"
    )
    bar.outline.set_visible(False)
    bar.set_ticks([-CLIP, -10, 0, 10, CLIP])
    bar.ax.tick_params(labelsize=6.5, length=0)
    bar.set_label(
        "Change in MSE (%):  blue = lower MSE (better),  red = higher MSE (worse)", fontsize=7
    )
    save_fig(fig, "fig_91_information_sets", used_in="draft: results, question 2")
    plt.close(fig)


# MSE scale per target, so every panel prints two decimals with three to four digits.
SCALE = {"HICPOV_EA": 100, "HICPNG_EA": 1, "HICPNEF_EA": 100, "IPMN_EA": 1}
# Tol YlOrBr (sequential, colorblind-safe), stopped at orange so dark text stays readable.
WARM = LinearSegmentedColormap.from_list(
    "warm", ["#FB9A29", "#FEC44F", "#FEE391", "#FFF7BC", "#FFFFF2"]
)
NEAR = 1.5  # MSE at least 50% above the target's best gets the lightest color
SHARED = ("RW", "AR")  # one forecast per target, the same in every information set


def matrix_panel(ax, m, scale):
    """One target: cells colored by the distance from the target's best MSE."""
    best = np.nanmin(m.to_numpy(dtype=float))
    norm_ = plt.Normalize(vmin=1, vmax=NEAR)
    column_best = {s: m[s].idxmin() for s in SETS}
    overall = m.stack().idxmin()
    y, ticks = 0.0, []
    for model in m.index:
        ticks.append(y)
        cells = [(j, j, s) for j, s in enumerate(SETS)]
        if model in SHARED:
            assert m.loc[model].nunique() == 1
            cells = [(0, len(SETS) - 1, SETS[0])]
        for first, last, s in cells:
            value = m.loc[model, s]
            left, width = first - 0.5 + 0.04, last - first + 1 - 0.08
            if pd.isna(value):
                ax.text((first + last) / 2, y, "\u2013", ha="center", va="center", color="#BBBBBB")
                continue
            color = WARM(norm_(min(value / best, NEAR)))
            ax.add_patch(plt.Rectangle((left, y - 0.46), width, 0.92, color=color, linewidth=0))
            text = ax.text(
                (first + last) / 2,
                y,
                f"{scale * value:.2f}",
                ha="center",
                va="center",
                color=INK,
                fontsize=7,
            )
            if any(column_best[c] == model for c in SETS[first : last + 1]):
                text.set_fontweight("bold")
            if (model, s) == overall:
                ax.add_patch(
                    plt.Rectangle(
                        (left, y - 0.46), width, 0.92, fill=False, edgecolor=INK, linewidth=1.0
                    )
                )
        y += 1.0 + (0.45 if model in FAMILY_ENDS else 0.0)
    ax.set_xlim(-0.5, len(SETS) - 0.5)
    ax.set_ylim(y - 0.5, -0.6)
    ax.set_xticks(range(len(SETS)), SETS)
    ax.xaxis.tick_top()
    ax.set_yticks(ticks, [NAMES[x] for x in m.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return norm_


def mse_figure():
    """The full MSE matrix of each target, models grouped by family."""
    models = list(NAMES)
    fig, axes = plt.subplots(
        2, 2, figsize=(SIZES["full"][0], 6.0), sharey=True, gridspec_kw={"hspace": 0.3}
    )
    for k, (ax, (target, label)) in enumerate(zip(axes.flat, TARGETS.items(), strict=True)):
        scale = SCALE[target]
        unit = "MSE \u00d7 100" if scale == 100 else "MSE"
        norm_ = matrix_panel(ax, mse(target).loc[models, SETS], scale)
        ax.set_title(f"({'abcd'[k]}) {label}, {unit}", loc="left", pad=16)
    mappable = plt.cm.ScalarMappable(norm=norm_, cmap=WARM)
    bar = fig.colorbar(
        mappable, ax=axes, orientation="horizontal", fraction=0.03, pad=0.03, aspect=60
    )
    bar.outline.set_visible(False)
    bar.set_ticks(
        [1, 1.1, 1.2, 1.3, 1.4, 1.5],
        labels=["best", "+10%", "+20%", "+30%", "+40%", "+50%+"],
    )
    bar.ax.tick_params(labelsize=6.5, length=0)
    bar.set_label(
        "MSE relative to the best specification of the target.  Bold: lowest MSE in the "
        "column.  Frame: lowest MSE of the target.",
        fontsize=7,
    )
    save_fig(fig, "fig_92_mse_heatmap", used_in="draft: results, full MSE matrix")
    plt.close(fig)


def main():
    sparse_step_figure()
    information_figure()
    mse_figure()


if __name__ == "__main__":
    main()
