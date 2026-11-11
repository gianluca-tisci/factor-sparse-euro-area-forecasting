"""Exhibits and numbers for Results 4.1 (the model ladder in M); saved forecasts only.

Complements scripts/reorganized_results_exhibits.py (Pietro Ganzerli), whose layout and colors
the ladder table and figure keep. Changes: model names as in the Methods section (PCA-VAR,
DFM-AR), the figure plotted by target month with plain axis labels, and the evidence his text
marks as open: every ladder step in every set, the outside-2021--2023 tests and the split of
each step by period. Estimates nothing.

Writes (rebuilt in place) outputs/reports/results_ladder/:
  tab_104_ladder_m          the ladder in M, MSE relative to AR, models grouped by factor step
  fig_200_gains_over_time   cumulative squared-error gain over AR in M, 2021-23 shaded
  tab_105_ladder_all_sets   appendix: every ladder step in every information set, with p-values
  tab_106_ladder_periods    appendix: every ladder step in M by period, and outside 2021-23
  fig_197_selection         appendix: how often each lasso selects a predictor, and the own lag
  fig_198_ladder_steps      main figure: gain from each ingredient by set, full and without 2021-23
  fig_199_energy_reversal   energy forecasts against the outcome, 2021-2023, set M
  tab_107_top_predictors    body: what each lasso selects in M (LASSO, FP-DFM, FP-PCA)
  tab_108_external_routes   body: where the external series enter, MSE relative to AR
  tab_109_fixed_factor_mse  appendix: MSE levels of the fixed-factor model and its comparisons
  fig_201_information_over_time  when the quarterly and external series help
  numbers.tex               macros quoted in 4.1 (prefix Lad)

Run from the repo root:  uv run python -m scripts.results_ladder_exhibits
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from scipy.stats import norm

from src import _style
from src.forecasting.report import loss_test

RUN = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/results_ladder")
# Fixed-factor model (E only in the sparse step), with the M and the MQ factors kept
FIXED_FACTOR = {
    "ME": Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr"),
    "MQE": Path("outputs/reports/nonzero_robustness/mqe_ablation"),
}
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
    "PCA_VAR": "PCA-VAR",
    "DFM": "DFM",
    "DFM_AR": "DFM-AR",
    "PCA_VAR_LASSO": "FP-PCA",
    "FarmPredict": "FP-DFM",
}
MACRO = {  # model keys inside macro names (letters only)
    "AR": "AR",
    "LASSO": "LASSO",
    "PCR": "PCR",
    "PCA_VAR": "PCAVAR",
    "DFM": "DFM",
    "DFM_AR": "DFMAR",
    "PCA_VAR_LASSO": "FPPCA",
    "FarmPredict": "FPDFM",
}
SETS = ["M", "MQ", "ME", "MQE"]
# Short plain names for the selection table; full names are in Appendix Table A1
SHORT_SERIES = {
    "Producer Price Index: Consumer Goods": "PPI, consumer goods",
    "Producer Price Index: Durable Consumer Goods": "PPI, durable consumer goods",
    "Producer Price Index: Energy": "PPI, energy",
    "3-Months Interest Rates": "3-month interest rate",
    "6-Months Interest Rates": "6-month interest rate",
    "Long-Term Interest Rates (EMU Criterion)": "Long-term interest rate",
    "Harmonized Index of Consumer Prices: Industrial Goods": "HICP, industrial goods",
    "Harmonized Index of Consumer Prices: All Items, no Energy&Food": "Core HICP",
    "Industrial Production Index: Capital Goods": "Production, capital goods",
    "Unemployment: Under 25 years (% active population)": "Youth unemployment",
    "Passenger’s Cars Registrations": "Car registrations",
    "Stock Price Index": "Stock prices",
    "Consumer Confidence Index": "Consumer confidence",
    "Business Confidence Index": "Business confidence",
    "Industrial Confidence Indicator": "Industrial confidence",
    "Construction Sentiment Indicator": "Construction sentiment",
    "Economic Sentiment Indicator": "Economic sentiment",
}
SHORT = {"Manufacturing": "Manuf."}  # column headers of the one-column ladder table
# Rows of the ladder table: each block adds one ingredient at a time
BLOCKS = [
    ("Benchmark", ["AR"]),
    ("No factors", ["LASSO"]),
    ("PCA factors", ["PCR", "PCA_VAR", "PCA_VAR_LASSO"]),
    ("DFM factors", ["DFM", "DFM_AR", "FarmPredict"]),
]
PCA_SETS = {"M", "ME"}  # PCR, PCA-VAR, FP-PCA: no quarterly series (Table 3)
# (macro key, label, larger model, smaller model, nested -> one-sided Clark-West)
STEPS = [
    ("SparsePCA", "FP-PCA / PCA-VAR", "PCA_VAR_LASSO", "PCA_VAR", True),
    ("SparseDFM", "FP-DFM / DFM", "FarmPredict", "DFM", True),
    ("OwnLag", "DFM-AR / DFM", "DFM_AR", "DFM", True),
    ("HybridLasso", "FP-DFM / LASSO", "FarmPredict", "LASSO", False),
    ("HybridPCALasso", "FP-PCA / LASSO", "PCA_VAR_LASSO", "LASSO", False),
]
# Periods of the target month (manufacturing starts in December 2014); their
# contributions add up to the full-sample reduction.
PERIODS = [
    ("BeforeTwenty", "Before 2020", 2014, 2019),
    ("FromTwentyOne", "2021--22", 2021, 2022),
    ("InTwentyThree", "2023", 2023, 2023),
    ("FromTwentyFour", "2024--25", 2024, 2025),
]
SHOW_FROM = pd.Timestamp("2016-01-01")
YTICKS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0]
# Notes under the tables: what is shown, marks, sample, source (paper conventions).
NOTE_SAMPLE = "Target months to December 2025, without 2020."
NOTE_FULL = "Full sample: target months to December 2025, without 2020"
NOTE_WITHOUT = "without 2021--2023: also without January 2021 to December 2023"
NOTE_SOURCE = "\\textit{Source:} Own calculations."
NOTES = {
    "tab_104_ladder_m": (
        "MSE relative to AR, information set M; lowest per target and sample in bold. "
        f"{NOTE_FULL}; {NOTE_WITHOUT} (for manufacturing, a descriptive split). Tests: "
        "Appendix Tables~\\ref{tab:ladder-all-sets} and~\\ref{tab:ladder-periods}. " + NOTE_SOURCE
    ),
    "tab_105_ladder_all_sets": (
        "MSE of the first model over that of the second, same information set; below 1: the "
        "first is more accurate. In parentheses: $p$-values, one-sided Clark--West for the three "
        "nested steps, two-sided Diebold--Mariano for the comparisons with LASSO; * 0.10, "
        "** 0.05, *** 0.01, not adjusted for multiple testing. ---: the PCA models use no "
        f"quarterly series. {NOTE_SAMPLE} {NOTE_SOURCE}"
    ),
    "tab_106_ladder_periods": (
        "Information set M. Full: percentage reduction in the MSE of the first model relative to "
        "the second; negative: the second is more accurate. Period columns: contribution of the "
        "target months of each period, in percentage points of the second model's MSE; they add "
        "up to Full. Outside: reduction without January 2021 to December 2023 (for "
        "manufacturing, a descriptive split). $p$-values and stars as in "
        f"Table~\\ref{{tab:ladder-all-sets}}. {NOTE_SAMPLE} {NOTE_SOURCE}"
    ),
    "tab_107_top_predictors": (
        "Information set M. For each lasso: share of forecast dates on which it selects no "
        "predictor, and its three most frequent predictors with their selection shares. LASSO "
        "selects among the published series, FP-DFM and FP-PCA among idiosyncratic components; "
        "own past: the target's latest value (LASSO) or latest idiosyncratic component. Names "
        f"shortened; full names in Table~\\ref{{tab:series}}. {NOTE_SAMPLE} {NOTE_SOURCE}"
    ),
    "tab_108_external_routes": (
        "MSE relative to AR; lowest of the four per row and sample in bold. The first three "
        "columns are FP-DFM: in M or MQ; in ME or MQE; and the fixed-factor model, which keeps "
        "the factors and common forecast of M or MQ and adds the external series only to the "
        f"sparse step. E in LASSO: LASSO in ME or MQE. {NOTE_FULL}; {NOTE_WITHOUT}. " + NOTE_SOURCE
    ),
    "tab_109_fixed_factor_mse": (
        "MSE in squared percentage points. Columns as in Table~\\ref{tab:external-routes}; the "
        "candidate predictors and training months of the sparse step can differ between "
        f"columns. {NOTE_SAMPLE} {NOTE_SOURCE}"
    ),
}


def group_headers(table_id):
    """Centre the grouped column headers of a saved table and rule them off."""
    path = OUT / "tables" / f"{table_id}.tex"
    tex = path.read_text()
    header = tex.split("\\toprule\n", 1)[1].split("\\\\\n", 1)[0]
    rules, col = [], 0
    for cell in header.split("&"):
        cell = cell.strip()
        width = int(cell.split("{", 2)[1].split("}")[0]) if cell.startswith("\\multicolumn") else 1
        if cell.startswith("\\multicolumn") and not cell.endswith("{}"):
            rules.append(f"\\cmidrule(lr){{{col + 1}-{col + width}}}")
        col += width
    new = header.replace("}{r}{", "}{c}{").replace("}{l}{", "}{c}{")
    tex = tex.replace(header + "\\\\\n", new + "\\\\\n" + " ".join(rules) + "\n", 1)
    path.write_text(tex)


def years(months):
    return np.array([int(m[:4]) for m in months])


def primary_frame(forecasts, coverage, target, sample="primary"):
    """Errors and forecasts of every specification on a sample's dates, by target month."""
    dates = coverage[coverage.target.eq(target) & coverage["sample"].eq(sample)]
    dates = dates.common_dates.unique()
    assert len(dates) == 1
    dates = dates[0].split(";")
    d = forecasts[forecasts.target.eq(target) & forecasts.origin.isin(dates)]
    d = d.assign(spec=d.information_set + ":" + d.model)
    frame = d.pivot(index="target_month", columns="spec", values=["error", "forecast"])
    frame = frame.sort_index()
    assert frame.index.is_unique and len(frame) == len(dates)
    return frame


def test(frame, a, b, nested, rows):
    """CW (one-sided) or DM (two-sided) p-value of a against b on the selected rows."""
    ea, eb = frame["error"][a][rows], frame["error"][b][rows]
    d = eb**2 - ea**2
    if nested:
        d = d + (frame["forecast"][b][rows] - frame["forecast"][a][rows]) ** 2
    stat, _, _ = loss_test(d.to_numpy(), dates=list(frame.index[rows]))
    p = norm.sf(stat) if nested else 2 * norm.sf(abs(stat))
    return stat, p, (ea**2).sum() / (eb**2).sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(RUN / "forecasts.csv")
    coverage = pd.read_csv(RUN / "coverage.csv")
    scores = pd.read_csv(RUN / "scores.csv")
    scores = scores[scores["sample"].eq("primary")]
    tests = pd.read_csv(RUN / "statistical_tests.csv")
    tests = tests[tests["sample"].eq("primary")].set_index(["target", "augmented", "baseline"])
    selected = pd.read_csv(RUN / "selected_predictors.csv")
    frames = {t: primary_frame(forecasts, coverage, t) for t in NAMES}

    def rel(target, model):
        panel = "shared" if model == "AR" else "M"
        row = scores[
            scores.target.eq(target) & scores.information_set.eq(panel) & scores.model.eq(model)
        ]
        return row.relative_MSE_AR.item()

    def rel_set(target, model, info):
        row = scores[
            scores.target.eq(target) & scores.information_set.eq(info) & scores.model.eq(model)
        ]
        return row.relative_MSE_AR.item()

    old = _style.OUT
    _style.OUT = OUT
    num = _style.record_number
    try:
        # ---- The ladder in M: full sample and without 2021-2023, side by side ------------
        values = {(m, t): rel(t, m) for m in LABELS for t in NAMES}
        for (model, target), value in values.items():
            num(f"Lad{NAMES[target]}{MACRO[model]}Rel", f"{value:.3f}")
        outside_values = {}
        for target in NAMES:
            e = frames[target]["error"]
            y = years(e.index)
            out = (y < 2021) | (y > 2023)
            ar = (e["shared:AR"][out] ** 2).sum()
            for model in LABELS:
                col = "shared:AR" if model == "AR" else f"M:{model}"
                outside_values[(model, target)] = (e[col][out] ** 2).sum() / ar

        def cell(value, low):
            text = f"{value:.3f}"
            return f"\\textbf{{{text}}}" if np.isclose(value, low) else text

        samples = [("Full sample", values), ("Without 2021--2023", outside_values)]
        best = {
            (label, t): min(vals[(m, t)] for m in LABELS) for label, vals in samples for t in NAMES
        }
        rows = []
        for block, models in BLOCKS:
            for k, model in enumerate(models):
                row = {("", "Block"): block if k == 0 else "", ("", "Model"): LABELS[model]}
                for label, vals in samples:
                    for target, title in NAMES.items():
                        row[(label, SHORT.get(title, title))] = cell(
                            vals[(model, target)], best[(label, target)]
                        )
                rows.append(row)
        table = pd.DataFrame(rows)
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        _style.save_table(
            table,
            "tab_104_ladder_m",
            index=False,
            escape=False,
            column_format="@{}ll" + "r" * 8 + "@{}",
            rule_before=[b for b, _ in BLOCKS[1:]],
            notes=NOTES["tab_104_ladder_m"],
        )
        group_headers("tab_104_ladder_m")

        # ---- When each model gains on AR: cumulative SSE difference, set M --------------
        # Rising = the model beats AR in that month; a step marks an episode (Pietro's colors).
        styles = [
            ("DFM", _style.PALETTE_MANY[0], "-", 1.1),
            ("FarmPredict", _style.PALETTE_MANY[2], "--", 1.1),
            ("DFM_AR", _style.PALETTE_MANY[1], "-", 1.1),
            ("PCA_VAR_LASSO", _style.PALETTE_MANY[4], "-", 1.1),
            ("LASSO", _style.PALETTE_MANY[3], "-", 1.4),
        ]
        # 2x2 panels across both text columns, kept flat (about half a page with its caption)
        fig, axes = plt.subplots(2, 2, figsize=(_style.SIZES["full"][0], 3.3), sharex=True)
        for ax, (target, title) in zip(axes.flat, NAMES.items(), strict=True):
            e = frames[target]["error"]
            months = pd.PeriodIndex(e.index, freq="M").to_timestamp()
            calendar = pd.date_range(months.min(), months.max(), freq="MS")
            for model, color, style, width in styles:
                gain = (e["shared:AR"] ** 2 - e[f"M:{model}"] ** 2).cumsum()
                np.testing.assert_allclose(  # the end point is the full-sample SSE difference
                    gain.iloc[-1] / (e["shared:AR"] ** 2).sum(), 1 - rel(target, model), rtol=1e-9
                )
                curve = pd.Series(gain.to_numpy(), index=months).reindex(calendar)
                ax.plot(
                    curve.index,
                    curve,
                    color=color,
                    linestyle=style,
                    linewidth=width,
                    label=LABELS[model],
                )
            ax.axvspan(
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2024-01-01"),
                color="gray",
                alpha=0.13,
                linewidth=0,
            )
            ax.axhline(0, color="#555555", linewidth=0.6)
            ax.set_title(title)
            ax.xaxis.set_major_locator(YearLocator(1))
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))
            ax.tick_params(labelsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("SSE of AR minus model", fontsize=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(styles), loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.5, w_pad=1.5)
        _style.save_fig(
            fig,
            "fig_200_gains_over_time",
            formats=["pdf", "png"],
            used_in="Results 4.1, the 2021-2023 episode",
        )
        plt.close(fig)

        # ---- Every step in every set, with tests ------------------------------------------
        grid, cells = [], {}
        for target, title in NAMES.items():
            frame = frames[target]
            everything = np.ones(len(frame), dtype=bool)
            y = years(frame.index)
            outside = (y < 2021) | (y > 2023)
            for info in SETS:
                row = {"Target": title, "Set": info}
                for key, label, a, b, nested in STEPS:
                    if "PCA" in a + b and info not in PCA_SETS:
                        row[label] = "---"
                        continue
                    stat, p, ratio = test(frame, f"{info}:{a}", f"{info}:{b}", nested, everything)
                    saved = (
                        tests.loc[(target, f"{info}:{a}", f"{info}:{b}")]
                        if (target, f"{info}:{a}", f"{info}:{b}") in tests.index
                        else None
                    )
                    if saved is not None:  # same numbers as the run's own test file
                        np.testing.assert_allclose(saved.MSE_ratio, ratio, rtol=1e-9)
                        np.testing.assert_allclose(saved.statistic, stat, rtol=1e-7)
                    row[label] = f"{ratio:.3f}{_style.stars(p)} ({p:.2f})"
                    _, p_out, ratio_out = test(frame, f"{info}:{a}", f"{info}:{b}", nested, outside)
                    cells[(target, info, key)] = (ratio, p, ratio_out, p_out)
                grid.append(row)
        _style.save_table(
            pd.DataFrame(grid),
            "tab_105_ladder_all_sets",
            index=False,
            escape=False,
            notes=NOTES["tab_105_ladder_all_sets"],
        )

        # ---- Every step in M by period, and outside 2021-23 -------------------------------
        split = []
        for target, title in NAMES.items():
            frame = frames[target]
            y = years(frame.index)
            outside = (y < 2021) | (y > 2023)
            everything = np.ones(len(frame), dtype=bool)
            for key, label, a, b, nested in STEPS:
                ea, eb = frame["error"][f"M:{a}"], frame["error"][f"M:{b}"]
                _, p, ratio = test(frame, f"M:{a}", f"M:{b}", nested, everything)
                _, p_out, ratio_out = test(frame, f"M:{a}", f"M:{b}", nested, outside)
                gain, gain_out = 100 * (1 - ratio), 100 * (1 - ratio_out)
                row = {"Target": title, "Step": label}
                parts = []
                for pkey, plabel, first, last in PERIODS:
                    rows_p = (y >= first) & (y <= last)
                    part = 100 * ((eb[rows_p] ** 2).sum() - (ea[rows_p] ** 2).sum()) / (eb**2).sum()
                    parts.append(part)
                    row[plabel] = f"{part:.1f}"
                    num(f"Lad{title}{key}{pkey}", f"{part:.1f}")
                np.testing.assert_allclose(sum(parts), gain, atol=1e-8)
                row["Full ($p$)"] = f"{gain:.1f}{_style.stars(p)} ({p:.2f})"
                row["Outside ($p$)"] = f"{gain_out:.1f}{_style.stars(p_out)} ({p_out:.2f})"
                split.append(row)
                num(f"Lad{title}{key}Gain", f"{gain:.1f}")
                num(f"Lad{title}{key}GainAbs", f"{abs(gain):.1f}")
                num(f"Lad{title}{key}P", f"{p:.3f}")
                num(f"Lad{title}{key}Out", f"{gain_out:.1f}")
                num(f"Lad{title}{key}OutAbs", f"{abs(gain_out):.1f}")
                num(f"Lad{title}{key}OutP", f"{p_out:.3f}")
        _style.save_table(
            pd.DataFrame(split),
            "tab_106_ladder_periods",
            index=False,
            escape=False,
            rule_before=[f"{t} & FP-PCA / PCA-VAR" for t in list(NAMES.values())[1:]],
            notes=NOTES["tab_106_ladder_periods"],
        )

        # Largest MSE change from the DFM sparse step among the targets other than core
        others = [
            abs(100 * (1 - cells[(t, "M", "SparseDFM")][0])) for t in NAMES if t != "HICPNEF_EA"
        ]
        num("LadSparseDFMMaxAbsOther", f"{max(others):.1f}")

        # ---- Other sets: macros for the sparse step --------------------------------------
        for target, title in NAMES.items():
            for info in ["MQ", "ME", "MQE"]:
                for key in ["SparseDFM", "SparsePCA"]:
                    if (target, info, key) not in cells:
                        continue
                    ratio, p, ratio_out, p_out = cells[(target, info, key)]
                    num(f"Lad{title}{key}{info}Gain", f"{100 * (1 - ratio):.1f}")
                    num(f"Lad{title}{key}{info}P", f"{p:.3f}")
                    num(f"Lad{title}{key}{info}Out", f"{100 * (1 - ratio_out):.1f}")
                    num(f"Lad{title}{key}{info}OutP", f"{p_out:.3f}")

        # ---- What the lassos select ---------------------------------------------------------
        sel = []
        for target in NAMES:
            scored = coverage[coverage.target.eq(target) & coverage["sample"].eq("primary")]
            scored = set(scored.common_dates.iloc[0].split(";"))  # origins of the scored dates
            for info in SETS:
                for model in ["FarmPredict", "PCA_VAR_LASSO", "LASSO"]:
                    if model == "PCA_VAR_LASSO" and info not in PCA_SETS:
                        continue
                    d = selected[
                        selected.target.eq(target)
                        & selected.information_set.eq(info)
                        & selected.model.eq(model)
                        & selected.origin.isin(scored)
                    ]
                    sel.append(
                        {
                            "target": target,
                            "set": info,
                            "model": model,
                            "nonempty": d.origin.nunique() / len(scored),
                            "own": d[d.predictor.eq(target)].origin.nunique() / len(scored),
                        }
                    )
        sel = pd.DataFrame(sel).set_index(["target", "set", "model"])
        sel.to_csv(OUT / "selection.csv")
        fig, axes = plt.subplots(1, 2, figsize=_style.SIZES["full"], sharey=True)
        labels = [f"{NAMES[t]} {s}" for t in NAMES for s in SETS]
        yy = np.arange(len(labels))[::-1]
        marks = {
            "FarmPredict": ("FP-DFM", "o", _style.PALETTE[0]),
            "PCA_VAR_LASSO": ("FP-PCA", "s", _style.PALETTE[2]),
            "LASSO": ("LASSO", "^", _style.PALETTE[1]),
        }
        for model, (lab, marker, color) in marks.items():
            for col, ax in zip(["nonempty", "own"], axes, strict=True):
                vals = [sel[col].get((t, s, model), np.nan) for t in NAMES for s in SETS]
                ax.scatter(
                    100 * np.array(vals), yy, marker=marker, color=color, s=14, label=lab, zorder=3
                )
        for ax, title in zip(
            axes,
            [
                "(a) Dates with any predictor selected (%)",
                "(b) Dates with the target's own lag selected (%)",
            ],
            strict=True,
        ):
            ax.set_title(title)
            ax.set_xlim(-3, 103)
            ax.grid(axis="x", color="#E3E3E3", linewidth=0.5)
            for k in range(1, len(NAMES)):
                ax.axhline(len(labels) - 4 * k - 0.5, color="#BBBBBB", linewidth=0.5)
        axes[0].set_yticks(yy, labels)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, ncol=3, loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        _style.save_fig(
            fig, "fig_197_selection", formats=["pdf", "png"], used_in="Results 4.1, appendix"
        )
        plt.close(fig)
        for target, title in NAMES.items():
            num(
                f"LadEmpty{title}",
                f"{100 * (1 - sel.loc[(target, 'M', 'FarmPredict'), 'nonempty']):.1f}",
            )
            num(f"LadOwnLagFP{title}", f"{100 * sel.loc[(target, 'M', 'FarmPredict'), 'own']:.0f}")
        # ---- Main figure: the gain from each ingredient, by information set --------------
        panels = [
            ("SparseDFM", "(a) Sparse step on DFM factors", "FP-DFM vs DFM"),
            ("SparsePCA", "(b) Sparse step on PCA factors", "FP-PCA vs PCA-VAR"),
            ("OwnLag", "(c) Own lag on DFM factors", "DFM-AR vs DFM"),
        ]
        block_color = {
            "M": _style.PALETTE[0],
            "MQ": _style.PALETTE[0],
            "ME": _style.PALETTE[1],
            "MQE": _style.PALETTE[1],
        }
        gap = len(SETS) + 1.0
        ypos = {
            (t, info): -(i * gap + j) for i, t in enumerate(NAMES) for j, info in enumerate(SETS)
        }
        fig, axes = plt.subplots(
            1, 3, figsize=(_style.SIZES["full"][0], 3.7), sharex=True, sharey=True
        )
        for ax, (key, title, pair) in zip(axes, panels, strict=True):
            for (target, info), y0 in ypos.items():
                if (target, info, key) not in cells:
                    continue
                ratio, p, ratio_out, _ = cells[(target, info, key)]
                color = block_color[info]
                ax.plot(
                    [100 * (1 - ratio), 100 * (1 - ratio_out)],
                    [y0, y0],
                    color=color,
                    linewidth=0.6,
                    alpha=0.5,
                    zorder=2,
                )
                ax.scatter(
                    100 * (1 - ratio_out),
                    y0,
                    s=14,
                    marker="x",
                    color=color,
                    linewidths=0.8,
                    zorder=3,
                )
                ax.scatter(
                    100 * (1 - ratio),
                    y0,
                    s=22,
                    marker="o",
                    color=color,
                    zorder=4,
                    facecolors=color if p < 0.05 else "white",
                    linewidths=0.9,
                )
            ax.axvline(0, color="#555555", linewidth=0.6)
            ax.set_title(f"{title}\n{pair}", fontsize=7)
            ax.set_xlabel("MSE reduction (%); right is better")
            ax.grid(axis="x", color="#E3E3E3", linewidth=0.5)
            ax.grid(axis="y", visible=False)
            for i in range(1, len(NAMES)):
                ax.axhline(-(i * gap) + 1.0, color="#CCCCCC", linewidth=0.5)
        axes[0].set_xlim(-9, 16)
        axes[0].set_yticks(list(ypos.values()), [info for (_, info) in ypos])
        for i, title in enumerate(NAMES.values()):
            axes[0].annotate(
                title,
                xy=(-0.2, -(i * gap) - 1.5),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=7,
            )
        from matplotlib.lines import Line2D

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
                [],
                [],
                marker="o",
                linestyle="",
                color=_style.PALETTE[1],
                label="ME, MQE (with external series)",
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
        _style.save_fig(
            fig, "fig_198_ladder_steps", formats=["pdf", "png"], used_in="Results 4.1, main exhibit"
        )
        plt.close(fig)

        # ---- Energy inflation during the 2022-2023 reversal, set M -------------------------
        e = frames["HICPNG_EA"]
        window = [m for m in e.index if "2021-01" <= m <= "2023-12"]
        actual = e["forecast"]["shared:AR"] + e["error"]["shared:AR"]
        months = pd.PeriodIndex(window, freq="M").to_timestamp()
        fig, ax = plt.subplots(figsize=_style.SIZES["column"])
        # Only the lines the argument needs: the outcome, FP-DFM, LASSO, AR
        ax.plot(months, actual[window], color="#222222", linewidth=1.3, label="Actual")
        for model, label, color, width in [
            ("shared:AR", "AR", "#BBBBBB", 0.9),
            ("M:FarmPredict", "FP-DFM", _style.PALETTE_MANY[2], 1.2),
            ("M:LASSO", "LASSO", _style.PALETTE_MANY[3], 1.6),
        ]:
            ax.plot(
                months,
                e["forecast"][model][window],
                color=color,
                linewidth=width,
                label=label,
            )
        ax.axvspan(
            pd.Timestamp("2022-10-15"),
            pd.Timestamp("2023-03-15"),
            color="gray",
            alpha=0.15,
            linewidth=0,
        )
        ax.axhline(0, color="#555555", linewidth=0.5)
        ax.set_ylabel("Monthly change (%)")
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(MonthLocator(bymonth=[4, 7, 10]))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, fontsize=6.5, loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        _style.save_fig(
            fig,
            "fig_199_energy_reversal",
            formats=["pdf", "png"],
            used_in="Results 4.1, the 2021-2023 episode",
        )
        plt.close(fig)
        # The months that decide the energy ranking: prices fall while the DFM forecasts rises
        fall = [m for m in window if "2022-11" <= m <= "2023-03"]
        num("LadFallFPMin", f"{e['forecast']['M:FarmPredict'][fall].min():.1f}")
        num("LadFallFPMax", f"{e['forecast']['M:FarmPredict'][fall].max():.1f}")
        num("LadFallDFMMin", f"{e['forecast']['M:DFM'][fall].min():.1f}")
        num("LadFallDFMMax", f"{e['forecast']['M:DFM'][fall].max():.1f}")
        num("LadFallActualMinAbs", f"{-actual[fall].max():.1f}")
        num("LadFallActualMaxAbs", f"{-actual[fall].min():.1f}")
        num("LadFallLassoMaxAbs", f"{e['forecast']['M:LASSO'][fall].abs().max():.1f}")
        rows_f = e.index.isin(fall)
        gap = e["error"]["M:FarmPredict"] ** 2 - e["error"]["M:LASSO"] ** 2
        num("LadFallShareFP", f"{100 * gap[rows_f].sum() / gap.sum():.0f}")

        # ---- What each lasso selects, set M (body), and external picks in ME (macro) --------
        catalogue = pd.read_csv("data/metadata/ea_md_qd_series.csv").set_index("series_id")

        def scored_dates(target):
            dates = coverage[coverage.target.eq(target) & coverage["sample"].eq("primary")]
            return set(dates.common_dates.iloc[0].split(";"))

        def picks(target, model, info):
            sc = scored_dates(target)
            d = selected[
                selected.target.eq(target)
                & selected.model.eq(model)
                & selected.information_set.eq(info)
                & selected.origin.isin(sc)
            ]
            return d, sc

        lassos = [
            ("LASSO", "LASSO"),
            ("FP-DFM", "FarmPredict"),
            ("FP-PCA", "PCA_VAR_LASSO"),
        ]
        top_rows = []
        for target, title in NAMES.items():
            picked = {}
            for label, model in lassos:
                d, sc = picks(target, model, "M")
                share = (d.groupby("predictor").origin.nunique() / len(sc)).nlargest(3)
                names = [
                    "Own past"
                    if code == target
                    else SHORT_SERIES.get(
                        catalogue.description.get(code, code),
                        catalogue.description.get(code, code),
                    )
                    for code in share.index
                ]
                empty = 100 * (1 - sel.loc[(target, "M", model), "nonempty"])
                picked[label] = [("\\textit{No predictor}", f"{empty:.0f}")] + [
                    (n, f"{100 * v:.0f}") for n, v in zip(names, share.values, strict=True)
                ]
            for k in range(4):
                row = {("", "Target"): title if k == 0 else ""}
                for label, _ in lassos:
                    name, pct = picked[label][k] if k < len(picked[label]) else ("---", "")
                    row[(label, "Predictor")] = name
                    row[(label, "\\%")] = pct
                top_rows.append(row)
        table = pd.DataFrame(top_rows)
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        _style.save_table(
            table,
            "tab_107_top_predictors",
            escape=False,
            index=False,
            rule_before=list(NAMES.values())[1:],
            column_format="@{}l" + "lr" * 3 + "@{}",
            notes=NOTES["tab_107_top_predictors"],
        )
        group_headers("tab_107_top_predictors")
        # With the external series: how often each lasso picks an external series (ME)
        external = []
        for target, title in NAMES.items():
            for model, key in [("FarmPredict", "FP"), ("LASSO", "Lasso")]:
                d, sc = picks(target, model, "ME")
                share = d[~d.predictor.isin(catalogue.index)].origin.nunique() / len(sc)
                num(f"LadExtPick{key}{title}ME", f"{100 * share:.0f}")
                if model == "FarmPredict":
                    external.append(100 * share)
        num("LadExtPickFPMaxME", f"{max(external):.0f}")

        # ---- Robustness of the robust gains: other samples and the earlier outlier rule ----
        def sample_frame(run, target, sample):
            folder = Path("outputs/forecasts") / run
            f = pd.read_csv(folder / "forecasts.csv")
            c = pd.read_csv(folder / "coverage.csv")
            return primary_frame(f, c, target, sample)

        checks = [
            ("Core", "HICPNEF_EA", "SparseDFM", "FarmPredict", "DFM"),
            ("Core", "HICPNEF_EA", "OwnLag", "DFM_AR", "DFM"),
            ("Manufacturing", "IPMN_EA", "SparsePCA", "PCA_VAR_LASSO", "PCA_VAR"),
        ]
        for title, target, key, a, b in checks:
            variants = [
                ("Incl", sample_frame(RUN.name, target, "including_2020"), None),
                ("Narrow", frames[target], ("2022-02", "2023-12")),
                ("OldRule", sample_frame("paper_experiment", target, "primary"), None),
            ]
            for vkey, frame, window in variants:
                rows_v = np.ones(len(frame), dtype=bool)
                if window is not None:
                    rows_v = ~((frame.index >= window[0]) & (frame.index <= window[1]))
                _, p_v, ratio_v = test(frame, f"M:{a}", f"M:{b}", True, rows_v)
                num(f"LadRob{title}{key}{vkey}", f"{100 * (1 - ratio_v):.1f}")
                num(f"LadRob{title}{key}{vkey}P", f"{p_v:.3f}")

        # ---- Numbers for Subsection 4.2 (Pietro's red markers) ----------------------------
        # Does E replace the own lag in LASSO? Selection shares on scored dates.
        for target in ["HICPOV_EA", "HICPNG_EA"]:
            title = NAMES[target]
            for info in ["M", "ME"]:
                num(
                    f"LadOwnLagLasso{title}{info}",
                    f"{100 * sel.loc[(target, info, 'LASSO'), 'own']:.0f}",
                )
        scored = coverage[coverage.target.eq("HICPOV_EA") & coverage["sample"].eq("primary")]
        scored = set(scored.common_dates.iloc[0].split(";"))
        d = selected[
            selected.target.eq("HICPOV_EA")
            & selected.information_set.eq("ME")
            & selected.model.eq("LASSO")
            & selected.origin.isin(scored)
        ]
        share = d.groupby("predictor").origin.nunique() / len(scored)
        for series, macro in [
            ("RETA_006", "LadRetailExpShareME"),
            ("MCOILBRENTEU", "LadBrentShareME"),
            ("INDU_006", "LadIndustryExpShareME"),
        ]:
            num(macro, f"{100 * share.get(series, 0.0):.0f}")

        # Do Q and E add up? Largest interaction, in percentage points of the M MSE.
        def change(target, model, info):
            row = scores[
                scores.target.eq(target) & scores.information_set.eq(info) & scores.model.eq(model)
            ]
            return 100 * (
                row.MSE.item()
                / scores[
                    scores.target.eq(target)
                    & scores.information_set.eq("M")
                    & scores.model.eq(model)
                ].MSE.item()
                - 1
            )

        inter = [
            change(t, m, "MQE") - change(t, m, "MQ") - change(t, m, "ME")
            for t in NAMES
            for m in ["DFM", "DFM_AR", "FarmPredict", "LASSO"]
        ]
        num("LadInteractionMax", f"{np.max(np.abs(inter)):.1f}")

        # Two-sided DM p-value of each model in MQ or ME against itself in M (Results 4.2)
        for target, title in NAMES.items():
            for model in [
                "LASSO",
                "PCR",
                "PCA_VAR",
                "PCA_VAR_LASSO",
                "DFM",
                "DFM_AR",
                "FarmPredict",
            ]:
                for info in ["MQ", "ME"]:
                    if info not in PCA_SETS and model in {"PCR", "PCA_VAR", "PCA_VAR_LASSO"}:
                        continue
                    row = tests.loc[(target, f"{info}:{model}", f"M:{model}")]
                    assert row.test == "Diebold_Mariano"
                    num(f"LadDM{title}{MACRO[model]}{info}", f"{row.pvalue_two_sided:.2f}")

        # ---- Results 4.2: where the external series enter (replaces Table 7 in the body) ----
        # FP-DFM without E, with E in the factors, with E only in the sparse step (fixed-factor
        # model, factors of M or MQ kept), and LASSO with E; MSE relative to AR.
        fixed_factor = {
            "ME": pd.read_csv(FIXED_FACTOR["ME"] / "forecasts.csv"),
            "MQE": pd.read_csv(FIXED_FACTOR["MQE"] / "forecasts.csv"),
        }
        inflation = [t for t in NAMES if t != "IPMN_EA"]
        routes = [
            ("Without E", lambda e, fx, base, ext: e[f"{base}:FarmPredict"]),
            ("E in factors", lambda e, fx, base, ext: e[f"{ext}:FarmPredict"]),
            ("E in sparse step", lambda e, fx, base, ext: fx),
            ("E in LASSO", lambda e, fx, base, ext: e[f"{ext}:LASSO"]),
        ]
        rows, route_rel, level_rows = [], {}, []

        def level(errors):
            return f"{(errors**2).mean():.6f}"

        for base, ext, block in [("M", "ME", "M $\\to$ ME"), ("MQ", "MQE", "MQ $\\to$ MQE")]:
            for k, target in enumerate(inflation):
                e = frames[target]["error"]
                fx = fixed_factor[ext]
                fx = fx[fx.target.eq(target) & fx.model.eq(f"Fixed_DFM_{base}_external_sparse")]
                fx = fx.set_index("target_month").error
                assert fx.index.is_unique
                fx = fx.reindex(e.index)
                assert fx.notna().all()  # the fixed-factor model covers every scored month
                y = years(e.index)
                out = (y < 2021) | (y > 2023)
                row = {("", "Sets"): block if k == 0 else "", ("", "Target"): NAMES[target]}
                for label, rows_used in [("Full sample", slice(None)), ("Without 2021--2023", out)]:
                    ar = (e["shared:AR"][rows_used] ** 2).sum()
                    values = {
                        name: (pick(e, fx, base, ext)[rows_used] ** 2).sum() / ar
                        for name, pick in routes
                    }
                    low = min(values.values())
                    for name, value in values.items():
                        route_rel[(target, ext, label, name)] = value
                        text = f"{value:.3f}"
                        row[(label, name)] = (
                            f"\\textbf{{{text}}}" if np.isclose(value, low) else text
                        )
                rows.append(row)
                # Appendix Table A10: full-sample MSE levels (squared percentage points)
                level_rows.append(
                    {
                        ("", "Sets"): block if k == 0 else "",
                        ("", "Target"): NAMES[target],
                        ("DFM", "Without E"): level(e[f"{base}:DFM"]),
                        ("DFM", "E in factors"): level(e[f"{ext}:DFM"]),
                        ("FP-DFM", "E in factors"): level(e[f"{ext}:FarmPredict"]),
                        ("FP-DFM", "E in sparse step"): level(fx),
                        ("LASSO", "E in LASSO"): level(e[f"{ext}:LASSO"]),
                    }
                )
        # The full-sample cells of the saved run's models match its scores file
        assert np.isclose(
            route_rel[("HICPOV_EA", "ME", "Full sample", "E in LASSO")],
            rel_set("HICPOV_EA", "LASSO", "ME"),
        )
        table = pd.DataFrame(rows)
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        _style.save_table(
            table,
            "tab_108_external_routes",
            index=False,
            escape=False,
            column_format="@{}ll" + "r" * 8 + "@{}",
            rule_before=["MQ $\\to$ MQE"],
            used_in="Results 4.2, external series only in the sparse step",
            notes=NOTES["tab_108_external_routes"],
        )
        group_headers("tab_108_external_routes")
        mse_saved = scores.set_index(["target", "information_set", "model"]).MSE
        assert np.isclose(
            float(level_rows[0][("LASSO", "E in LASSO")]),
            mse_saved[("HICPOV_EA", "ME", "LASSO")],
            atol=1e-6,
        )
        table = pd.DataFrame(level_rows)
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        _style.save_table(
            table,
            "tab_109_fixed_factor_mse",
            index=False,
            escape=False,
            column_format="@{}ll" + "r" * 5 + "@{}",
            rule_before=["MQ $\\to$ MQE"],
            used_in="Appendix A.6, fixed-factor model",
            notes=NOTES["tab_109_fixed_factor_mse"],
        )
        group_headers("tab_109_fixed_factor_mse")

        # ---- Results 4.2: when the extra information helps, over time ----------------------
        # Cumulative SSE of a model in M minus that of the same model with the block; rising = the
        # block helps in those months (same layout and model colors as fig_200).
        lines = [
            ("DFM", "E", _style.PALETTE_MANY[0], "-", "DFM, with E"),
            ("LASSO", "E", _style.PALETTE_MANY[3], "-", "LASSO, with E"),
            ("DFM", "Q", _style.PALETTE_MANY[0], (0, (4, 2)), "DFM, with Q"),
            ("LASSO", "Q", _style.PALETTE_MANY[3], (0, (4, 2)), "LASSO, with Q"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(_style.SIZES["full"][0], 3.1), sharex=True)
        for ax, (target, title) in zip(axes.flat, NAMES.items(), strict=True):
            e = frames[target]["error"]
            months = pd.PeriodIndex(e.index, freq="M").to_timestamp()
            calendar = pd.date_range(months.min(), months.max(), freq="MS")
            for model, block, color, style, label in lines:
                if target == "IPMN_EA" and block == "E":
                    continue  # E is studied for the inflation targets only
                info = "MQ" if block == "Q" else "ME"
                gain = (e[f"M:{model}"] ** 2 - e[f"{info}:{model}"] ** 2).cumsum()
                np.testing.assert_allclose(  # end point = full-sample MSE change of Figure 7
                    gain.iloc[-1] / (e[f"M:{model}"] ** 2).sum(),
                    1 - rel_set(target, model, info) / rel_set(target, model, "M"),
                    rtol=1e-9,
                )
                curve = pd.Series(gain.to_numpy(), index=months).reindex(calendar)
                ax.plot(
                    curve.index, curve, color=color, linestyle=style, linewidth=1.5, label=label
                )
            ax.axvspan(
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2024-01-01"),
                color="gray",
                alpha=0.13,
                linewidth=0,
            )
            ax.axhline(0, color="#555555", linewidth=0.6)
            ax.set_title(title if target != "IPMN_EA" else f"{title} (quarterly series only)")
            ax.xaxis.set_major_locator(YearLocator(1))
            ax.xaxis.set_major_formatter(DateFormatter("%Y"))
            ax.tick_params(labelsize=6)
        for ax in axes[:, 0]:
            ax.set_ylabel("SSE in M minus\nSSE with the block", fontsize=7)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(lines), loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.5, w_pad=1.5)
        _style.save_fig(
            fig,
            "fig_201_information_over_time",
            formats=["pdf", "png"],
            used_in="Results 4.2, when the quarterly and external series help",
        )
        plt.close(fig)

        # Share of each change earned in 2021-2022: percentage points of the M MSE of the model
        def by_period(target, model, info):
            e = frames[target]["error"]
            d = e[f"M:{model}"] ** 2 - e[f"{info}:{model}"] ** 2
            y = years(e.index)
            base = (e[f"M:{model}"] ** 2).sum()
            return 100 * d.sum() / base, 100 * d[(y >= 2021) & (y <= 2022)].sum() / base

        for target, model, info, sign in [
            ("IPMN_EA", "DFM", "MQ", 1),
            ("HICPOV_EA", "LASSO", "ME", 1),
            ("HICPNG_EA", "LASSO", "ME", 1),
            ("HICPNEF_EA", "LASSO", "ME", 1),
            ("HICPNEF_EA", "DFM", "ME", 1),
            ("HICPOV_EA", "DFM", "ME", -1),  # losses, reported as positive numbers
            ("HICPNG_EA", "DFM", "ME", -1),
        ]:
            total, early = by_period(target, model, info)
            assert sign * total > 0
            kind = "Gain" if sign > 0 else "Loss"
            key = f"LadInfo{NAMES[target]}{MACRO[model]}{info}{kind}"
            num(f"{key}Total", f"{sign * total:.1f}")
            num(f"{key}TwentyOne", f"{sign * early:.1f}")

        # ---- Numbers from the logs: the BIC entry threshold and the DFM's mean errors -----
        tuning = pd.read_csv(RUN / "tuning.csv")
        t = tuning[tuning.model.eq("FarmPredict") & tuning.information_set.eq("M")]
        cut = 100 * (1 - np.exp(-(np.log(t.n_train) / t.n_train) * np.log(np.log(t.d))))
        num("LadBicCutMedian", f"{cut.median():.1f}")
        num("LadBicCutMin", f"{cut.min():.1f}")
        num("LadBicCutMax", f"{cut.max():.1f}")
        num("LadBicCandidates", f"{int(t.d.max())}")
        num("LadBicTrainMedian", f"{int(t.n_train.median())}")
        num(
            "LadBicCorr",
            f"{np.sqrt(1 - np.exp(-np.median(np.log(t.n_train) / t.n_train * np.log(np.log(t.d))))):.2f}",
        )
        num(
            "LadEmptyPCACore",
            f"{100 * (1 - sel.loc[('HICPNEF_EA', 'M', 'PCA_VAR_LASSO'), 'nonempty']):.1f}",
        )
        num(
            "LadEmptyPCAEnergy",
            f"{100 * (1 - sel.loc[('HICPNG_EA', 'M', 'PCA_VAR_LASSO'), 'nonempty']):.1f}",
        )
        for target, title in [("HICPNG_EA", "Energy"), ("IPMN_EA", "Manufacturing")]:
            err = frames[target]["error"]
            num(f"LadDFMOver{title}", f"{-err['M:DFM'].mean():.2f}")
            num(f"LadARMeanErr{title}", f"{err['shared:AR'].mean():.2f}")
    finally:
        _style.OUT = old
    _style.record_output(
        "results_ladder",
        [str(OUT / "numbers.tex"), str(OUT / "selection.csv")],
        "Results 4.1-4.2: ladder and information-set exhibits and macros",
    )


if __name__ == "__main__":
    main()
