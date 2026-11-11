"""Descriptive data exhibits for the data section (D-68).

tab_01_data (what EA-MD-QD contains and what the panel uses, by block), tab_02_target (the four
targets by period, appendix), tab_03_series (the 87 series used, appendix), fig_01_targets (the
four targets), and the numbers the text quotes.

Run from the repo root:  uv run python -m scripts.descriptives
Reads the paper run's own definitions (config/forecast_paper.yaml, the EA-MD-QD 09.2026 file,
data/external/factor_sparse/manifest.json) and, for the forecast dates and the outliers flagged,
outputs/forecasts/paper_experiment_nonzero_iqr/. Plots and summary statistics use the whole sample.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src._style import (
    OUT,
    PALETTE,
    PALETTE_MANY,
    SIZES,
    record_number,
    record_output,
    save_fig,
)
from src.data.ea_md_qd import load, release_lags
from src.data.external import load_external
from src.preprocessing.transform import transform

DICTIONARY = Path("data/metadata/data_dictionary.csv")
DESCRIPTIONS = Path("data/metadata/variable_dictionary.csv")
PAPER = Path("config/forecast_paper.yaml")
RUN = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
SOURCE_EA = "EA-MD-QD, release 09.2026, euro-area file \\citep{barigozzi2026}"
SOURCE_E = "the external providers listed in Table~\\ref{tab:series}"

TARGETS = {
    "HICPOV_EA": "Headline HICP",
    "HICPNG_EA": "Energy HICP",
    "HICPNEF_EA": "Core HICP (excl. energy and food)",
    "IPMN_EA": "Manufacturing production",
}
# External series by economic block (D-43, D-68); the order is the table's.
E_BLOCKS = {
    "Energy commodity prices": ["MCOILBRENTEU", "PNGASEUUSDM", "PCOALAUUSDM"],
    "Metal prices": ["PCOPPUSDM", "PALUMUSDM", "PNICKUSDM", "PIORECRUSDM"],
    "Food and agricultural prices": [
        "WB_FOOD",
        "PMAIZMTUSDM",
        "PSOYBUSDM",
        "PSOILUSDM",
        "PCOCOUSDM",
        "PCOTTINDUSDM",
        "PRUBBUSDM",
        "PRAWMINDEXM",
    ],
    "Risk and supply-chain indices": ["GPR", "EUEPUINDXM", "GSCPI"],
    "Price expectations (surveys)": ["INDU_006", "RETA_006", "CONS_006"],
    "HICP subindices": ["FOODPR", "FOODUN", "IGDSX"],
}
PROVIDERS = {
    "US Energy Information Administration via FRED": "EIA",
    "International Monetary Fund via FRED": "IMF",
    "World Bank": "World Bank",
    "Dario Caldara and Matteo Iacoviello": "Caldara--Iacoviello",
    "Scott Baker, Nicholas Bloom and Steven Davis via FRED": "Baker--Bloom--Davis",
    "Federal Reserve Bank of New York": "FRBNY",
    "European Commission, DG ECFIN": "European Commission",
    "European Central Bank": "ECB",
}
TR3 = {"100*dlog": 2, "level": 4}  # external transformations in the TR3 code convention
PERIODS = {
    "2000--14": ("2000-02", "2014-12"),
    "2015--19": ("2015-01", "2019-12"),
    "2020": ("2020-01", "2020-12"),
    "2021--23": ("2021-01", "2023-12"),
    "2024--25": ("2024-01", "2025-12"),
    "2015--25 excl. 2020": ("2015-01", "2025-12"),  # the scored months
}
EVALUATION = ("2015-01", "2025-12")
FIRST_SCORED_IP = "2014-12"
IP_CLIP = 6  # fig_01 panel (d): 2020 would set the scale (about -20 to +10)
# EA-MD-QD blocks in the order of tab_01, with example series (from data/metadata/data_dictionary.csv).
EXAMPLES = {
    "Industrial Production and Turnover": "IP and turnover by industry, car registrations",
    "Prices": "HICP headline, energy, core; PPI",
    "Confidence Indicators": "EC and OECD confidence and sentiment",
    "Labor Market Indicators": "Unemployment; employment, hours, wages",
    "Interest Rates": "3- and 6-month, long-term rates",
    "Financial Markets": "Exchange rates, stock prices",
    "Monetary Aggregates": "Currency, M1, M2",
    "National Accounts": "GDP, consumption, investment, trade",
    "Labor Costs": "Unit labor costs by sector",
    "Credit Aggregates": "Loans and debt securities by sector",
    "Energy commodity prices": "Brent oil, EU gas, coal",
    "Metal prices": "Copper, aluminum, nickel, iron ore",
    "Food and agricultural prices": "World Bank food index, grains, cocoa",
    "Risk and supply-chain indices": "Geopolitical risk, policy uncertainty, GSCPI",
    "Price expectations (surveys)": "Industry, retail and consumer surveys",
    "HICP subindices": "Processed and unprocessed food, NEIG",
}
RAGGED = ">{\\raggedright\\arraybackslash}"
# Short block names for the one-line rows of tab_03 (full names in tab_01).
SHORT_BLOCK = {
    "Industrial Production and Turnover": "IP and turnover",
    "Prices": "Prices",
    "Confidence Indicators": "Confidence",
    "Labor Market Indicators": "Labor market",
    "Interest Rates": "Interest rates",
    "Financial Markets": "Financial markets",
    "Monetary Aggregates": "Money",
    "National Accounts": "National accounts",
    "Labor Costs": "Labor costs",
    "Credit Aggregates": "Credit",
    "Energy commodity prices": "Energy commodities",
    "Metal prices": "Metals",
    "Food and agricultural prices": "Food and agriculture",
    "Risk and supply-chain indices": "Risk and supply chain",
    "Price expectations (surveys)": "Price expectations",
    "HICP subindices": "HICP subindices",
}
CODES = "2 = 100\\,$\\Delta\\log x$, 3 = 100\\,$\\Delta^2\\log x$, 4 = $x$, 5 = $\\Delta x$"


def catalogue():
    """One row per series of the 87-series panel MQE: set, block, TR3 code, delay, source."""
    data, info = load()
    quarterly = yaml.safe_load(PAPER.read_text())["quarterly_series"]
    meta = pd.read_csv(DICTIONARY).set_index("series_id").loc[data.columns]
    lags = release_lags(data, info)
    monthly = list(meta.index[meta["frequency"] == "M"])
    ea = meta.loc[monthly + quarterly]
    rows = pd.DataFrame(
        {
            "set": ["M"] * len(monthly) + ["Q"] * len(quarterly),
            "block": ea["group"],
            "tr3": ea["transformation_code"].astype(int),
            "delay": lags[ea.index].astype(int),
            "source": ea["source"],
        }
    )
    external, manifest = load_external()
    words = pd.read_csv(DESCRIPTIONS).set_index("code")
    block = {name: b for b, names in E_BLOCKS.items() for name in names}
    if sorted(block) != sorted(external.columns):
        raise ValueError("E_BLOCKS must list exactly the 24 external series")
    ext = pd.DataFrame(
        {
            "set": "E",
            "block": [block[n] for n in external.columns],
            "tr3": [TR3[manifest["series"][n]["transformation"]] for n in external.columns],
            "delay": [manifest["series"][n]["release_lag_months"] for n in external.columns],
            "source": [PROVIDERS[words.loc[n, "source"]] for n in external.columns],
        },
        index=external.columns,
    )
    rows = pd.concat([rows, ext])
    rows["description"] = words.loc[rows.index, "meaning"]
    order = {b: k for k, b in enumerate(dict.fromkeys(rows["block"]))}
    rows = rows.iloc[np.lexsort((rows["block"].map(order), rows["set"].map("MQE".index)))]
    return rows, data, info, meta


def table_data(rows, meta):
    """tab_01_data: what EA-MD-QD contains and what our panel takes from it, by block."""
    lines = []
    for block, example in EXAMPLES.items():
        ours = rows[rows["block"] == block]
        ea = None if block in E_BLOCKS else meta[meta["group"] == block]
        counts = ["--" if ea is None else str(int((ea["frequency"] == f).sum())) for f in "MQ"] + [
            str(int((ours["set"] == s).sum())) for s in "MQE"
        ]
        # TR3 and delay of the series in our panel, not of the whole EA-MD-QD block.
        if ours.empty:
            codes, delay = "--", "--"
        else:
            codes = ", ".join(str(c) for c in sorted(ours["tr3"].unique()))
            delay = _span(ours["delay"])
        lines.append(" & ".join([block, *counts, codes, delay, example]) + " \\\\")
    first_external = list(EXAMPLES).index(next(iter(E_BLOCKS)))
    lines.insert(first_external, "\\midrule")
    total = [str(int((meta["frequency"] == f).sum())) for f in "MQ"]
    total += [str(int((rows["set"] == s).sum())) for s in "MQE"]
    tex = "\n".join(
        [
            "\\begin{threeparttable}",
            "\\begin{tabular}{@{}lrrrrrccl@{}}",
            "\\toprule",
            " & \\multicolumn{2}{c}{EA-MD-QD} & \\multicolumn{3}{c}{Our panel} & & & \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
            "Block & M & Q & M & Q & E & TR3 & Delay & Examples \\\\",
            "\\midrule",
            *lines,
            "\\midrule",
            "Total & " + " & ".join(total) + " & & & \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            (
                "\\item \\textit{Notes:} Number of series. M: monthly; Q: quarterly; E: external "
                f"monthly. TR3: transformation code of the series we use, {CODES}. Delay: "
                "publication delay in months, assumed for the external series. "
                f"\\textit{{Source:}} {SOURCE_EA}; {SOURCE_E}."
            ),
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "",
        ]
    )
    path = OUT / "tables" / "tab_01_data.tex"
    path.write_text(tex)
    record_output("tab_01_data", [str(path)], "paper: data section")


def _span(values):
    lo, hi = int(values.min()), int(values.max())
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def table_series(rows):
    """tab_03_series: every series of MQE, as a longtable for the one-column appendix."""
    table = pd.DataFrame(
        {
            "Code": rows.index,
            "Description": rows["description"].values,
            "Set": rows["set"].values,
            "Block": rows["block"].map(SHORT_BLOCK).values,
            "TR3": rows["tr3"].values,
            "Delay": rows["delay"].values,
            "Source": rows["source"].values,
        }
    )
    tex = table.to_latex(
        index=False,
        escape=True,
        longtable=True,
        column_format=f"@{{}}l{RAGGED}p{{7.4cm}}clrr{RAGGED}p{{2.6cm}}@{{}}",
        caption="Series in the panel, by information set",
        label="tab:series",
    )
    notes = (
        "\\par\\footnotesize\\textit{Notes:} M: monthly; Q: quarterly; E: external monthly. "
        f"Block, TR3 and Delay as in Table~\\ref{{tab:data}}. \\textit{{Source:}} {SOURCE_EA}; "
        "EIA, IMF and the European policy uncertainty index \\citep{baker2016} via FRED; World "
        "Bank; \\citet{caldara2022}; \\citet{benigno2022}; European Commission surveys; ECB.\\par\n"
    )
    path = OUT / "tables" / "tab_03_series.tex"
    path.write_text(tex + notes)
    record_output("tab_03_series", [str(path)], "paper: appendix, series list")


def table_target(panel):
    """tab_02_target: the four targets by period; one panel per statistic, targets in columns."""
    rows = {}
    for name, label in TARGETS.items():
        s = panel[name]
        row = {}
        for period, (start, end) in PERIODS.items():
            if period.endswith("excl. 2020") and name == "IPMN_EA":
                start = FIRST_SCORED_IP  # its longer delay makes December 2014 its first target
            x = s.loc[start:end].copy()
            if period.endswith("excl. 2020"):
                x.loc["2020-01":"2020-12"] = np.nan
            row[(period, "Mean")] = x.mean()
            row[(period, "SD")] = x.std()
            # Twelve months are too few for an autocorrelation worth reporting.
            row[(period, "AR(1)")] = np.nan if period == "2020" else x.autocorr(1)
        rows[label] = row
    table = pd.DataFrame(rows).T
    table.columns = pd.MultiIndex.from_tuples(table.columns)

    def cell(v):
        return "---" if pd.isna(v) else f"{v:.2f}".replace("-", "$-$")

    short = {
        "HICPOV_EA": "Headline",
        "HICPNG_EA": "Energy",
        "HICPNEF_EA": "Core",
        "IPMN_EA": "Manuf.",
    }
    labels = {p: p for p in PERIODS} | {"2015--25 excl. 2020": "Scored months"}
    lines = []
    for k, (stat, title) in enumerate(
        [("Mean", "Mean"), ("SD", "Standard deviation"), ("AR(1)", "First-order autocorrelation")]
    ):
        if k:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for period in PERIODS:
            if period.endswith("excl. 2020"):
                lines.append("\\cmidrule(l){2-5}")
            values = [cell(table.loc[TARGETS[t], (period, stat)]) for t in TARGETS]
            lines.append(f"\\quad {labels[period]} & " + " & ".join(values) + " \\\\")
    tex = "\n".join(
        [
            "\\begin{threeparttable}",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            " & " + " & ".join(short.values()) + " \\\\",
            "\\midrule",
            *lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}[flushleft]\\footnotesize",
            (
                "\\item \\textit{Notes:} Month-on-month change in percent, by period of the "
                "target month. Scored months: 2015--2025 without 2020, from December 2014 for "
                "manufacturing. \\textit{Source:} " + SOURCE_EA + "."
            ),
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "",
        ]
    )
    path = OUT / "tables" / "tab_02_target.tex"
    path.write_text(tex)
    record_output("tab_02_target", [str(path)], "paper: data section, targets")
    return table


SIGNAL_WINDOW = ("2000-02", "2014-12")  # before the first scored month: no test data


def aligned_correlations(rows, data, info, start):
    """Correlation of each target with every other series, timed as at the forecast date.

    At date t a target with delay d_i is forecast for month s = t - d_i + 1, and series j has
    published up to month t - d_j, so series j enters at month s + d_i - 1 - d_j.
    """
    monthly = list(rows.index[rows["set"] == "M"])
    external, _ = load_external()
    panel = transform(data[monthly], info).loc[str(start) :]
    series = pd.concat([panel, external[list(rows.index[rows["set"] == "E"])]], axis=1)
    series = series.loc[panel.index]
    out = []
    for target in TARGETS:
        y = panel[target]
        for name in series.columns.drop(target):
            shift = int(rows.loc[target, "delay"] - 1 - rows.loc[name, "delay"])
            pair = pd.concat([y, series[name].shift(-shift)], axis=1).loc[slice(*SIGNAL_WINDOW)]
            pair = pair.dropna()
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            out.append((target, name, rows.loc[name, "set"], rows.loc[name, "block"], r, len(pair)))
    return pd.DataFrame(out, columns=["target", "series", "set", "block", "r", "n"])


def external_signal_numbers(corr):
    """Strongest external and monthly euro-area correlation per target, quoted in Section 2.1."""
    keys = {
        "HICPOV_EA": "Headline",
        "HICPNG_EA": "Energy",
        "HICPNEF_EA": "Core",
        "IPMN_EA": "Manuf",
    }
    for target, key in keys.items():
        c = corr[corr.target == target].assign(size=lambda d: d.r.abs())
        record_number(f"ExtCorr{key}", f"{c[c.set == 'E']['size'].max():.2f}")
        record_number(f"MCorr{key}", f"{c[c.set == 'M']['size'].max():.2f}")


TIMING_ORIGIN = "2022-09"  # example forecast date for fig_03 (saved, not in the paper)
SET_NAMES = {"M": "Monthly", "E": "External", "Q": "Quarterly"}


def figure_release_timing(rows):
    """fig_03_release_timing: what is published at the end of one month, by delay."""
    origin = pd.Period(TIMING_ORIGIN, "M")
    months = pd.period_range(origin - 6, origin, freq="M")
    colors = {"M": PALETTE[0], "E": PALETTE[1], "Q": PALETTE[2]}
    groups = rows.groupby(["set", "delay"], sort=False)
    order = sorted(groups.groups, key=lambda k: ("MEQ".index(k[0]), k[1]))
    targets = {
        rows.loc["HICPOV_EA", "delay"]: "the three HICP targets",
        rows.loc["IPMN_EA", "delay"]: "manufacturing production",
    }
    fig, ax = plt.subplots(figsize=(SIZES["full"][0], 2.9))

    def cell(x0, width, y, filled, color, text=None, outline=False):
        ax.add_patch(
            plt.Rectangle(
                (x0 + 0.06, y - 0.38),
                width - 0.12,
                0.76,
                linewidth=1.3 if outline else 0.6,
                facecolor=color if filled else "white",
                edgecolor="#222222" if outline else (color if filled else "#BBBBBB"),
            )
        )
        if text:
            ax.text(
                x0 + width / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if filled else "#222222",
            )

    for i, (kind, delay) in enumerate(order):
        members = groups.get_group((kind, delay))
        if kind == "Q":
            quarters = pd.period_range(months[0].asfreq("Q"), months[-1].asfreq("Q"), freq="Q")
            for q in quarters:
                first = max(0, (q.asfreq("M", "start") - months[0]).n)
                last = min(len(months) - 1, (q.asfreq("M", "end") - months[0]).n)
                published = q.asfreq("M", "end") + delay <= origin
                cell(first, last - first + 1, i, published, colors[kind], f"Q{q.quarter}")
            label = f"{SET_NAMES[kind]}, {delay} months after the quarter"
        else:
            latest = origin - delay
            for j, month in enumerate(months):
                if kind == "M" and month == latest + 1 and delay in targets:
                    cell(j, 1, i, False, colors[kind], "target", outline=True)
                else:
                    cell(j, 1, i, month <= latest, colors[kind])
            lag = "no delay" if delay == 0 else f"{delay} month" + "s" * (delay > 1)
            label = f"{SET_NAMES[kind]}, {lag}"
        blocks = members.groupby("block").size().sort_values(ascending=False)
        content = ", ".join(f"{SHORT_BLOCK[b]} ({k})" for b, k in blocks.items())
        if kind == "M" and delay in targets:
            content += f"; target: {targets[delay]}"
        ax.text(
            len(months) + 0.2,
            i,
            textwrap.fill(content, 74),
            va="center",
            fontsize=6.5,
            color="#333333",
            linespacing=1.1,
        )
        ax.text(-0.2, i, label, va="center", ha="right", fontsize=7, color="#222222")
    ax.axvline(len(months), color="#222222", lw=1.0)
    ax.text(
        len(months) - 0.08,
        len(order) - 0.45,
        f"forecast date: end of {origin.strftime('%B %Y')}",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#222222",
    )
    ax.set_xlim(0, len(months))
    ax.set_ylim(len(order) + 0.3, -0.6)
    ax.set_xticks(np.arange(len(months)) + 0.5, [m.strftime("%b") for m in months])
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.55, top=0.91, bottom=0.03)
    save_fig(
        fig,
        "fig_03_release_timing",
        used_in="saved, not in the paper: publication delays at one date",
    )
    plt.close(fig)


def figure_targets(panel):
    """fig_01_targets: the four targets, scored months shaded, 2020 marked."""
    series, labels = panel, TARGETS
    fig, axes = plt.subplots(2, 2, figsize=(SIZES["full"][0], 3.6), sharex=True)
    ev = [pd.Period(p, "M").to_timestamp() for p in EVALUATION]
    covid = [pd.Period(p, "M").to_timestamp() for p in ("2020-01", "2020-12")]
    for k, (ax, (name, label)) in enumerate(zip(axes.flat, labels.items(), strict=True)):
        s = series[name].dropna()
        ax.axvspan(*ev, color="#DDDDDD", alpha=0.6, lw=0)
        ax.axvspan(*covid, color=PALETTE_MANY[1], alpha=0.25, lw=0)
        ax.axhline(0, color="black", lw=0.5)
        ax.plot(s.index.to_timestamp(), s.values, color=PALETTE_MANY[0], lw=0.7)
        if name == "IPMN_EA":
            ax.set_ylim(-IP_CLIP, IP_CLIP)
            ax.annotate(
                f"2020 off scale: \u2212{-s.min():.1f} and +{s.max():.1f}",
                xy=(0.02, 0.9),
                xycoords="axes fraction",
                fontsize=7,
            )
        ax.set_title(f"({'abcd'[k]}) {label}")
    for ax in axes[:, 0]:
        ax.set_ylabel("Percent, month on month")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#DDDDDD", alpha=0.6, label="Scored target months"),
        plt.Rectangle(
            (0, 0), 1, 1, color=PALETTE_MANY[1], alpha=0.25, label="2020, excluded from main scores"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    save_fig(fig, "fig_01_targets", used_in="paper: data section")
    plt.close(fig)


def origins_first_year():
    """Calendar year of the first forecast date of the paper run."""
    origins = pd.read_csv(RUN / "forecasts.csv", usecols=["origin"])["origin"]
    return pd.Period(origins.min(), "M").year


def numbers(rows, meta, panel, target):
    """Counts, dates and target statistics quoted in the data section."""
    m, q, e = (rows[rows["set"] == s] for s in "MQE")
    record_number("NSeriesEA", str(len(meta)))
    record_number("NSeriesMonthly", str(len(m)))
    record_number("NSeriesQuarterlyAll", str(int((meta["frequency"] == "Q").sum())))
    record_number("NQuarterlyUnused", str(int((meta["frequency"] == "Q").sum()) - len(q)))
    for k, word in enumerate(["Zero", "One", "Two", "Three"]):
        record_number(f"NDelay{word}", str(int((m["delay"] == k).sum())))
        record_number(f"NExtDelay{word}", str(int((e["delay"] == k).sum())))
    core = ["Industrial Production and Turnover", "Prices", "Confidence Indicators"]
    record_number("NMonthlyCore", str(int(m["block"].isin(core).sum())))
    record_number("FirstWindowYears", str(origins_first_year() - 2000))
    record_number("NExternalLogDiff", str(int((e["tr3"] == 2).sum())))
    record_number("NExternalLevel", str(int((e["tr3"] == 4).sum())))
    record_number(
        "NAuthorAdjusted",
        str(int(meta.loc[m.index.union(q.index), "seasonal_adjustment"].eq("NSA").sum())),
    )

    forecasts = pd.read_csv(RUN / "forecasts.csv", usecols=["origin"])
    origins = pd.PeriodIndex(forecasts["origin"].unique(), freq="M").sort_values()
    record_number("PaperOrigins", str(len(origins)))
    record_number("FirstOrigin", origins[0].strftime("%B %Y"))
    record_number("LastOrigin", origins[-1].strftime("%B %Y"))
    scores = pd.read_csv(RUN / "scores.csv")
    primary = scores[scores["sample"] == "primary"].groupby("target")["n"].max()
    record_number("NScoredHicp", str(int(primary["HICPNG_EA"])))
    record_number("NScoredIp", str(int(primary["IPMN_EA"])))
    start = panel.dropna(how="all").index[0]
    record_number("SampleStart", start.strftime("%B %Y"))
    record_number("SampleEnd", panel["HICPOV_EA"].last_valid_index().strftime("%B %Y"))
    through = pd.Period(yaml.safe_load(PAPER.read_text())["score_through"], "M")
    record_number("ScoreThrough", through.strftime("%B %Y"))

    flagged = pd.read_csv(RUN / "data_availability.csv")
    last = flagged[(flagged["origin"] == str(origins[-1])) & (flagged["information_set"] == "MQE")]
    last = last[last["outliers"] > 0]
    record_number("NOutliersLast", str(int(last["outliers"].sum())))
    per_date = flagged[flagged["information_set"] == "MQE"].groupby("origin")["outliers"].sum()
    record_number("NOutliersMedian", f"{per_date.median():.0f}")
    record_number("NOutliersMax", str(int(per_date.max())))
    first = flagged[(flagged["origin"] == str(origins[0])) & (flagged["information_set"] == "MQE")]
    first = first.set_index("series")
    record_number("NOutliersIronFirst", str(int(first.loc["PIORECRUSDM", "outliers"])))
    record_number("NOutliersGasFirst", str(int(first.loc["PNGASEUUSDM", "outliers"])))
    # Sample size: monthly and quarterly observations published at the first and last date.
    for label, origin in (("First", origins[0]), ("Last", origins[-1])):
        x = flagged[(flagged["origin"] == str(origin)) & (flagged["information_set"] == "MQE")]
        quarterly = x["series"].isin(q.index)
        record_number(f"T{label}", str(int(x.loc[~quarterly, "published"].max())))
        record_number(f"Tq{label}", str(int(x.loc[quarterly, "published"].max())))
    t_first = int(flagged.loc[flagged["origin"] == str(origins[0]), "published"].max())
    record_number("RatioFirstMQE", f"{len(rows) / t_first:.2f}")
    tuning = pd.read_csv(RUN / "tuning.csv")
    rows_sparse = tuning[tuning["model"] == "FarmPredict"].pivot_table(
        index="origin", columns="information_set", values="n_train", aggfunc="max"
    )
    record_number("SparseRowsMinME", str(int(rows_sparse["ME"].min())))
    record_number(
        "SparseRowsMinMEDate", pd.Period(rows_sparse["ME"].idxmin(), "M").strftime("%B %Y")
    )
    record_number("SparseRowsMSameDate", str(int(rows_sparse.loc[rows_sparse["ME"].idxmin(), "M"])))
    record_number(
        "NOutliersCurrency", str(int(last.loc[last.series == "CURR_EACC", "outliers"].sum()))
    )

    external, _ = load_external()
    before = external.loc["2000-02":"2009-12"]
    record_number("IronZeroShare", f"{100 * (before['PIORECRUSDM'].abs() < 1e-9).mean():.0f}")
    record_number("GasZeroShare", f"{100 * (before['PNGASEUUSDM'].abs() < 1e-9).mean():.0f}")

    energy = panel["HICPNG_EA"]
    record_number("EnergySdTrain", f"{target.loc[TARGETS['HICPNG_EA'], ('2000--14', 'SD')]:.2f}")
    record_number("EnergySdCrisis", f"{target.loc[TARGETS['HICPNG_EA'], ('2021--23', 'SD')]:.2f}")
    record_number("EnergyPeakValue", f"{energy.max():.1f}")
    record_number("EnergyPeakMonth", energy.idxmax().strftime("%B %Y"))
    record_number(
        "CoreSdEval", f"{target.loc[TARGETS['HICPNEF_EA'], ('2015--25 excl. 2020', 'SD')]:.2f}"
    )
    record_number(
        "HeadlineSdEval", f"{target.loc[TARGETS['HICPOV_EA'], ('2015--25 excl. 2020', 'SD')]:.2f}"
    )
    record_number(
        "EnergySdEval", f"{target.loc[TARGETS['HICPNG_EA'], ('2015--25 excl. 2020', 'SD')]:.2f}"
    )
    record_number(
        "CoreArEval", f"{target.loc[TARGETS['HICPNEF_EA'], ('2015--25 excl. 2020', 'AR(1)')]:.2f}"
    )
    record_number(
        "EnergyArEval", f"{target.loc[TARGETS['HICPNG_EA'], ('2015--25 excl. 2020', 'AR(1)')]:.2f}"
    )
    ip = panel["IPMN_EA"]
    record_number("IpFallValue", f"{-ip.min():.1f}")
    record_number("IpMinMonth", ip.idxmin().strftime("%B %Y"))
    record_number("IpMaxValue", f"{ip.max():.1f}")
    record_number("IpMaxMonth", ip.idxmax().strftime("%B %Y"))
    record_number(
        "IpSdEval", f"{target.loc[TARGETS['IPMN_EA'], ('2015--25 excl. 2020', 'SD')]:.2f}"
    )


def main():
    rows, data, info, meta = catalogue()
    start = yaml.safe_load(Path("config/research_design.yaml").read_text())["research"]["forecast"][
        "start_date"
    ]
    panel = transform(data[list(TARGETS)], info).loc[str(start) :]

    table_data(rows, meta)
    table_series(rows)
    target = table_target(panel)
    figure_targets(panel)
    figure_release_timing(rows)
    external_signal_numbers(aligned_correlations(rows, data, info, start))
    numbers(rows, meta, panel, target)
    print(target.round(2).to_string())


if __name__ == "__main__":
    main()
