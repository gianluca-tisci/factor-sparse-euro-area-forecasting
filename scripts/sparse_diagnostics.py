"""Sparse-step settings and residual-dependence diagnostics quoted in Subsection 3.3 and Appendix A.

Run from the repo root:  uv run python -m scripts.sparse_diagnostics
EXPLORATORY diagnostic: no model is re-estimated. At each date the saved paper-run parameters
(outputs/forecasts/paper_experiment_nonzero_iqr/<set>/<date>/parameters.npz) are passed once through the
Kalman smoother, exactly as fixed_external.py does, and the monthly residuals are compared with
the cleaned series. The FMM test is pre-specified for M at the first forecast date only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import solve_discrete_lyapunov

from src._style import PALETTE_MANY, SIZES, record_number, save_fig, save_table
from src.factors.residual_diagnostics import (
    correlation_summary,
    fmm_diagonal_test,
    standardized,
)
from src.forecasting.design import complete_design, dictionary, training_sample
from src.forecasting.dfm import Fit, MixedFrequencyEM, Parameters
from src.forecasting.experiment import clean_window, prepare

SOURCE = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
PROFILE = "config/forecast_paper.yaml"
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def residual_panel(information_set, origin, prepared):
    """Cleaned monthly series and their residuals from the saved fit at one date."""
    _, settings, _, panel, info, lags, _ = prepared
    monthly = list(info.index[info.Frequency == "M"])
    _, cleaned, z, mean, scale, _ = clean_window(panel, lags, pd.Period(origin, "M"))
    with np.load(SOURCE / information_set / origin / "parameters.npz") as saved:
        np.testing.assert_allclose(scale.to_numpy(), saved["scale"], rtol=1e-10)
        params = Parameters(
            *(saved[k].copy() for k in ["loadings", "variances", "ar", "innovation"])
        )
    model = MixedFrequencyEM(
        z, len(monthly), params.loadings.shape[1], settings["initial_factor_variance"]
    )
    fit = Fit(model, params, model.expectation(params), 0, True, [])
    residual = cleaned - (fit.common() * scale + mean)
    return cleaned[monthly], residual[monthly]


def record_groups(residual, components=3):
    """EA-MD-QD group with the largest squared-loading share in each leading residual component."""
    groups = pd.read_csv("data/metadata/data_dictionary.csv", index_col="series_id")["group"]
    u = standardized(residual)
    _, vectors = np.linalg.eigh(u.T @ u / len(u))
    leading = []
    for k in range(components):
        weight = pd.Series(vectors[:, -1 - k] ** 2, index=residual.columns)
        group = weight.groupby(groups.reindex(residual.columns).to_numpy()).sum().idxmax()
        if group.lower() not in leading:
            leading.append(group.lower())
    record_number("ResidGroups", ", ".join(leading[:-1]) + " and " + leading[-1])


TARGETS = {
    "HICPOV_EA": "Headline HICP",
    "HICPNG_EA": "Energy HICP",
    "HICPNEF_EA": "Core HICP",
    "IPMN_EA": "Manufacturing production",
}
# Tol BuRd: blue = negative correlation, red = positive.
CORRELATION = LinearSegmentedColormap.from_list(
    "tol_burd",
    ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D"]
    + ["#B2182B"],
)
PANDEMIC_WINDOWS = {"2020-04", "2020-05", "2020-06"}
SHORT_BLOCK = {
    "Industrial Production and Turnover": "IP and turnover",
    "Prices": "Prices",
    "Confidence Indicators": "Confidence",
    "Labor Market Indicators": "Labor market",
    "Interest Rates": "Interest rates",
    "Financial Markets": "Financial",
    "Monetary Aggregates": "Money",
}


def common_shares():
    """Share of each target's variance that the saved factor model assigns to the factors.

    Model-implied, on the standardized window data: lambda' Sigma_F lambda over that plus the
    idiosyncratic variance, with Sigma_F the unconditional covariance of the factor VAR(2).
    """
    rows = []
    for information_set in ["M"]:
        info = pd.read_csv(SOURCE / information_set / "selected_series.csv", index_col=0)
        names, monthly = list(info.index), (info.Frequency == "M").to_numpy()
        origins = yaml.safe_load((SOURCE / information_set / "metadata.json").read_text())[
            "origins"
        ]
        for origin in origins:
            with np.load(SOURCE / information_set / origin / "parameters.npz") as saved:
                loadings, variances = saved["loadings"], saved["variances"]
                ar, innovation = saved["ar"], saved["innovation"]
            r = loadings.shape[1]
            companion = np.zeros((2 * r, 2 * r))
            companion[:r], companion[r:, :r] = ar, np.eye(r)
            shocks = np.zeros((2 * r, 2 * r))
            shocks[:r, :r] = innovation
            sigma = solve_discrete_lyapunov(companion, shocks)[:r, :r]
            total = np.einsum("ij,jk,ik->i", loadings, sigma, loadings) + variances
            # Check of scale: the standardized monthly series have a total model variance close
            # to 1. Not in the windows ending in April-June 2020, where the factor VAR absorbs
            # the (unscreened) pandemic months and its implied variance exceeds the data's.
            if origin not in PANDEMIC_WINDOWS:
                assert abs(np.median(total[monthly]) - 1) < 0.05
            for target in TARGETS:
                i = names.index(target)
                common = loadings[i] @ sigma @ loadings[i]
                rows.append((information_set, pd.Period(origin, "M"), target, common / total[i]))
    return pd.DataFrame(rows, columns=["set", "origin", "target", "share"])


def core_share_numbers(shares):
    """Median share of each target's variance explained by the factors, in M (core by period)."""
    core = shares[shares.target == "HICPNEF_EA"]
    for name, (start, end) in {
        "Early": ("2015-01", "2019-12"),
        "Late": ("2021-01", "2025-12"),
    }.items():
        window = core[(core.origin >= pd.Period(start, "M")) & (core.origin <= pd.Period(end, "M"))]
        record_number(f"CommonShareCore{name}", f"{100 * window.share.median():.0f}")
    others = shares[shares.target != "HICPNEF_EA"].groupby("target").share.median()
    record_number("CommonShareOthersMin", f"{100 * others.min():.0f}")


def lasso_correlations(origin, prepared):
    """|corr| of each target's training response with each FarmPredict candidate, in M.

    The same inputs as the forecasting run at this date: residuals from the saved fit, the
    candidates of the latest published months (design.dictionary), the training rows whose
    label is published by the date, and complete rows only (design.complete_design).
    """
    _, settings, _, panel, info, lags, _ = prepared
    published, cleaned, z, mean, scale, _ = clean_window(panel, lags, pd.Period(origin, "M"))
    with np.load(SOURCE / "M" / origin / "parameters.npz") as saved:
        params = Parameters(
            *(saved[k].copy() for k in ["loadings", "variances", "ar", "innovation"])
        )
    model = MixedFrequencyEM(
        z, z.shape[1], params.loadings.shape[1], settings["initial_factor_variance"]
    )
    common = Fit(model, params, model.expectation(params), 0, True, []).common() * scale + mean
    policy = settings.get("training_outlier_policy", "published_labels_and_own_lags")
    labels = (cleaned if policy == "cleaned_inputs_and_labels" else published) - common
    features, _ = dictionary(cleaned - common, info.Frequency, lags, z.index)
    out = {}
    for target in TARGETS:
        x, y = training_sample(features, labels[target], lags[target], pd.Period(origin, "M"))
        x, y, _ = complete_design(x, y, features.loc[pd.Period(origin, "M")])
        r = np.abs([np.corrcoef(x[c], y)[0, 1] for c in x.columns])
        out[target] = (np.sort(r)[::-1], len(y))
    return out


def correlation_figure(raw, residual, profiles):
    """fig_83_correlation: (a) correlations before and after the factors; (b) lasso inputs."""
    groups = pd.read_csv("data/metadata/data_dictionary.csv", index_col="series_id")["group"]
    order = sorted(raw.columns, key=lambda c: (list(SHORT_BLOCK).index(groups[c]), c))
    blocks = [SHORT_BLOCK[groups[c]] for c in order]
    edges = [k for k in range(1, len(order)) if blocks[k] != blocks[k - 1]]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(SIZES["full"][0], 3.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    # (a) One matrix: series below the diagonal, idiosyncratic components above it.
    matrices = []
    for frame in (raw, residual):
        c = standardized(frame[order])
        matrices.append(c.T @ c / len(c))
    noise = 2 / np.sqrt(len(standardized(raw)))
    n = len(order)
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    merged = np.where(lower, matrices[0], matrices[1])
    merged[np.abs(merged) < noise] = np.nan  # indistinguishable from zero: white
    np.fill_diagonal(merged, np.nan)
    cmap = CORRELATION.copy()
    cmap.set_bad("white")
    image = left.imshow(merged, cmap=cmap, vmin=-1, vmax=1, interpolation="none")
    left.plot([-0.5, n - 0.5], [-0.5, n - 0.5], color="#333333", lw=0.6)
    for e in edges:
        left.axhline(e - 0.5, color="#BBBBBB", lw=0.4)
        left.axvline(e - 0.5, color="#BBBBBB", lw=0.4)
    starts, ends = [0, *edges], [*edges, n]
    centers = [(a + b - 1) / 2 for a, b in zip(starts, ends, strict=True)]
    left.set_yticks(centers, [blocks[a] for a in starts])
    left.set_xticks([])
    left.tick_params(length=0)
    left.grid(False)
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("(a) Correlation between every pair of monthly series", pad=16)
    # Kept out of tight_layout so the labels do not shrink the heatmap; anchored top to align titles.
    corner = {"transform": left.transAxes, "fontsize": 7, "color": "#333333", "in_layout": False}
    left.set_anchor("N")
    left.text(
        1.0,
        1.01,
        "Upper triangle: idiosyncratic components (factors removed)",
        ha="right",
        va="bottom",
        **corner,
    )
    left.text(
        0.0,
        -0.01,
        "Lower triangle: the series themselves",
        ha="left",
        va="top",
        **corner,
    )
    bar = fig.colorbar(image, ax=left, fraction=0.04, pad=0.02, shrink=0.8)
    bar.set_label("Correlation")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    # (b) The lasso's inputs: sorted |corr| of each target's response with every candidate.
    names = {
        "HICPOV_EA": "Headline",
        "HICPNG_EA": "Energy",
        "HICPNEF_EA": "Core",
        "IPMN_EA": "Manufacturing",
    }
    for color, (target, (r, rows)) in zip(PALETTE_MANY, profiles.items(), strict=False):
        above = int((r > 2 / np.sqrt(rows)).sum())
        right.plot(
            np.arange(1, len(r) + 1),
            r,
            color=color,
            lw=1.5 if color == PALETTE_MANY[3] else 1.1,
            label=f"{names[target]}: {above}",
        )
    rows = int(np.median([rows for _, rows in profiles.values()]))
    chance = 2 / np.sqrt(rows)
    right.axhspan(0, chance, color="#EEEEEE", zorder=0, lw=0)
    right.axhline(chance, color="#555555", lw=0.8, linestyle="--")
    right.text(
        len(r),
        chance + 0.01,
        "noise level, $2/\\sqrt{T}$",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#333333",
    )
    right.set_xlim(1, len(r))
    right.set_ylim(0, None)
    right.set_xlabel(f"Candidate predictors, ranked from most to least correlated (1 to {len(r)})")
    right.set_ylabel("Absolute correlation with the target's\nidiosyncratic component")
    right.set_title("(b) What the sparse step can choose from", pad=16)
    right.legend(loc="upper right", frameon=False, title="Target (candidates above noise)")
    fig.tight_layout()
    save_fig(
        fig,
        "fig_83_correlation",
        used_in="paper: methodology, Figure fig:residual-correlation (3.3.3)",
    )
    plt.close(fig)
    for key, values in [
        ("Max", [r[0] for r, _ in profiles.values()]),
        ("AboveNoise", [int((r > 2 / np.sqrt(m)).sum()) for r, m in profiles.values()]),
    ]:
        fmt = (lambda v: f"{v:.2f}") if key == "Max" else str
        record_number(f"Lasso{key}Min", fmt(min(values)))
        record_number(f"Lasso{key}Max", fmt(max(values)))
    record_number("LassoCandidates", str(len(r)))


def block_numbers(raw, residual):
    """Correlation within and across EA-MD-QD blocks, before and after removing the factors."""
    groups = pd.read_csv("data/metadata/data_dictionary.csv", index_col="series_id")["group"]
    for key, frame in [("Raw", raw), ("Resid", residual)]:
        u = standardized(frame)
        t, n = u.shape
        upper = np.triu_indices(n, k=1)
        r = np.abs((u.T @ u / t)[upper])
        block = groups.reindex(frame.columns).to_numpy()
        same = block[upper[0]] == block[upper[1]]
        noise = 2 / np.sqrt(t)
        record_number(f"{key}CorrWithin", f"{r[same].mean():.2f}")
        record_number(f"{key}CorrAcross", f"{r[~same].mean():.2f}")
        record_number(f"{key}PairsAboveNoise", f"{100 * np.mean(r > noise):.0f}")
        record_number(f"{key}AcrossAboveNoise", f"{100 * np.mean(r[~same] > noise):.0f}")
    record_number("CorrNoise", f"{noise:.2f}")


def settings_numbers():
    design = yaml.safe_load(Path("config/research_design.yaml").read_text())
    sparse = design["sparse_model"]
    record_number("ArMaxLag", WORDS[design["benchmarks"]["ar_max_lag"]])
    record_number("SparseMonthlyPositions", WORDS[sparse["monthly_lags"]])
    record_number("SparseQuarterlyPositions", WORDS[sparse["quarterly_lags"]])
    record_number("SparseCapDivisor", str(sparse["cap_divisor"]))
    tuning = pd.read_csv(SOURCE / "tuning.csv")
    farm = tuning[tuning.model == "FarmPredict"]
    for name, group in farm.groupby("information_set"):
        record_number(f"NCandidates{name}", str(int(group.d.max())))
    rows = farm.groupby("information_set").n_train.median()
    record_number("SparseRowsMedianM", str(int(rows["M"])))
    record_number("SparseRowsMedianME", str(int(rows["ME"])))


def main():
    settings_numbers()
    seed = yaml.safe_load(Path("config/research_design.yaml").read_text())["project"]["random_seed"]
    origins = yaml.safe_load((SOURCE / "M" / "metadata.json").read_text())["origins"]
    first, last = origins[0], origins[-1]
    names = pd.read_csv("data/metadata/data_dictionary.csv", index_col="series_id")["description"]

    cuts = []
    for information_set in ["M", "MQ", "ME", "MQE"]:
        prepared = prepare(PROFILE, information_set)
        for origin in [first, last]:
            raw, residual = residual_panel(information_set, origin, prepared)
            before = correlation_summary(raw)["mean_abs_corr"]
            after = correlation_summary(residual)["mean_abs_corr"]
            cuts.append(100 * (1 - after / before))
            if information_set == "M" and origin == first:
                main_raw, main_residual = raw, residual
    record_number("ResidCorrCutMin", f"{min(cuts):.0f}")
    record_number("ResidCorrCutMax", f"{max(cuts):.0f}")
    record_groups(main_residual)
    correlation_figure(main_raw, main_residual, lasso_correlations(first, prepare(PROFILE, "M")))
    block_numbers(main_raw, main_residual)
    core_share_numbers(common_shares())

    rng = np.random.default_rng(seed)
    columns = {}
    for label, frame in [("Series", main_raw), ("Residuals", main_residual)]:
        summary = correlation_summary(frame)
        test = fmm_diagonal_test(frame, rng)
        columns[label] = {
            "Average absolute correlation": f"{summary['mean_abs_corr']:.3f}",
            "Pairs with absolute correlation above 0.5": f"{summary['strong_pairs']}",
            "Largest absolute correlation": f"{abs(test['top_corr']):.3f}",
            "Variance share of first component (percent)": f"{100 * summary['share_pc1']:.1f}",
            "Variance share of first three components (percent)": f"{100 * summary['share_pc3']:.1f}",
            "Components above the noise edge": f"{summary['above_noise']}",
            "FMM statistic S": f"{test['statistic']:.2f}",
            "Critical value at 5 percent": f"{test['critical_05']:.2f}",
            "p-value": f"{test['p_value']:.3f}",
        }
        key = "Raw" if label == "Series" else "Resid"
        record_number(f"{key}MeanCorr", f"{summary['mean_abs_corr']:.2f}")
        record_number(f"{key}StrongPairs", str(summary["strong_pairs"]))
        record_number(f"{key}PcOneShare", f"{100 * summary['share_pc1']:.0f}")
        record_number(f"{key}AboveNoise", str(summary["above_noise"]))
        record_number(f"Fmm{key}P", f"{test['p_value']:.3f}")
        record_number(f"Fmm{key}TopCorr", f"{abs(test['top_corr']):.3f}")
        if key == "Resid":
            i, j = (frame.columns[k] for k in test["top_pair"])
            record_number("FmmResidTopPair", f"{names[i].lower()} and {names[j].lower()}")
    record_number("ResidPairs", str(summary["pairs"]))
    record_number("ResidT", str(summary["T"]))
    record_number("FmmBandwidth", str(test["bandwidth"]))
    table = pd.DataFrame(columns)
    table.index.name = None
    save_table(
        table,
        "tab_84_residual_dependence",
        notes=(
            f"The {summary['n']} monthly series of M at the first forecast date, "
            f"{pd.Period(first, 'M').strftime('%B %Y')}, on the {summary['T']} months observed for "
            "every series. Series: cleaned and standardized; residuals: series minus the common "
            "component of the dynamic factor model, standardized. Strong pairs: out of "
            f"{summary['pairs']}. Noise edge: largest eigenvalue for independent series, "
            "$(1+\\sqrt{n/T})^2$. FMM test: test of a diagonal covariance \\citep{fan2023}, on "
            f"correlations; Bartlett kernel, bandwidth {test['bandwidth']}, 5{{,}}000 multiplier "
            "draws; exploratory, as its theory assumes principal-component factors and its size "
            "at this $n$ and $T$ was not simulated. \\textit{Source:} EA-MD-QD, own calculations."
        ),
        used_in="not in the paper; its numbers are quoted in the notes of fig_83 (3.3.3)",
        index=True,
    )


if __name__ == "__main__":
    main()
