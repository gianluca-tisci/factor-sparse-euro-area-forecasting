"""Export draft exhibits from saved results only; never import estimators.

Run from the repository root: python -m scripts.paper_draft_report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src._style import record_number, save_table

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
FIXED = Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr")
TARGETS = {
    "HICPOV_EA": "Headline",
    "HICPNG_EA": "Energy",
    "HICPNEF_EA": "Core",
    "IPMN_EA": "Manufacturing",
}


def clark_west_p(row):
    """One-sided Clark--West p-value from the saved statistic (D-38).

    The saved run reports two-sided probabilities; the alternative is that the larger model
    forecasts better, so only a positive statistic counts against the null. At the next
    rerun src/forecasting/paper.py writes this p-value itself and this helper reads it.
    """
    assert row.test == "Clark_West"
    np.testing.assert_allclose(row.pvalue_two_sided, 2 * norm.sf(abs(row.statistic)), rtol=1e-9)
    return norm.sf(row.statistic)


def decimals(reference):
    """Decimals that print `reference` with four significant digits."""
    return max(0, 3 - int(np.floor(np.log10(abs(reference)))))


def bold_column_minimum(matrix, places):
    """Fixed decimals per table; the lowest value of each column in bold."""
    out = matrix.map(lambda v: "---" if pd.isna(v) else f"{v:.{places}f}")
    for column in matrix.columns:
        if matrix[column].notna().any():
            row = matrix[column].idxmin()
            out.loc[row, column] = rf"\textbf{{{out.loc[row, column]}}}"
    return out


def main():
    scores = pd.read_csv(ROOT / "scores.csv").query('sample == "primary"')
    pairs = pd.read_csv(ROOT / "paired_comparisons.csv").query('sample == "primary"')
    forecasts = pd.read_csv(ROOT / "forecasts.csv")
    primary = forecasts[
        (forecasts.target_month <= "2025-12") & ~forecasts.target_month.str.startswith("2020")
    ].copy()
    assert primary.valid.all()
    assert not primary.duplicated(["target", "origin", "information_set", "model"]).any()
    notes = (
        r"Primary matched target months through December 2025, excluding 2020; "
        r"MSE in squared percentage points of monthly log changes. "
        r"M: monthly panel; Q: quarterly additions; E: external monthly additions. "
        r"RW and AR are shared forecasts; --- denotes an inapplicable specification. "
        r"Expanding-window estimation and in-window tuning as in Section~\ref{sec:methods}. "
        r"Bold: the lowest MSE in each column; it does not imply statistical significance. "
        r"\textit{Source:} Saved paper experiment, EA-MD-QD and external providers."
    )
    for i, (target, name) in enumerate(TARGETS.items(), 90):
        s = scores[scores.target == target]
        n = int(s.n.iloc[0])
        assert (s.n == n).all() and len(s) == 24
        dates = None
        for _, row in s.iterrows():
            f = primary[
                (primary.target == target)
                & (primary.model == row.model)
                & (primary.information_set == row.information_set)
            ]
            assert len(f) == n
            np.testing.assert_allclose((f.error**2).mean(), row.MSE, rtol=1e-12)
            current = set(f.target_month)
            assert dates is None or dates == current
            dates = current
        matrix = pd.read_csv(ROOT / f"mse_{target}.csv", index_col=0)
        for model in matrix.index:
            for panel in matrix.columns:
                value = matrix.loc[model, panel]
                if pd.notna(value):
                    found = s[
                        (s.model == model)
                        & (s.information_set == ("shared" if model in ["RW", "AR"] else panel))
                    ]
                    np.testing.assert_allclose(value, found.MSE.item(), rtol=1e-12)
        matrix.index = matrix.index.str.replace("_", "-")
        matrix.index.name = "Model"
        formatted = bold_column_minimum(matrix, decimals(matrix.loc["AR", "M"]))
        save_table(
            formatted,
            f"tab_{i}_{name.lower()}_mse",
            notes=f"N = {n}. " + notes,
            used_in="Results: primary matrix",
            escape=False,
        )
        record_number(f"Draft{name}N", str(n))
        winner = s.loc[s.MSE.idxmin()]
        record_number(f"Draft{name}WinnerMSE", f"{winner.MSE:.5f}")
        ar = s.loc[s.model == "AR", "MSE"].item()
        record_number(f"Draft{name}GainAR", f"{100 * (1 - winner.MSE / ar):.2f}")
        p = pairs[
            (pairs.target == target)
            & (pairs.augmented == f"{winner.information_set}:{winner.model}")
            & (pairs.baseline == "shared:AR")
        ].pvalue_two_sided.item()
        record_number(f"Draft{name}WinnerP", f"{p:.3f}")
    comparison = pd.read_csv(FIXED / "comparison.csv")
    fixed_forecasts = pd.read_csv(FIXED / "forecasts.csv")
    counts = {}
    for target in TARGETS:
        baseline = primary[
            (primary.target == target) & (primary.model == "DFM") & (primary.information_set == "M")
        ]
        fixed = fixed_forecasts[fixed_forecasts.target == target].merge(
            baseline[["origin", "target_month", "actual"]],
            on=["origin", "target_month"],
            validate="one_to_one",
            suffixes=("", "_base"),
        )
        counts[target] = len(baseline)
        assert len(fixed) == counts[target]
        np.testing.assert_allclose(fixed.actual, fixed.actual_base, rtol=1e-12)
        for _, row in comparison[comparison.target == target].iterrows():
            assert row.n == counts[target]
            if row.model == "Fixed_DFM_M_external_sparse":
                expected_mse = (fixed.error**2).mean()
            else:
                model, panel = row.model.rsplit("-", 1)
                expected_mse = scores[
                    (scores.target == target)
                    & (scores.model == model)
                    & (scores.information_set == panel)
                ].MSE.item()
            np.testing.assert_allclose(row.MSE, expected_mse, rtol=1e-12)
    assert len({counts[t] for t in TARGETS if t != "IPMN_EA"}) == 1
    table = comparison.pivot(index="model", columns="target", values="MSE")
    table = table.loc[
        ["DFM-M", "FarmPredict-M", "FarmPredict-ME", "LASSO-ME", "Fixed_DFM_M_external_sparse"],
        list(TARGETS),
    ]
    table.columns = list(TARGETS.values())
    table.index = [
        "DFM-M",
        "FarmPredict-M",
        "FarmPredict-ME",
        "LASSO-ME",
        "Fixed DFM-M + Sparse(M+E)",
    ]
    table.index.name = "Model"
    formatted = pd.concat(
        [bold_column_minimum(table[[c]], decimals(table[c].iloc[0])) for c in table.columns],
        axis=1,
    )
    save_table(
        formatted,
        "tab_94_fixed_factor_ablation",
        notes=(
            r"Same primary dates as Tables~\ref{tab:headline-mse}--\ref{tab:manufacturing-mse}; "
            f"inflation N = {counts['HICPOV_EA']}, manufacturing N = {counts['IPMN_EA']}. "
            r"MSE in squared percentage points. "
            r"The ablation retains saved DFM-M common forecasts and projects E onto the fixed M "
            r"factors within each origin; the unchanged modified-BIC LASSO uses M and E residuals. "
            r"No new factor estimation. Comparisons are descriptive. "
            r"Bold: the lowest MSE for each target. "
            r"\textit{Source:} Saved fixed-factor ablation comparison.csv."
        ),
        used_in="Results: separate ablation",
        escape=False,
    )
    tuning = pd.read_csv(ROOT / "tuning.csv")
    matched = tuning.merge(
        primary[["target", "origin", "model", "information_set"]], validate="one_to_one"
    )
    for target, name in TARGETS.items():
        for panel in ["M", "ME"]:
            x = matched[
                (matched.target == target)
                & (matched.model == "FarmPredict")
                & (matched.information_set == panel)
            ].selected
            record_number(f"Draft{name}Zero{panel}", f"{100 * x.eq(0).mean():.1f}")
    for target, name, panel in [("HICPNEF_EA", "Core", "M"), ("IPMN_EA", "Manufacturing", "MQ")]:
        row = pairs[
            (pairs.target == target)
            & (pairs.augmented == f"{panel}:FarmPredict")
            & (pairs.baseline == f"{panel}:DFM")
        ].iloc[0]
        record_number(f"Draft{name}SparseGain", f"{100 * (1 - row.MSE_ratio):.2f}")
        record_number(f"Draft{name}SparseP", f"{clark_west_p(row):.3f}")
    for target, name, panel in [
        ("HICPNEF_EA", "CoreMQ", "MQ"),
        ("HICPOV_EA", "HeadlineME", "ME"),
        ("HICPOV_EA", "HeadlineMQE", "MQE"),
    ]:
        row = pairs[
            (pairs.target == target)
            & (pairs.augmented == f"{panel}:FarmPredict")
            & (pairs.baseline == f"{panel}:DFM")
        ].iloc[0]
        record_number(f"Draft{name}SparseGain", f"{100 * (1 - row.MSE_ratio):.2f}")
        record_number(f"Draft{name}SparseP", f"{clark_west_p(row):.3f}")
    bandwidth = pairs.HAC_bandwidth.unique()
    assert len(bandwidth) == 1
    record_number("HacBandwidth", str(int(bandwidth[0])))
    ext = pd.read_csv(FIXED / "external_selection.csv")
    retail = ext[(ext.target == "HICPOV_EA") & (ext.predictor == "RETA_006")].iloc[0]
    record_number("DraftRetailSelections", str(int(retail.selected_origins)))
    record_number("DraftRetailSelectionPct", f"{retail.selection_pct:.1f}")
    # Descriptive exclusion of 2022, retaining the primary sample's other exclusions.
    for panel in ["M", "ME"]:
        x = primary[
            (primary.target == "HICPNG_EA")
            & (primary.model == "DFM")
            & (primary.information_set == panel)
            & ~primary.target_month.str.startswith("2022")
        ]
        record_number(f"DraftEnergyExCrisis{panel}", f"{(x.error**2).mean():.3f}")
    print(
        "Verified 96 primary scores, four original matrices, matched calendars; exported five tables and draft macros."
    )


if __name__ == "__main__":
    main()
