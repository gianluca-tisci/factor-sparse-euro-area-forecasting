"""Restrict manuscript diagnostics and robustness rankings to M, MQ and ME."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/paper_three_sets")
SETS = ["M", "MQ", "ME"]


def read_table(path):
    text = path.read_text()
    body = text.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    return [
        [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
        for line in body.splitlines()
        if " & " in line
    ]


def main():
    if OUT.exists():
        raise FileExistsError("Use a fresh directory; preserve saved reports")
    paths = [
        ROOT / x
        for x in ["fit_monitoring.csv", "forecasts.csv", "data_availability.csv", "robustness.csv"]
    ]
    ladder = Path("outputs/reports/results_ladder/tables/tab_105_ladder_all_sets.tex")
    robustness = Path(
        "outputs/reports/robustness_table_display/tables/tab_101_robustness_summary.tex"
    )
    paths += [ladder, robustness, Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    fits = pd.read_csv(paths[0])
    fits = fits[fits.information_set.isin(SETS)]
    forecasts = pd.read_csv(paths[1])
    forecasts = forecasts[forecasts.information_set.isin([*SETS, "shared"])]
    n_specs = forecasts.groupby("target").apply(
        lambda g: len(g[["information_set", "model"]].drop_duplicates()), include_groups=False
    )
    assert n_specs.nunique() == 1 and n_specs.iloc[0] == 20
    diag = []
    for label in [*SETS, "All"]:
        g = fits if label == "All" else fits[fits.information_set.eq(label)]
        diag.append(
            {
                "Set": label,
                "Fits": len(g),
                "Not converged": int((~g.converged).sum()),
                "Median iterations": f"{g.iterations.median():.0f}",
                "Maximum iterations": g.iterations.max(),
                "Largest VAR root": f"{g.rho.max():.2f}",
                "Flagged": int(g.flagged.sum()),
            }
        )
    ladder_rows = read_table(ladder)
    ladder_frame = pd.DataFrame(
        ladder_rows,
        columns=[
            "Target",
            "Set",
            "FP-PCA / PCA-VAR",
            "FP-DFM / DFM",
            "DFM-AR / DFM",
            "FP-DFM / LASSO",
            "FP-PCA / LASSO",
        ],
    )
    ladder_frame = ladder_frame[ladder_frame.Set.isin(SETS)]
    assert len(ladder_frame) == 12
    robust_frame = pd.DataFrame(
        read_table(robustness), columns=["Sample", "Target", "Scope", "Best", "MSE / AR"]
    )
    scores = pd.read_csv(paths[3])
    broad = scores[scores["sample"].eq("broader") & scores.information_set.isin([*SETS, "shared"])]
    winners = []
    labels = {
        "HICPOV_EA": "Headline",
        "HICPNG_EA": "Energy",
        "HICPNEF_EA": "Core",
        "IPMN_EA": "Manufacturing",
    }
    model_labels = {
        "DFM_AR": "DFM-AR",
        "FarmPredict": "FP-DFM",
        "PCA_VAR": "PCA-VAR",
        "PCA_VAR_LASSO": "FP-PCA",
    }
    for target, title in labels.items():
        row = broad[broad.target.eq(target)].sort_values("MSE").iloc[0]
        winners.append(row)
        chosen = robust_frame.Target.eq(title) & robust_frame.Sample.eq("All forecasts")
        assert chosen.sum() == 1
        model = model_labels.get(row.model, row.model)
        robust_frame.loc[chosen, "Best"] = (
            model if row.information_set == "shared" else f"{model} in {row.information_set}"
        )
        robust_frame.loc[chosen, "Scope"] = "M/MQ/ME"
        robust_frame.loc[chosen, "MSE / AR"] = f"{row.relative_MSE_AR:.3f}"
    availability = pd.read_csv(paths[2])
    availability = availability[availability.information_set.eq("ME")]
    last = availability[availability.origin.eq(availability.origin.max())]
    first = availability[availability.origin.eq(availability.origin.min())]
    macros = {
        "PaperRetainedSpecifications": str(int(n_specs.iloc[0])),
        "PaperRetainedDfmFits": str(len(fits)),
        "PaperRetainedDfmNotConverged": str(int((~fits.converged).sum())),
        "PaperRetainedDfmFlagged": str(int(fits.flagged.sum())),
        "PaperRetainedMaxRoot": f"{fits.rho.max():.2f}",
        "PaperRatioFirstME": f"{len(first) / first.published.max():.2f}",
        "PaperOutliersLastME": str(int(last.outliers.sum())),
        "PaperOutliersCurrencyME": str(int(last[last.series.eq("CURR_EACC")].outliers.sum())),
        "PaperOutliersMaxME": str(int(availability.groupby("origin").outliers.sum().max())),
    }
    OUT.mkdir(parents=True)
    old = _style.OUT
    try:
        _style.OUT = OUT
        for name, value in macros.items():
            _style.record_number(name, value)
        _style.save_table(
            pd.DataFrame(diag).set_index("Set"),
            "tab_82_em_diagnostics",
            notes="Retained information sets M, MQ and ME only; one EM fit per date and set. Source: saved paper experiment.",
        )
        _style.save_table(
            ladder_frame,
            "tab_105_ladder_all_sets",
            index=False,
            escape=False,
            notes="M, MQ and ME only. Saved ratios and p-values unchanged. Source: saved paper experiment.",
        )
        _style.save_table(
            robust_frame,
            "tab_101_robustness_summary",
            index=False,
            escape=False,
            notes="All-forecast winners recomputed across M, MQ and ME and the shared benchmarks; narrower-exclusion comparisons retain M/MQ. No refits. Source: saved robustness scores.",
        )
    finally:
        _style.OUT = old
    fits.to_csv(OUT / "retained_fit_diagnostics.csv", index=False)
    pd.DataFrame(winners).to_csv(OUT / "retained_robustness_winners.csv", index=False)
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "sources": hashes,
                "retained_sets": SETS,
                "model_refits": 0,
                "specifications_per_target": int(n_specs.iloc[0]),
                "dfm_fits": len(fits),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "paper_three_sets",
        [str(OUT / "audit.json"), str(OUT / "numbers.tex")],
        "Manuscript scope M/MQ/ME; historical forecasts preserved",
    )
    print(json.dumps(macros, indent=2))
    print(
        pd.DataFrame(winners)[["target", "model", "information_set", "relative_MSE_AR"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
