"""Display the monthly M-to-ME external comparison, preserving full saved tables."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts import results_ladder_exhibits as ladder
from src import _style

SOURCE = Path("outputs/reports/results_ladder/tables")
OUT = Path("outputs/reports/monthly_external_tables")


def monthly_rows(table_id):
    path = SOURCE / (table_id + ".tex")
    text = path.read_text()
    body = text.split("\\midrule\n", 1)[1].split("\\midrule", 1)[0]
    rows = [
        [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
        for line in body.splitlines()
        if line.strip()
    ]
    assert len(rows) == 3 and rows[0][0] == r"M $\to$ ME"
    assert [row[1] for row in rows] == ["Headline", "Energy", "Core"]
    return [row[1:] for row in rows], path


def main():
    old, old_ladder = _style.OUT, ladder.OUT
    hashes = {}
    try:
        _style.OUT = ladder.OUT = OUT
        table_id = "tab_108_external_routes"
        rows, path = monthly_rows(table_id)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        routes = ["Without E", "E in factors", "E in sparse step", "E in LASSO"]
        columns = [("", "Target")] + [
            (sample, route) for sample in ["Full sample", "Without 2021--2023"] for route in routes
        ]
        frame = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))
        _style.save_table(
            frame,
            table_id,
            index=False,
            escape=False,
            column_format="@{}lrrrrrrrr@{}",
            notes="MSE relative to AR; lowest of the four in each row and sample in bold. The first three columns are FP-DFM: in M without E; in ME with E entering factor estimation and the sparse step; and with the M factors and common forecast fixed and external residuals added only to the sparse step. E in LASSO is direct LASSO in ME. Full sample: matched target months through December 2025, excluding 2020; the second sample additionally excludes January 2021--December 2023. Source: saved paper and fixed-factor experiments; M to ME only.",
        )
        ladder.group_headers(table_id)
        table_id = "tab_109_fixed_factor_mse"
        rows, path = monthly_rows(table_id)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        columns = [
            ("", "Target"),
            ("DFM", "Without E"),
            ("DFM", "E in factors"),
            ("FP-DFM", "E in factors"),
            ("FP-DFM", "E in sparse step"),
            ("LASSO", "E in LASSO"),
        ]
        frame = pd.DataFrame(rows, columns=pd.MultiIndex.from_tuples(columns))
        _style.save_table(
            frame,
            table_id,
            index=False,
            escape=False,
            column_format="@{}lrrrrr@{}",
            notes="MSE in squared percentage points. Without E: M; E in factors: ME; E in sparse step: FP-DFM with the M factors and common forecast fixed, adding external residuals only to the sparse step; E in LASSO: direct LASSO in ME. Candidate predictors and training rows may differ. Matched target months through December 2025, excluding 2020. Source: saved paper and fixed-factor experiments.",
        )
        ladder.group_headers(table_id)
    finally:
        _style.OUT, ladder.OUT = old, old_ladder
    (OUT / "audit.json").write_text(
        json.dumps({"sources": hashes, "rows_per_table": 3, "model_refits": 0}, indent=2),
        encoding="utf-8",
    )
    _style.record_output(
        "monthly_external_tables",
        [str(OUT / "audit.json")],
        "External discussion restricted to M/ME; full saved tables retained",
    )


if __name__ == "__main__":
    main()
