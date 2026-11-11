"""Display the saved robustness table without its four including-2020 rows."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src import _style

SOURCE = Path("outputs/reports/results_reorganized/tables/tab_101_robustness_summary.tex")
OUT = Path("outputs/reports/robustness_table_display")


def main():
    source = SOURCE.read_bytes()
    text = source.decode().replace("\r\n", "\n")
    body = text.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    rows = [
        [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
        for line in body.splitlines()
        if line.strip()
    ]
    assert len(rows) == 11
    frame = pd.DataFrame(rows, columns=["Sample", "Target", "Scope", "Best", "MSE / AR"])
    frame = frame[frame.Sample.ne("including 2020")]
    assert len(frame) == 7
    frame["Sample"] = frame.Sample.replace(
        {"broader": "All forecasts", "exclude Feb 2022--Dec 2023": "Without Feb 2022--Dec 2023"}
    )
    # Model names as in the paper's Table 3 (tab:models): "DFM+AR-MQ" -> "DFM-AR in MQ"
    names = {"DFM+AR": "DFM-AR", "PCA+VAR": "PCA-VAR"}
    best = frame.Best.str.rsplit("-", n=1, expand=True)
    assert best[1].isin(["M", "MQ", "ME", "MQE"]).all()
    frame["Best"] = best[0].replace(names) + " in " + best[1]
    old = _style.OUT
    try:
        _style.OUT = OUT
        _style.save_table(
            frame,
            "tab_101_robustness_summary",
            index=False,
            notes=(
                "Most accurate specification of each target, with the same forecasts scored on "
                "two other samples. All forecasts: all \\PaperOrigins{} forecast dates, 2020 and "
                "2026 target months included, all information sets. Without Feb 2022--Dec 2023: "
                "target months to December 2025, without 2020 and without February 2022 to "
                "December 2023; inflation targets in M and MQ only. \\textit{Source:} Own "
                "calculations."
            ),
        )
    finally:
        _style.OUT = old
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "sha256": hashlib.sha256(source).hexdigest(),
                "rows_removed": 4,
                "rows_retained": len(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "robustness_table_display",
        [str(OUT / "tables/tab_101_robustness_summary.tex")],
        "Appendix robustness table: including-2020 rows omitted",
    )


if __name__ == "__main__":
    main()
