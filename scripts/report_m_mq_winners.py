"""Export within-information-set winners from saved primary paper scores only."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/m_mq_winners")
TARGETS = {
    "HICPOV_EA": "Headline",
    "HICPNG_EA": "Energy",
    "HICPNEF_EA": "Core",
    "IPMN_EA": "Manufacturing",
}
LABELS = {
    "RW": "RW",
    "AR": "AR",
    "PCR": "PCR",
    "PCA_VAR": "PCA+VAR",
    "PCA_VAR_LASSO": "FarmPredict (PCA+VAR+LASSO)",
    "DFM": "DFM",
    "DFM_AR": "DFM+AR",
    "LASSO": "LASSO",
    "FarmPredict": "FarmPredict (DFM+LASSO)",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    sections = []
    for panel, first_id in [("M", 95), ("MQ", 97)]:
        summary, gains = [], {}
        for target, title in TARGETS.items():
            path = ROOT / "samples" / "primary" / f"MSE_{target}.csv"
            counts_path = path.with_name(f"n_{target}.csv")
            for source in [path, counts_path, ROOT / panel / "metadata.json"]:
                hashes[source.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = json.loads((ROOT / panel / "metadata.json").read_text())
            assert metadata["profile"]["reestimation_start"] == "cold"
            assert metadata["completion"] == "COMPLETE"
            values = pd.read_csv(path, index_col=0)[panel].dropna()
            assert len(values) == (9 if panel == "M" else 6)
            assert np.isfinite(values).all() and values.gt(0).all()
            counts = pd.read_csv(counts_path, index_col=0).loc[values.index, panel]
            assert counts.nunique() == 1 and counts.iloc[0] > 0
            ranked = values.sort_values()
            winner, runner = ranked.index[:2]
            reduction = 100 * (1 - ranked.iloc[0] / values)
            assert reduction.ge(-1e-10).all() and reduction[winner] == 0
            gains[title] = reduction.rename(index=LABELS)
            summary.append(
                {
                    "Target": title,
                    "Winner": LABELS[winner],
                    "Winner MSE": f"{ranked.iloc[0]:.6f}",
                    "Runner-up": LABELS[runner],
                    "Runner-up MSE": f"{ranked.iloc[1]:.6f}",
                    "Reduction (%)": f"{reduction[runner]:.3f}",
                    "N": int(counts.iloc[0]),
                }
            )
        summary = pd.DataFrame(summary).set_index("Target")
        gains = pd.DataFrame(gains)
        summary.to_csv(OUT / f"winners_{panel}.csv")
        gains.to_csv(OUT / f"reductions_{panel}.csv")
        notes = (
            f"Information set {panel}, evaluated separately. One-step forecasts; origins from "
            "2015-01; scoring through 2025-12; 2020 target months excluded. "
            "All-cold paper run. Reduction is 100(1 - MSE(winner)/MSE(comparator)); "
            "zero identifies the winner. FarmPredict includes both the DFM+LASSO and "
            "PCA+VAR+LASSO variants; the latter is estimated only in M here. "
            "Rankings are descriptive, not significance tests. "
            "Source: saved EA-MD-QD paper experiment primary MSE tables."
        )
        _style.save_table(
            summary,
            f"tab_{first_id}_winners_{panel.lower()}",
            notes=notes,
            used_in="M/MQ within-set comparison, no 2020",
        )
        _style.save_table(
            gains,
            f"tab_{first_id + 1}_winner_reductions_{panel.lower()}",
            notes=notes,
            used_in="M/MQ within-set comparison, no 2020",
            float_format="%.3f",
        )
        sections.append(
            f"<h2>{panel}: winners and runners-up</h2>"
            + summary.to_html()
            + f"<h2>{panel}: winner MSE reductions (%)</h2>"
            + gains.to_html(float_format=lambda value: f"{value:.3f}")
            + f"<p>{notes}</p>"
        )
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>M and MQ winners, excluding 2020</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1400px}"
        "table{border-collapse:collapse}th,td{padding:.5rem;border:1px solid #ccc}"
        "td{text-align:right}p{max-width:1000px}</style>"
        "<h1>M and MQ: separate primary-sample comparisons</h1>"
        "<p>No external-series information sets are included.</p>" + "".join(sections) + "</html>",
        encoding="utf-8",
    )
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "script": "scripts/report_m_mq_winners.py",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported four tables and report to {OUT}")


if __name__ == "__main__":
    main()
