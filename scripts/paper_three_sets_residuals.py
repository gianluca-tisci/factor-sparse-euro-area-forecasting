"""Recompute the residual-correlation range for the three reported sets only."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.sparse_diagnostics import PROFILE, SOURCE, residual_panel
from src import _style
from src.factors.residual_diagnostics import correlation_summary
from src.forecasting.experiment import prepare

OUT = Path("outputs/reports/paper_three_sets/residual_correlations")


def main():
    if OUT.exists():
        raise FileExistsError("Use a fresh output directory")
    paths = [Path(__file__), Path("scripts/sparse_diagnostics.py")]
    rows = []
    with threadpool_limits(limits=1):
        for panel in ["M", "MQ", "ME"]:
            meta_path = SOURCE / panel / "metadata.json"
            paths.append(meta_path)
            meta = json.loads(meta_path.read_text())
            origins = [meta["origins"][0], meta["origins"][-1]]
            prepared = prepare(PROFILE, panel)
            for origin in origins:
                paths.append(SOURCE / panel / origin / "parameters.npz")
                raw, residual = residual_panel(panel, origin, prepared)
                before = correlation_summary(raw)["mean_abs_corr"]
                after = correlation_summary(residual)["mean_abs_corr"]
                rows.append(
                    {
                        "panel": panel,
                        "origin": origin,
                        "before": before,
                        "after": after,
                        "reduction_percent": 100 * (1 - after / before),
                    }
                )
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / "correlations.csv", index=False)
    old = _style.OUT
    try:
        _style.OUT = OUT
        _style.record_number("PaperResidualCorrCutMin", f"{frame.reduction_percent.min():.0f}")
        _style.record_number("PaperResidualCorrCutMax", f"{frame.reduction_percent.max():.0f}")
    finally:
        _style.OUT = old
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "parameter_refits": 0,
                "fixed_parameter_smoothers": len(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "paper_three_sets_residuals",
        [str(OUT / "correlations.csv")],
        "Residual-correlation range: M, MQ and ME, first and last origins",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
