"""Re-evaluate current saved forecasts in separate nonzero-IQR robustness outputs."""

import hashlib
import json
from pathlib import Path

from scripts import external_crisis_sensitivity as external
from scripts import inflation_crisis_sensitivity as inflation
from scripts import plot_fixed_factor_nonzero as fixed_plot
from scripts import report_energy_lasso_selections as selections
from scripts import report_mq_energy_diagnostics as diagnostics
from scripts import report_nonzero_energy_reversal as reversal
from src import _style

ROOT = Path("outputs/reports/nonzero_robustness")
SOURCE = Path("outputs/forecasts/paper_experiment_nonzero_iqr")


def main():
    paths = [SOURCE / "forecasts.csv", Path(__file__)]
    paths += [
        Path(module.__file__)
        for module in [external, inflation, fixed_plot, selections, diagnostics, reversal]
    ]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if (ROOT / "refresh_audit.json").exists():
        raise FileExistsError("Refuse to overwrite a completed report refresh")
    old_out = _style.OUT
    _style.OUT = ROOT
    try:
        inflation.CONFIG = Path("config/evaluation_inflation_crisis_nonzero.yaml")
        inflation.OUT = ROOT / "inflation_crisis"
        inflation.main()
        external.CONFIG = Path("config/evaluation_external_crisis_nonzero.yaml")
        external.main()
        selections.ROOT = SOURCE
        selections.OUT = ROOT / "energy_lasso_selections"
        selections.main()
        diagnostics.ROOT = SOURCE
        diagnostics.OUT = ROOT / "mq_energy_diagnostics"
        diagnostics.main()
        fixed_plot.main()
        reversal.main()
    finally:
        _style.OUT = old_out
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (ROOT / "refresh_audit.json").write_text(
        json.dumps(
            {
                "run": str(SOURCE),
                "sources": hashes,
                "model_refits": 0,
                "historical_outputs_overwritten": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
