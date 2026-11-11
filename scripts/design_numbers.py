"""Design counts and fit diagnostics quoted in the methodology section (D-68).

Run from the repo root:  uv run python -m scripts.design_numbers
Reads the paper run's own definitions (src/forecasting/paper.py, config/forecast_paper.yaml,
data/external/factor_sparse/manifest.json) and its saved fit log
(outputs/forecasts/paper_experiment_nonzero_iqr/fit_monitoring.csv), so the text cannot drift from the code.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src._style import record_number
from src.forecasting import dfm
from src.forecasting.design import complete_design
from src.forecasting.dfm import MixedFrequencyEM
from src.forecasting.paper import MODELS, PANELS, TARGETS, specifications

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def words(n):
    """Counts below ten are spelled out in the text, the rest are digits."""
    return WORDS[n] if n < 10 else str(n)


def factor_var_order(r):
    """Lag order of the factor VAR as built by the estimator (D-30), read from a toy start fit."""
    rng = np.random.default_rng(0)
    index = pd.period_range("2000-01", periods=60, freq="M")
    panel = pd.DataFrame(rng.standard_normal((60, 6)), index=index)
    return MixedFrequencyEM(panel, 6, r).start().ar.shape[1] // r


def power_of_ten(value):
    """1e-06 -> $10^{-6}$; fails loudly if the setting is not a power of ten."""
    exponent = int(np.rint(np.log10(value)))
    assert np.isclose(value, 10.0**exponent)
    return rf"$10^{{{exponent}}}$"


def sparse_min_rows():
    """Smallest number of complete training rows the sparse step accepts, found by probing."""
    for n in range(1, 500):
        x = pd.DataFrame({"a": np.arange(n, dtype=float)})
        try:
            complete_design(x, np.ones(n), pd.Series({"a": 0.0}))
            return n
        except ValueError:
            continue
    raise RuntimeError("No minimum training size found")


def main():
    design = yaml.safe_load(Path("config/research_design.yaml").read_text())["factor_model"]
    record_number("FactorsR", str(design["selected_r"]))
    record_number("FactorVarOrder", str(factor_var_order(design["selected_r"])))
    config = yaml.safe_load(Path("config/research_design.yaml").read_text())
    record_number("PcaVarMaxLag", str(config["factor_forecast"]["max_lag"]))
    monitoring = design["fit_monitoring"]
    paper = yaml.safe_load(Path("config/forecast_paper.yaml").read_text())
    inherited = yaml.safe_load(Path(paper["inherits"]).read_text())
    assert paper["reestimation_start"] == "cold"  # every fit uses the cold-start cap
    tolerance = inspect.signature(MixedFrequencyEM.fit).parameters["tolerance"].default
    record_number("EmTolerance", power_of_ten(tolerance))
    record_number("EmMaxIter", f"{monitoring['em_cap_cold_first_fit']:,}")
    record_number("FlagRho", f"{monitoring['flag_rho']:.2f}")
    record_number("InitialFactorVariance", f"{inherited['initial_factor_variance']:g}")
    assert dfm.FLOOR == inherited["variance_floor"]
    record_number("VarianceFloor", power_of_ten(dfm.FLOOR))
    record_number("SparseMinRows", str(sparse_min_rows()))
    fits = pd.read_csv("outputs/forecasts/paper_experiment_nonzero_iqr/fit_monitoring.csv")
    record_number("NDfmFits", str(len(fits)))
    not_converged = int((~fits["converged"]).sum())
    record_number("NDfmNotConverged", "none" if not_converged == 0 else str(not_converged))
    record_number("NDfmFlagged", str(int(fits["flagged"].sum())))
    record_number("MaxFactorRoot", f"{fits['rho'].max():.2f}")
    quarterly = yaml.safe_load(Path("config/forecast_paper.yaml").read_text())["quarterly_series"]
    external = json.loads(Path("data/external/factor_sparse/manifest.json").read_text())["series"]
    meta = pd.read_csv("data/metadata/data_dictionary.csv")
    monthly = int((meta["frequency"] == "M").sum())
    record_number("NTargets", words(len(TARGETS)))
    record_number("NInfoSets", words(len(PANELS)))
    record_number("NModelFamilies", words(len(MODELS)))
    record_number("NSpecifications", str(len(specifications())))
    record_number("NQuarterlySelected", str(len(quarterly)))
    record_number("NExternal", str(len(external)))
    record_number("NPanelMQ", str(monthly + len(quarterly)))
    record_number("NPanelME", str(monthly + len(external)))
    record_number("NPanelMQE", str(monthly + len(quarterly) + len(external)))


if __name__ == "__main__":
    main()
