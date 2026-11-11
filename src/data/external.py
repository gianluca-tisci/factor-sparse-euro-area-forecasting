"""Immutable external snapshot for the proposed paper experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data/external/factor_sparse"
IDS = (
    "MCOILBRENTEU",
    "PNGASEUUSDM",
    "PCOALAUUSDM",
    "PCOPPUSDM",
    "WB_FOOD",
    "GPR",
    "EUEPUINDXM",
    "GSCPI",
    "INDU_006",
    "RETA_006",
    "CONS_006",
    "PMAIZMTUSDM",
    "PSOYBUSDM",
    "PSOILUSDM",
    "PCOCOUSDM",
    "PCOTTINDUSDM",
    "PRUBBUSDM",
    "PALUMUSDM",
    "PNICKUSDM",
    "PIORECRUSDM",
    "PRAWMINDEXM",
    "FOODPR",
    "FOODUN",
    "IGDSX",
)


def load_external(root=ROOT):
    """Validate source levels and return once-transformed monthly observations."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if tuple(manifest["series"]) != IDS:
        raise ValueError("External indicator membership/order mismatch")
    series = {}
    for name, meta in manifest["series"].items():
        path = root / meta["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256_processed"]:
            raise ValueError(f"External checksum mismatch: {name}")
        frame = pd.read_csv(path, index_col=0)
        if list(frame.columns) != [name]:
            raise ValueError(f"External columns mismatch: {name}")
        x = frame[name]
        x.index = pd.PeriodIndex(x.index, freq="M")
        if not x.index.is_unique or not x.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered months: {name}")
        if not x.index.equals(pd.period_range(x.index[0], x.index[-1], freq="M")):
            raise ValueError(f"Missing calendar rows: {name}")
        if (
            len(x) != meta["file_rows"]
            or str(x.index[0]) != meta["file_first"]
            or str(x.index[-1]) != meta["file_last"]
        ):
            raise ValueError(f"External dates/count mismatch: {name}")
        if meta["stored_values"] != "levels" or meta["frequency"] != "M":
            raise ValueError(f"Unexpected storage convention: {name}")
        if meta["transformation"] == "100*dlog":
            if (x.dropna() <= 0).any():
                raise ValueError(f"Nonpositive log input: {name}")
            x = 100 * np.log(x).diff()
        elif meta["transformation"] != "level":
            raise ValueError(f"Unknown transformation: {name}")
        if not np.isfinite(x.dropna()).all():
            raise ValueError(f"Nonfinite observations: {name}")
        series[name] = x
    panel = pd.DataFrame(series)
    if panel.T.duplicated().any():
        raise ValueError("Duplicate external indicator values")
    return panel, manifest
