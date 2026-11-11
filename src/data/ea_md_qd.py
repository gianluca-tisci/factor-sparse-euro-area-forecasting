"""Load EA-MD-QD 09.2026 (euro-area file) and the release delay of each series.

Data contract (protocol §D): a monthly PeriodIndex from 2000-01 and one column per series
(118). Quarterly values sit in the third month of their quarter, as in the file. Values are
the adjusted series as provided (D-12), in levels; transformations are in
src/preprocessing/transform.py.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

RELEASE = Path("data/raw/ea_md_qd/2026-09")
FILE = RELEASE / "EA-MD-QD-2026-09" / "EAdata.xlsx"
SNAPSHOT = pd.Period("2026-09", "M")  # "data as of 30-Sep-2026"
TARGET = "HICPNG_EA"  # D-09

# Quarterly series published about 4 months after their quarter; all others about 2 (D-15)
# fmt: off
QUARTERLY_4_MONTHS = {
    "TASS", "TLB", "NFCASS", "NFCLB", "GGASS", "GGLB", "HHASS", "HHLB",  # credit aggregates
    "AHRDI", "AHFCE", "GNFCPS", "GNFCIR", "GHIR", "GHSR",  # sector-account ratios
    "CONSND", "HPRC",
}
# fmt: on


def base_name(name):
    """GDP_EA -> GDP, TASS.SDB_EA -> TASS.SDB, IRT3M_EACC -> IRT3M."""
    return name.rsplit("_", 1)[0]


def verify():
    """Check the file against the SHA-256 recorded in the download manifest."""
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    expected = manifest["sha256"][FILE.relative_to(RELEASE).as_posix()]
    actual = hashlib.sha256(FILE.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{FILE} does not match its manifest: {actual} != {expected}")


def load():
    """Return (data, info): levels as [month x series] and the info sheet indexed by name."""
    verify()
    info = pd.read_excel(FILE, sheet_name="info").set_index("Name")
    data = pd.read_excel(FILE, sheet_name="data", index_col=0)
    data.index = pd.PeriodIndex(data.index, freq="M")
    if list(data.columns) != list(info.index):
        raise ValueError("data and info sheets list different series")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError("months are not unique and increasing")
    return data, info


def release_lags(data, info):
    """Months between the end of a reference period and its publication, per series.

    Monthly: read from the snapshot's ragged edge (D-14). Quarterly: 2 or 4 months after
    the quarter, from the EA-MD-QD paper (D-15), counted from the quarter's third month.
    """
    lags = {}
    for name in data.columns:
        if info.loc[name, "Frequency"] == "Q":
            lags[name] = 4 if base_name(name).split(".")[0] in QUARTERLY_4_MONTHS else 2
        else:
            lags[name] = (SNAPSHOT - data[name].last_valid_index()).n
    return pd.Series(lags, name="release_lag_months")


def available(panel, lags, origin):
    """What was published by the last day of month `origin`.

    Rows after the origin are dropped; for each series, values whose reference month is
    later than origin - lag are set to missing. No value is filled in.
    """
    origin = pd.Period(origin, "M")
    out = panel.loc[:origin].copy()
    for name in out.columns:
        out.loc[out.index > origin - int(lags[name]), name] = float("nan")
    return out
