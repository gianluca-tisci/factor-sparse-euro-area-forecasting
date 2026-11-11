"""Stationarity transformations from EA-MD-QD's code columns (D-16: TR3).

Codes (EA-MD-QD data description, Table 1): 1 = 100 log x, 2 = 100 Δlog x,
3 = 100 Δ² log x, 4 = x, 5 = Δx. EA-MD-QD's own code omits the factor 100; scale does not
matter after standardization (D-19), and the target is in percent per month.
Quarterly series are differenced across consecutive quarters and stay in their quarter's
third month. Nothing is fitted, so the transformation may be applied before the
availability mask of src/data/ea_md_qd.available.
"""

import numpy as np
import pandas as pd


def transform(data, info, codes="TR3"):
    """Transformed panel with the same [month x series] shape as `data`."""
    out = {}
    for name in data.columns:
        s = data[name]
        if info.loc[name, "Frequency"] == "Q":
            s = s[s.index.month % 3 == 0]
        out[name] = _apply(s, int(info.loc[name, codes]), name).reindex(data.index)
    return pd.DataFrame(out)


def _apply(s, code, name):
    if code in (1, 2, 3) and (s.dropna() <= 0).any():
        raise ValueError(f"{name}: code {code} takes logs but the series has values <= 0")
    if code == 1:
        return 100 * np.log(s)
    if code == 2:
        return 100 * np.log(s).diff()
    if code == 3:
        return 100 * np.log(s).diff().diff()
    if code == 4:
        return s
    if code == 5:
        return s.diff()
    raise ValueError(f"{name}: unknown transformation code {code}")
