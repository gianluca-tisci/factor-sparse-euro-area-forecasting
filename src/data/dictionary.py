"""Build data/metadata/data_dictionary.csv, protocol Stage 1's data dictionary.

Run from the repo root:  uv run python -m src.data.dictionary
Descriptions and units come from data/metadata/ea_md_qd_series.csv, extracted once from
EA-MD-QD's _data_description.pdf (Table 2); everything else is computed from the data.
"""

from pathlib import Path

import pandas as pd

from src.data.ea_md_qd import TARGET, load, release_lags

SERIES = Path("data/metadata/ea_md_qd_series.csv")
OUT = Path("data/metadata/data_dictionary.csv")
TR_TEXT = {1: "100 log x", 2: "100 dlog x", 3: "100 d2 log x", 4: "x (levels)", 5: "dx"}


def build():
    data, info = load()
    lags = release_lags(data, info)
    meta = pd.read_csv(SERIES).set_index("series_id")
    rows = []
    for name in data.columns:
        s = data[name]
        quarterly = info.loc[name, "Frequency"] == "Q"
        obs = s[s.index.month % 3 == 0] if quarterly else s
        first, last = obs.first_valid_index(), obs.last_valid_index()
        rule = (
            f"D-15: {lags[name]} months after the quarter"
            if quarterly
            else "D-14: 09.2026 ragged edge"
        )
        rows.append(
            {
                "series_id": name,
                "description": meta.loc[name, "description"],
                "group": meta.loc[name, "group"],
                "unit": meta.loc[name, "unit"],
                "frequency": info.loc[name, "Frequency"],
                "source": info.loc[name, "Source"],
                "geography": "euro area, changing composition"
                if name.endswith("_EACC")
                else "euro area",
                "seasonal_adjustment": info.loc[name, "SA"],
                "transformation_code": int(info.loc[name, "TR3"]),
                "transformation": TR_TEXT[int(info.loc[name, "TR3"])],
                "first_obs": str(first),
                "last_obs": str(last),
                "missing_inside": int(obs.loc[first:last].isna().sum()),
                "release_lag_months": int(lags[name]),
                "release_lag_rule": rule,
                "vintage": "EA-MD-QD 09.2026, single vintage (pseudo-real-time)",
                "role": "target and factor panel (D-09, D-21)"
                if name == TARGET
                else "factor panel",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build()
    table.to_csv(OUT, index=False)
    print(f"{len(table)} series written to {OUT}")
