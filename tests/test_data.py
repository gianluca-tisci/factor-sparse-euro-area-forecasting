"""Tests for the data step: alignment, release availability, transformations, dictionary."""

import numpy as np
import pandas as pd
import pytest

from src.data import ea_md_qd
from src.data.dictionary import build
from src.preprocessing.transform import transform


@pytest.fixture(scope="module")
def panel():
    data, info = ea_md_qd.load()  # also checks the file against its manifest hash
    return data, info, ea_md_qd.release_lags(data, info)


def last(frame, name):
    return frame[name].last_valid_index()


def test_contract_and_alignment(panel):
    data, info, _ = panel
    assert data.shape[1] == 118 and data.index[0] == pd.Period("2000-01", "M")
    assert data.index.is_unique and data.index.is_monotonic_increasing
    quarterly = info.index[info.Frequency == "Q"]
    assert data[quarterly].notna().loc[data.index.month % 3 != 0].sum().sum() == 0


def test_release_lags_follow_d14_d15(panel):
    _, _, lags = panel
    expected = {
        "HICPNG_EA": 1,  # Eurostat flash about 1 day after the month
        "PPINRG_EA": 2,  # producer prices about 35 days after the month
        "ESENTIX_EA": 0,  # surveys in the same month
        "GDP_EA": 2,
        "TASS.SDB_EA": 4,
        "GHSR_EA": 4,
    }
    assert {k: lags[k] for k in expected} == expected


def test_available_respects_publication_dates(panel):
    data, _, lags = panel
    at = ea_md_qd.available(data, lags, "2026-09")
    assert last(at, "HICPNG_EA") == pd.Period("2026-08", "M")
    assert last(at, "PPINRG_EA") == pd.Period("2026-07", "M")
    assert last(ea_md_qd.available(data, lags, "2026-08"), "GDP_EA") == pd.Period("2026-06", "M")
    assert last(ea_md_qd.available(data, lags, "2026-07"), "GDP_EA") == pd.Period("2026-03", "M")
    for name in data.columns:
        cutoff = pd.Period("2026-09", "M") - int(lags[name])
        assert at.loc[at.index > cutoff, name].isna().all()


def test_leakage_sentinel_at_the_data_level(panel):
    data, _, lags = panel
    origin = pd.Period("2015-06", "M")
    perturbed = data.copy()
    perturbed.loc[perturbed.index > origin] += 1000.0
    pd.testing.assert_frame_equal(
        ea_md_qd.available(data, lags, origin), ea_md_qd.available(perturbed, lags, origin)
    )


def test_late_start_series_are_not_filled(panel):
    data, info, _ = panel
    x = transform(data, info)
    start = data["ULCIN_EA"].first_valid_index()  # unit labor costs start in 2001
    assert start == pd.Period("2001-03", "M")
    assert x.loc[:start, "ULCIN_EA"].isna().all()
    assert x["ULCIN_EA"].first_valid_index() == pd.Period("2001-06", "M")


def test_transformations(panel):
    data, info, _ = panel
    x = transform(data, info)
    energy = 100 * np.log(data["HICPNG_EA"]).diff()
    pd.testing.assert_series_equal(x["HICPNG_EA"], energy, check_names=False)
    gdp = data["GDP_EA"]
    q = gdp[gdp.index.month % 3 == 0]
    assert x.loc["2010-06", "GDP_EA"] == pytest.approx(100 * np.log(q["2010-06"] / q["2010-03"]))
    assert info.loc["UNETOT_EA", "TR3"] == 5  # unemployment differenced under TR3 (D-16)
    assert x.loc["2010-02", "UNETOT_EA"] == pytest.approx(
        data.loc["2010-02", "UNETOT_EA"] - data.loc["2010-01", "UNETOT_EA"]
    )


def test_data_dictionary_has_protocol_fields():
    table = build()
    assert len(table) == 118 and table.series_id.is_unique
    required = ["description", "unit", "frequency", "transformation", "release_lag_months", "role"]
    assert table[required].notna().all().all()
    assert (table.role.str.startswith("target")).sum() == 1
