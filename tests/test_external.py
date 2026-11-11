import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

from src.data.external import IDS, ROOT, load_external


def test_external_snapshot():
    panel, manifest = load_external()
    assert tuple(panel.columns) == IDS
    assert len(IDS) == len(set(IDS)) == 24
    assert panel.loc["2000-02":"2025-12"].notna().all().all()
    for name, meta in manifest["series"].items():
        source = pd.read_csv(ROOT / meta["file"], index_col=0)[name]
        source.index = pd.PeriodIndex(source.index, freq="M")
        expected = 100 * np.log(source).diff() if meta["transformation"] == "100*dlog" else source
        pd.testing.assert_series_equal(panel[name].reindex(source.index), expected)
        assert (
            hashlib.sha256((ROOT / meta["file"]).read_bytes()).hexdigest()
            == meta["sha256_processed"]
        )
    assert all(manifest["series"][name]["release_lag_months"] == 2 for name in IDS[11:])


def test_external_tampering_fails(tmp_path):
    shutil.copytree(ROOT, tmp_path / "snapshot")
    path = tmp_path / "snapshot" / f"{IDS[0]}.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_external(tmp_path / "snapshot")
