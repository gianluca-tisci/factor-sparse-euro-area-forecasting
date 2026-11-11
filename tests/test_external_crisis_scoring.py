"""Crisis exclusions use target dates and preserve retained chronological rows."""

import numpy as np
import pandas as pd

from scripts.external_crisis_sensitivity import scoring_samples


def test_exclusion_boundaries_and_target_month_alignment():
    # These labels deliberately differ from the dates being excluded.
    months = pd.Series(
        ["2020-12", "2021-01", "2022-01", "2022-02", "2023-12", "2024-01"],
        index=["o0", "o1", "o2", "o3", "o4", "o5"],
    )
    masks = scoring_samples(
        months,
        {
            "broad": {"start": "2021-01", "end": "2023-12"},
            "narrow": {"start": "2022-02", "end": "2023-12"},
        },
    )
    assert months[masks["broad"]].index.tolist() == ["o0", "o5"]
    assert months[masks["narrow"]].index.tolist() == ["o0", "o1", "o2", "o5"]
    np.testing.assert_array_equal(masks["primary"], np.ones(6, dtype=bool))
