"""The FMM max test keeps its size on independent noise and finds one strong pair."""

import numpy as np
import pandas as pd

from src.factors.residual_diagnostics import correlation_summary, fmm_diagonal_test


def test_independent_noise_is_not_rejected_and_has_no_strong_components():
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.standard_normal((200, 20)))
    result = fmm_diagonal_test(noise, np.random.default_rng(2), draws=2000)
    assert result["p_value"] > 0.05
    summary = correlation_summary(noise)
    assert summary["above_noise"] == 0
    assert summary["mean_abs_corr"] < 0.1


def test_one_correlated_pair_is_rejected_and_reported():
    rng = np.random.default_rng(3)
    panel = rng.standard_normal((200, 20))
    panel[:, 7] = panel[:, 3] + 0.3 * rng.standard_normal(200)
    result = fmm_diagonal_test(pd.DataFrame(panel), np.random.default_rng(4), draws=2000)
    assert result["p_value"] < 0.01
    assert result["top_pair"] == (3, 7)


def test_size_on_repeated_noise_panels_is_close_to_nominal():
    rejections = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        noise = pd.DataFrame(rng.standard_normal((150, 10)))
        rejections += fmm_diagonal_test(noise, rng, draws=1000)["p_value"] < 0.05
    assert rejections <= 6  # 5% of 40 is 2; allow sampling error
