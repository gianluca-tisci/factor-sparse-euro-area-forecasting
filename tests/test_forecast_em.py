"""Independent Kalman equivalence and exact EM checks for the D-59 correction."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.forecasting.dfm import MixedFrequencyEM, Parameters, radius


def synthetic():
    rng = np.random.default_rng(2026)
    dates = pd.period_range("2000-01", periods=36, freq="M")
    x = pd.DataFrame(
        rng.normal(size=(36, 8)), index=dates, columns=[f"m{i}" for i in range(6)] + ["q0", "q1"]
    )
    x.loc[dates.month % 3 != 0, ["q0", "q1"]] = np.nan
    x.iloc[-2:, 0] = np.nan
    x.iloc[14, 6] = np.nan
    return MixedFrequencyEM(x, 6, r=2)


def kalman(model, p):
    """Independent state-space representation with EXACTLY the same initial law."""
    sm = DynamicFactorMQ(
        model.panel,
        k_endog_monthly=model.nm,
        factors=1,
        factor_multiplicities=model.r,
        factor_orders=2,
        idiosyncratic_ar1=False,
        standardize=False,
    )
    params = np.zeros(sm.k_params)
    params[sm._p["loadings"]] = p.loadings.ravel()
    params[sm._p["factor_ar"]] = p.ar.ravel()
    params[sm._p["factor_cov"]] = np.linalg.cholesky(p.innovation)[np.tril_indices(model.r)]
    params[sm._p["idiosyncratic_var"]] = p.variances
    initial = np.zeros(sm.k_states)
    cov = np.eye(sm.k_states) * model.initial_variance
    for j, ix in enumerate(sm._s["idio_ar_Q_ix"]):
        cov[ix, ix] = p.variances[model.nm + j]
    sm.initialize_known(initial, cov)
    return sm.smooth(params, cov_type="none")


def test_dense_information_smoother_matches_independent_kalman():
    model = synthetic()
    params = model.start()
    dense = model.expectation(params)
    result = kalman(model, params)
    np.testing.assert_allclose(dense.llf, result.llf, atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        dense.mean.reshape(-1, 2)[4:].T, result.smoothed_state[:2], atol=1e-8, rtol=0
    )
    for t in [0, 12, 35]:
        sl = slice((t + 4) * 2, (t + 5) * 2)
        np.testing.assert_allclose(
            dense.covariance[sl, sl], result.smoothed_state_cov[:2, :2, t], atol=1e-8
        )
    # Cross-date covariance drives the corrected quarterly GLS update.
    cross = result.smoother_results.smoothed_state_autocovariance(lag=6, t=11)
    np.testing.assert_allclose(dense.covariance[30:32, 18:20], cross[:2, :2], atol=1e-8)


def test_gls_quarterly_m_step_solves_conditional_normal_equation():
    model = synthetic()
    post = model.expectation(model.start())
    updated = model.maximization(post)
    # Independently integrate the complete-data Q function over the posterior.
    j = 6
    times, w, (_, precision, _) = model.quarterly[j]
    r = model.r
    g = w @ post.mean.reshape(-1, r)
    from scipy.sparse import kron

    operator = kron(w, np.eye(r)).toarray()
    cov = (operator @ post.covariance @ operator.T).reshape(len(times), r, len(times), r)
    a = g.T @ precision @ g + np.einsum("ts,tasb->ab", precision, cov)
    b = g.T @ precision @ model.y[times, j]
    np.testing.assert_allclose(a @ updated.loadings[j] - b, 0, atol=1e-10)
    assert np.linalg.eigvalsh(post.covariance).min() > 0


def test_corrected_em_is_monotone_and_reproducible():
    model = synthetic()
    fitted = model.fit(maxiter=15, tolerance=0)
    assert np.min(np.diff(fitted.likelihood)) >= -1e-7
    again = model.fit(maxiter=15, tolerance=0)
    np.testing.assert_allclose(fitted.params.loadings, again.params.loadings, atol=1e-12)
    np.testing.assert_allclose(fitted.posterior.llf, kalman(model, fitted.params).llf, atol=1e-8)


def test_estimated_transition_is_not_rescaled():
    model = synthetic()
    params = model.start()
    params = Parameters(
        params.loadings,
        params.variances,
        np.hstack([1.1 * np.eye(2), np.zeros((2, 2))]),
        params.innovation,
    )
    assert radius(params.ar) == pytest.approx(1.1)
    assert np.isfinite(model.expectation(params).llf)
