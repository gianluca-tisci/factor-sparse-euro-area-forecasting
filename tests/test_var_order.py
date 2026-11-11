"""The order-p EM used for the appendix reproduces the forecasting estimator at p = 2."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.initialization import BalancedStartEM
from src.forecasting.var_order import OrderPEM, companion_radius


def panel():
    rng = np.random.default_rng(71)
    dates = pd.period_range("2000-01", periods=72, freq="M")
    x = pd.DataFrame(
        rng.normal(size=(72, 8)), index=dates, columns=[f"m{i}" for i in range(6)] + ["q0", "q1"]
    )
    x.loc[dates.month % 3 != 0, ["q0", "q1"]] = np.nan
    x.iloc[-2:, 0] = np.nan
    return x


def test_order_two_reproduces_the_forecasting_estimator():
    x = panel()
    reference = BalancedStartEM(x, 6, r=2).fit(maxiter=25, tolerance=0)
    ours = OrderPEM(x, 6, r=2, p=2).fit(maxiter=25, tolerance=0)
    np.testing.assert_allclose(ours.posterior.llf, reference.posterior.llf, rtol=1e-10)
    np.testing.assert_allclose(ours.params.ar, reference.params.ar, atol=1e-9)
    np.testing.assert_allclose(ours.params.loadings, reference.params.loadings, atol=1e-9)
    np.testing.assert_allclose(ours.params.innovation, reference.params.innovation, atol=1e-9)


@pytest.mark.parametrize("p", [1, 3])
def test_other_orders_have_the_right_shape_and_a_monotone_likelihood(p):
    fitted = OrderPEM(panel(), 6, r=2, p=p).fit(maxiter=15, tolerance=0)
    assert fitted.params.ar.shape == (2, 2 * p)
    assert np.min(np.diff(fitted.likelihood)) >= -1e-7


def test_companion_radius_matches_a_scalar_ar2():
    # y_t = 0.5 y_{t-1} + 0.3 y_{t-2}: roots of z^2 - 0.5 z - 0.3
    assert companion_radius(np.array([[0.5, 0.3]]), 1) == pytest.approx(
        max(abs(np.roots([1, -0.5, -0.3])))
    )
    assert companion_radius(np.array([[0.7]]), 1) == pytest.approx(0.7)
