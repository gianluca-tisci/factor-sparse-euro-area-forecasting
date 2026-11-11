"""Balanced-end PCA benchmark; D-65's aggregation, branch VAR rule unchanged."""

import numpy as np
import pandas as pd


def balanced_pca(panel, r, delay):
    """Fit up to origin-delay; ALS fills only internal observed-sample gaps.

    ALS equations follow main's src/factors/pca.py at 9d01c95. The EM and
    PCR initializers deliberately continue using their original pca_fit.
    """
    end = panel.index[-1] - int(delay)
    z = panel.loc[:end]
    x = z.to_numpy(dtype=float)
    observed = np.isfinite(x)
    x0 = np.where(observed, x, 0.0)
    loading = np.linalg.svd(x0, full_matrices=False)[2][:r].T.copy()
    previous = None
    for iteration in range(1, 5001):
        factors = x0 @ loading @ np.linalg.inv(loading.T @ loading)
        for i in np.flatnonzero(~observed.all(axis=1)):
            o = observed[i]
            factors[i] = np.linalg.solve(loading[o].T @ loading[o], loading[o].T @ x[i, o])
        loading = x0.T @ factors @ np.linalg.inv(factors.T @ factors)
        for j in np.flatnonzero(~observed.all(axis=0)):
            o = observed[:, j]
            loading[j] = np.linalg.solve(factors[o].T @ factors[o], factors[o].T @ x[o, j])
        common = factors @ loading.T
        if observed.all() or (
            previous is not None and np.abs(common[~observed] - previous[~observed]).max() < 1e-10
        ):
            break
        previous = common
    else:
        raise RuntimeError("Balanced PCA did not converge")
    u, s, vt = np.linalg.svd(common, full_matrices=False)
    return u[:, :r] * s[:r], vt[:r].T, z.index, iteration


def carried_common(factors, loading, fitted_var, factor_index, full_index):
    """Extend fitted common components using only the branch's fitted VAR."""
    steps = (full_index[-1] - factor_index[-1]).n
    if steps:
        if fitted_var.k_ar:
            ahead = fitted_var.forecast(factors[-fitted_var.k_ar :], steps)
        else:
            ahead = np.tile(fitted_var.intercept, (steps, 1))
        factors = np.vstack([factors, ahead])
    return pd.DataFrame(factors @ loading.T, index=full_index)
