"""Exact EM with a factor VAR of order p, for appendix diagnostics only; never used for forecasts.

The forecasting estimator fixes the factor VAR at two lags (D-30, src/forecasting/dfm.py). This
subclass changes only the VAR order p (1 <= p <= 5, the factor lags the quarterly aggregation
already keeps in the state), so the appendix can report the fit and the largest VAR root for other
orders. With p = 2 it reproduces the forecasting estimator exactly (tests/test_var_order.py).
It is a separate file on purpose: editing dfm.py would change the source hashes of saved runs.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, kron

from src.forecasting.dfm import FLOOR, MixedFrequencyEM, Parameters, Posterior, positive
from src.forecasting.pca import balanced_pca

DELAY = 3  # maximum monthly publication delay (D-14); the balanced-end start of D-65 and D-67


def companion_radius(ar, r):
    """Largest root (spectral radius) of the VAR with coefficients [A_1 ... A_p]."""
    p = ar.shape[1] // r
    companion = ar
    if p > 1:
        shift = np.hstack([np.eye((p - 1) * r), np.zeros(((p - 1) * r, r))])
        companion = np.vstack([ar, shift])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def lag_blocks(t, r, lags):
    """State indices of the factors at t, t-1, ..., for the given lag numbers (state block j = f_{j-4})."""
    return np.concatenate([np.arange((t + 4 - k) * r, (t + 5 - k) * r) for k in lags])


class OrderPEM(MixedFrequencyEM):
    """Same likelihood, balanced-end PCA start and loading updates; only the VAR order differs."""

    def __init__(self, panel, n_monthly, r=3, initial_variance=10.0, p=2):
        if not 1 <= p <= 5:
            raise ValueError("The state keeps five factor lags: 1 <= p <= 5")
        super().__init__(panel, n_monthly, r, initial_variance)
        self.p = p

    def start(self):
        """balanced_start of src/forecasting/initialization.py with p lags in the VAR seed."""
        r, p = self.r, self.p
        f, lm, index, _ = balanced_pca(self.panel.iloc[:, : self.nm], r, DELAY)
        seed = MixedFrequencyEM(self.panel.loc[index], self.nm, r, self.initial_variance)
        x = np.hstack([f[p - k : len(f) - k] for k in range(1, p + 1)])
        y = f[p:]
        ar = np.linalg.lstsq(x, y, rcond=None)[0].T
        if companion_radius(ar, r) >= 1:
            ar = np.hstack([0.5 * np.eye(r), np.zeros((r, (p - 1) * r))])
        errors = y - x @ ar.T
        innovation = positive(errors.T @ errors / len(errors))
        loadings = np.zeros((seed.y.shape[1], r))
        loadings[: self.nm] = lm
        variances = np.nanmean((seed.y[:, : self.nm] - f @ lm.T) ** 2, axis=0).tolist()
        full_f = np.vstack([np.zeros((4, r)), f])
        for j, (times, w, (_, precision, _)) in seed.quarterly.items():
            g = w @ full_f
            loading = np.linalg.solve(g.T @ precision @ g, g.T @ precision @ seed.y[times, j])
            loadings[j] = loading
            error = seed.y[times, j] - g @ loading
            variances.append(float(error @ precision @ error / len(times)))
        return Parameters(loadings, np.maximum(variances, FLOOR), ar, innovation)

    def expectation(self, params):
        """dfm.MixedFrequencyEM.expectation with p lags in the factor transition."""
        r, n = self.r, self.nf
        prior = np.zeros((n, n))
        prior[: 5 * r, : 5 * r] = np.eye(5 * r) / self.initial_variance
        qi = np.linalg.inv(params.innovation)
        transition = np.hstack([np.eye(r), -params.ar])
        update = transition.T @ qi @ transition
        for t in range(1, self.t):
            ix = lag_blocks(t, r, range(self.p + 1))
            prior[np.ix_(ix, ix)] += update
        precision = prior
        rhs = np.zeros(n)
        quadratic, logdet_r, nobs = 0.0, 0.0, 0
        for t in range(self.t):
            mask = np.isfinite(self.y[t, : self.nm])
            loading = params.loadings[: self.nm][mask]
            v = params.variances[: self.nm][mask]
            y = self.y[t, : self.nm][mask]
            sl = slice((t + 4) * r, (t + 5) * r)
            precision[sl, sl] += (loading.T / v) @ loading
            rhs[sl] += loading.T @ (y / v)
            quadratic += np.sum(y * y / v)
            logdet_r += np.log(v).sum()
            nobs += len(y)
        for j, (times, w, (_, ki, cf)) in self.quarterly.items():
            loading = params.loadings[j]
            h = kron(w, csr_matrix(loading.reshape(1, -1)), format="csr")
            ri = ki / params.variances[j]
            y = self.y[times, j]
            precision += np.asarray(h.T @ (ri @ h.toarray()))
            rhs += np.asarray(h.T @ (ri @ y)).ravel()
            quadratic += y @ ri @ y
            logdet_r += len(y) * np.log(params.variances[j]) + 2 * np.log(np.diag(cf[0])).sum()
            nobs += len(y)
        cf = cho_factor(precision, lower=True, check_finite=False)
        mean = cho_solve(cf, rhs, check_finite=False)
        covariance = cho_solve(cf, np.eye(n), check_finite=False)
        logdet_prior_cov = 5 * r * np.log(self.initial_variance)
        logdet_prior_cov += (self.t - 1) * np.linalg.slogdet(params.innovation)[1]
        ll = -0.5 * (
            nobs * np.log(2 * np.pi)
            + logdet_r
            + logdet_prior_cov
            + 2 * np.log(np.diag(cf[0])).sum()
            + quadratic
            - rhs @ mean
        )
        return Posterior(mean, covariance, float(ll))

    def maximization(self, post):
        """Loadings and variances as in dfm.py (they do not depend on p); VAR step with p lags."""
        base = super().maximization(post)
        r, p = self.r, self.p
        moments = post.covariance + np.outer(post.mean, post.mean)
        a, b, c = np.zeros((p * r, p * r)), np.zeros((r, p * r)), np.zeros((r, r))
        for t in range(1, self.t):
            iy = lag_blocks(t, r, [0])
            ix = lag_blocks(t, r, range(1, p + 1))
            a += moments[np.ix_(ix, ix)]
            b += moments[np.ix_(iy, ix)]
            c += moments[np.ix_(iy, iy)]
        ar = np.linalg.solve(a, b.T).T
        innovation = positive((c - ar @ b.T) / (self.t - 1))
        return Parameters(base.loadings, base.variances, ar, innovation)
