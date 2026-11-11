"""Exact Gaussian EM with quarterly WN integrated out (D-59 correction).

The model is unchanged: one joint factor VAR(2), monthly white noise and
Mariano--Murasawa quarterly observations with weights (1, 2, 3, 2, 1).
Quarterly errors have covariance sigma_Q^2 K, NOT independent quarterly WN.
K has diagonal 19 and off-diagonal 4 for adjacent observed quarters.

E-step: Gaussian information-form smoother of factors, algebraically equivalent
to a Kalman smoother. Keeping its full covariance permits the exact GLS M-step
for quarterly loadings. Initial five factor vectors have fixed N(0, 10 I)
distribution; quarterly presample WN has its stationary variance. Neither
initialization nor any preprocessing uses data beyond the current origin.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, kron

WEIGHTS = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
FLOOR = 1e-8


def positive(matrix):
    """Symmetrize and impose a documented numerical variance lower bound."""
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(values, FLOOR)) @ vectors.T


def radius(ar):
    r = ar.shape[0]
    companion = np.block([[ar], [np.eye(r), np.zeros((r, r))]])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def pca_fit(panel, r=3):
    """PCA on complete training rows; ragged rows project available series only.

    All monthly series remain in the PCA loading matrix. Calendar gaps are not
    collapsed: each incomplete row receives a least-squares factor estimate
    from that row's observed series. No invented observation enters the PCA fit.
    """
    x = np.asarray(panel, dtype=float)
    complete = x[np.isfinite(x).all(axis=1)]
    if len(complete) < max(24, r + 2):
        raise ValueError("Insufficient complete monthly rows for PCA")
    _, s, v = np.linalg.svd(complete, full_matrices=False)
    loadings = v[:r].T * s[:r] / np.sqrt(len(complete))
    factors = np.zeros((len(x), r))
    for t, row in enumerate(x):
        observed = np.isfinite(row)
        factors[t], _, rank, _ = np.linalg.lstsq(loadings[observed], row[observed], rcond=None)
        if rank != r:
            raise ValueError(f"Monthly PCA projection lacks rank at row {t}")
    return factors, loadings, 1


@dataclass
class Parameters:
    loadings: np.ndarray
    variances: np.ndarray
    ar: np.ndarray
    innovation: np.ndarray


@dataclass
class Posterior:
    mean: np.ndarray
    covariance: np.ndarray
    llf: float


@dataclass
class Fit:
    model: object
    params: Parameters
    posterior: Posterior
    iterations: int
    converged: bool
    likelihood: list

    def common(self):
        model = self.model
        f = self.posterior.mean.reshape(-1, model.r)
        result = pd.DataFrame(index=model.panel.index, columns=model.panel.columns, dtype=float)
        result.iloc[:, : model.nm] = f[4:] @ self.params.loadings[: model.nm].T
        quarterly_times = np.flatnonzero(model.panel.index.month % 3 == 0)
        for j in model.quarterly:
            for t in quarterly_times:
                g = WEIGHTS @ f[t + 4 - np.arange(5)]
                result.iloc[t, j] = g @ self.params.loadings[j]
        return result

    def common_at(self, date, name):
        """Smoother estimate for a nowcast; VAR forecast only beyond origin."""
        date = pd.Period(date, "M")
        j = self.model.panel.columns.get_loc(name)
        if j >= self.model.nm:
            raise ValueError("Only monthly targets are forecast")
        f = self.posterior.mean.reshape(-1, self.model.r)
        if date <= self.model.panel.index[-1]:
            return float(f[self.model.panel.index.get_loc(date) + 4] @ self.params.loadings[j])
        history = list(f)
        for _ in range((date - self.model.panel.index[-1]).n):
            history.append(self.params.ar @ np.concatenate([history[-1], history[-2]]))
        return float(history[-1] @ self.params.loadings[j])


class MixedFrequencyEM:
    """Exact EM using factors as complete data, integrating quarterly errors."""

    def __init__(self, panel, n_monthly, r=3, initial_variance=10.0):
        self.panel = panel
        self.y = np.asarray(panel, dtype=float)
        self.nm, self.r, self.t = n_monthly, r, len(panel)
        self.nf = (self.t + 4) * r
        self.initial_variance = initial_variance
        self.quarterly = {}
        for j in range(n_monthly, panel.shape[1]):
            times = np.flatnonzero(np.isfinite(self.y[:, j]))
            if len(times) < 2:
                raise ValueError(f"Insufficient quarterly observations: {panel.columns[j]}")
            w = np.zeros((len(times), self.t + 4))
            for row, time in enumerate(times):
                w[row, time + 4 - np.arange(5)] = WEIGHTS
            kernel = w @ w.T
            cf = cho_factor(kernel, lower=True)
            precision = cho_solve(cf, np.eye(len(times)))
            self.quarterly[j] = times, csr_matrix(w), (kernel, precision, cf)

    def start(self):
        f, lm, _ = pca_fit(self.y[:, : self.nm], self.r)
        x = np.hstack([f[1:-1], f[:-2]])
        ar = np.linalg.lstsq(x, f[2:], rcond=None)[0].T
        # Repair the START only. Estimated transitions are never rescaled.
        if radius(ar) >= 1:
            ar = np.hstack([0.5 * np.eye(self.r), np.zeros((self.r, self.r))])
        errors = f[2:] - x @ ar.T
        innovation = positive(errors.T @ errors / len(errors))
        loadings = np.zeros((self.y.shape[1], self.r))
        loadings[: self.nm] = lm
        variances = np.nanmean((self.y[:, : self.nm] - f @ lm.T) ** 2, axis=0).tolist()
        full_f = np.vstack([np.zeros((4, self.r)), f])
        for j, (times, w, (_, precision, _)) in self.quarterly.items():
            g = w @ full_f
            loading = np.linalg.solve(g.T @ precision @ g, g.T @ precision @ self.y[times, j])
            loadings[j] = loading
            error = self.y[times, j] - g @ loading
            variances.append(float(error @ precision @ error / len(times)))
        return Parameters(loadings, np.maximum(variances, FLOOR), ar, innovation)

    def expectation(self, params):
        """Exact posterior and marginal likelihood; no fitted future states."""
        r, n = self.r, self.nf
        # Prior on f_0,...,f_-4, then VAR transitions for f_1,...,f_(T-1).
        prior = np.zeros((n, n))
        prior[: 5 * r, : 5 * r] = np.eye(5 * r) / self.initial_variance
        qi = np.linalg.inv(params.innovation)
        transition = np.hstack([np.eye(r), -params.ar])
        update = transition.T @ qi @ transition
        for t in range(1, self.t):
            ix = np.concatenate(
                [
                    np.arange((t + 4) * r, (t + 5) * r),
                    np.arange((t + 3) * r, (t + 4) * r),
                    np.arange((t + 2) * r, (t + 3) * r),
                ]
            )
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
            # Each observation loads on weighted monthly factors.
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
        """GLS quarterly M-step includes ALL cross-quarter posterior moments."""
        r = self.r
        mean = post.mean.reshape(-1, r)
        moments = post.covariance + np.outer(post.mean, post.mean)
        loadings = np.zeros((self.y.shape[1], r))
        variances = np.zeros(self.y.shape[1])
        second = np.stack(
            [moments[(t + 4) * r : (t + 5) * r, (t + 4) * r : (t + 5) * r] for t in range(self.t)]
        )
        for j in range(self.nm):
            times = np.flatnonzero(np.isfinite(self.y[:, j]))
            y = self.y[times, j]
            a = second[times].sum(axis=0)
            b = mean[times + 4].T @ y
            loadings[j] = np.linalg.solve(a, b)
            variances[j] = (y @ y - 2 * loadings[j] @ b + loadings[j] @ a @ loadings[j]) / len(y)
        for j, (times, w, (_, ki, _)) in self.quarterly.items():
            g = w @ mean
            operator = kron(w, np.eye(r), format="csr")
            covariance = np.asarray(operator @ (operator @ post.covariance).T)
            covariance = covariance.reshape(len(times), r, len(times), r)
            a = g.T @ ki @ g + np.einsum("ts,tasb->ab", ki, covariance)
            b = g.T @ ki @ self.y[times, j]
            loadings[j] = np.linalg.solve(a, b)
            error = self.y[times, j] - g @ loadings[j]
            uncertainty = np.einsum("a,tasb,b->ts", loadings[j], covariance, loadings[j])
            variances[j] = (error @ ki @ error + np.sum(ki * uncertainty)) / len(times)
        a, b, c = np.zeros((2 * r, 2 * r)), np.zeros((r, 2 * r)), np.zeros((r, r))
        for t in range(1, self.t):
            iy = np.arange((t + 4) * r, (t + 5) * r)
            ix = np.r_[np.arange((t + 3) * r, (t + 4) * r), np.arange((t + 2) * r, (t + 3) * r)]
            a += moments[np.ix_(ix, ix)]
            b += moments[np.ix_(iy, ix)]
            c += moments[np.ix_(iy, iy)]
        ar = np.linalg.solve(a, b.T).T
        innovation = positive((c - ar @ b.T) / (self.t - 1))
        return Parameters(loadings, np.maximum(variances, FLOOR), ar, innovation)

    def fit(self, start=None, maxiter=500, tolerance=1e-6):
        params = start if start is not None else self.start()
        post = self.expectation(params)
        history = [post.llf]
        for iteration in range(1, maxiter + 1):
            candidate = self.maximization(post)
            next_post = self.expectation(candidate)
            change = next_post.llf - post.llf
            if change < -1e-7 * (1 + abs(post.llf)):
                raise RuntimeError(f"Exact EM likelihood decreased: {change}")
            # Only accept a finite, positive-covariance candidate. Never return a rejected pair.
            if not np.isfinite(next_post.llf):
                raise RuntimeError("Nonfinite EM likelihood")
            params, post = candidate, next_post
            history.append(post.llf)
            relative = 2 * abs(change) / (abs(history[-1]) + abs(history[-2]) + 1e-12)
            if relative < tolerance:
                return Fit(self, params, post, iteration, True, history)
        return Fit(self, params, post, iteration, False, history)
