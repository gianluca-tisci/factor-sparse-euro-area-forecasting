"""Experimental balanced-end starting parameters; observed likelihood unchanged."""

import numpy as np

from src.forecasting.dfm import FLOOR, MixedFrequencyEM, Parameters, positive, radius
from src.forecasting.pca import balanced_pca


def balanced_start(model, delay=3):
    """Estimate starting parameters only on months through t-delay.

    Use the PCA benchmark's ALS for internal gaps, then the DFM's original
    no-intercept VAR(2) initialization equations. Do not treat projected
    factors as observations when estimating innovation variances. The full
    model's E-step subsequently uses ALL released data through t.
    """
    f, lm, index, _ = balanced_pca(model.panel.iloc[:, : model.nm], model.r, delay)
    seed = MixedFrequencyEM(model.panel.loc[index], model.nm, model.r, model.initial_variance)
    x = np.hstack([f[1:-1], f[:-2]])
    ar = np.linalg.lstsq(x, f[2:], rcond=None)[0].T
    if radius(ar) >= 1:
        ar = np.hstack([0.5 * np.eye(model.r), np.zeros((model.r, model.r))])
    errors = f[2:] - x @ ar.T
    innovation = positive(errors.T @ errors / len(errors))
    loadings = np.zeros((seed.y.shape[1], model.r))
    loadings[: model.nm] = lm
    variances = np.nanmean((seed.y[:, : model.nm] - f @ lm.T) ** 2, axis=0).tolist()
    full_f = np.vstack([np.zeros((4, model.r)), f])
    for j, (times, w, (_, precision, _)) in seed.quarterly.items():
        g = w @ full_f
        loading = np.linalg.solve(g.T @ precision @ g, g.T @ precision @ seed.y[times, j])
        loadings[j] = loading
        error = seed.y[times, j] - g @ loading
        variances.append(float(error @ precision @ error / len(times)))
    return Parameters(loadings, np.maximum(variances, FLOOR), ar, innovation)


class BalancedStartEM(MixedFrequencyEM):
    """Same estimator; only its cold-start parameter construction changes."""

    def start(self):
        return balanced_start(self)
