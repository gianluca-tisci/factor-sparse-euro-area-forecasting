"""Cross-sectional dependence left in a panel: summary statistics and the FMM max test.

Used to check the premise of the sparse step (Fan, Masini & Medeiros 2023): removing the
common factors should leave idiosyncratic components that are far less correlated than
the series themselves. Both functions take a T x n frame and use its complete rows.
"""

import numpy as np


def standardized(frame):
    """Complete rows, each column centered and scaled to unit variance."""
    u = frame.dropna().to_numpy(dtype=float)
    return (u - u.mean(0)) / u.std(0)


def correlation_summary(frame, threshold=0.5):
    """Average absolute correlation, strong pairs and leading principal components.

    The noise edge is the Marchenko-Pastur upper bound (1 + sqrt(n/T))^2 for the largest
    eigenvalue of the correlation matrix of n independent series observed T times.
    """
    u = standardized(frame)
    t, n = u.shape
    corr = u.T @ u / t
    off = np.abs(corr[np.triu_indices(n, k=1)])
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    edge = (1 + np.sqrt(n / t)) ** 2
    return {
        "T": t,
        "n": n,
        "mean_abs_corr": float(off.mean()),
        "strong_pairs": int((off > threshold).sum()),
        "pairs": int(off.size),
        "share_pc1": float(eig[0] / n),
        "share_pc3": float(eig[:3].sum() / n),
        "above_noise": int((eig > edge).sum()),
    }


def fmm_diagonal_test(frame, rng, draws=5000):
    """Max-type test of H0: all off-diagonal covariances are zero (FMM 2023, eqs. 2.9-2.12).

    S = sqrt(T) max_{i<j} |sigma_ij| on standardized series, so sigma_ij is a correlation
    (our deviation: FMM use raw covariances, whose maximum favors high-variance series).
    Critical values: Gaussian multiplier bootstrap with the Bartlett kernel and bandwidth
    floor(T/3), FMM's practice (Supplement S.2). The products are centered at their sample
    means, as in FMM's long-run covariance.
    """
    u = standardized(frame)
    t, n = u.shape
    i, j = np.triu_indices(n, k=1)
    products = u[:, i] * u[:, j]
    sigma = products.mean(0)
    statistic = np.sqrt(t) * np.abs(sigma).max()
    h = t // 3
    distance = np.abs(np.subtract.outer(np.arange(t), np.arange(t))) / h
    kernel = np.clip(1 - distance, 0, None)
    values, vectors = np.linalg.eigh(kernel)
    root = vectors * np.sqrt(np.clip(values, 0, None))
    multipliers = rng.standard_normal((draws, t)) @ root.T
    maxima = np.abs(multipliers @ (products - sigma) / np.sqrt(t)).max(1)
    top = int(np.abs(sigma).argmax())
    return {
        "statistic": float(statistic),
        "critical_05": float(np.quantile(maxima, 0.95)),
        "p_value": float((maxima >= statistic).mean()),
        "bandwidth": h,
        "top_pair": (int(i[top]), int(j[top])),
        "top_corr": float(sigma[top]),
    }
