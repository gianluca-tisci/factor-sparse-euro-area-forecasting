"""Reference dates, publication dates and common one-step feature dictionaries."""

import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path


def target_month(origin, lag):
    """D-37: first unpublished month, including nowcasts for delayed targets."""
    return pd.Period(origin, "M") - int(lag) + 1


def dictionary(panel, frequencies, lags, origins, monthly_positions=4, quarterly_positions=2):
    """Each feature uses a calendar reference period released by its row origin.

    Monthly position 1 means latest released month; quarterly position 1 means
    latest released quarter. Missing reference periods are never skipped.
    D-41: the supplied residual panel is rebuilt at each OUTER fit cutoff.
    """
    origins = pd.PeriodIndex(origins, freq="M")
    result, references = {}, {}
    for name in panel:
        quarterly = frequencies[name] == "Q"
        delay = int(lags[name])
        latest = origins - delay
        if quarterly:
            latest = latest - (latest.month % 3)
        positions = quarterly_positions if quarterly else monthly_positions
        for position in range(positions):
            dates = latest - (3 * position if quarterly else position)
            if (dates + delay > origins).any():
                raise AssertionError("Unreleased sparse predictor")
            key = f"{name}__{'Q' if quarterly else 'M'}{position + 1}"
            result[key] = panel[name].reindex(dates).to_numpy()
            references[key] = dates
    return pd.DataFrame(result, index=origins), pd.DataFrame(references, index=origins)


def training_sample(features, response, target_lag, cutoff):
    """Train labels must be published at the OUTER cutoff; not necessarily at row origin."""
    dates = features.index - int(target_lag) + 1
    labels = response.reindex(dates).to_numpy()
    published = dates + int(target_lag) <= cutoff
    valid = published & np.isfinite(labels)
    return features.loc[valid], labels[valid]


def complete_design(features, response, forecast_row):
    """Explicit missing policy, shared across models; no fabricated residuals."""
    columns = forecast_row.index[np.isfinite(forecast_row.to_numpy(dtype=float))]
    x = features[columns]
    rows = np.isfinite(x.to_numpy()).all(axis=1) & np.isfinite(response)
    x, y = x.loc[rows], np.asarray(response)[rows]
    if len(y) < 24:
        raise ValueError(f"Only {len(y)} complete published training labels")
    if len(columns) == 0:
        raise ValueError("No available predictors")
    return x, y, forecast_row[columns]


def sparse_predict(features, response, forecast_row):
    """Modified BIC along the exact LASSO/LARS path, including the null model.

    BIC* = log(RSS/n) + |S| log(n)/n log(log(d_W)); support <= floor(n/10).
    Coefficients are returned in supplied feature/response units, with no
    post-LASSO refit. Centering/scaling uses only complete training rows.
    """
    x, y, forecast_row = complete_design(features, response, forecast_row)
    means, scales = x.mean(), x.std(ddof=0)
    keep = scales > 1e-12
    x, means, scales = x.loc[:, keep], means[keep], scales[keep]
    z = ((x - means) / scales).to_numpy()
    centered = y - y.mean()
    d, n = x.shape[1], len(y)
    if d == 0:
        return float(y.mean()), pd.Series(dtype=float), {"n_train": n, "d": 0, "alpha": 0.0}
    cap = n // 10
    # D-36 caps the PATH at n/10 selected predictors. Do not calculate the
    # high-dimensional near-OLS tail, which is outside the candidate menu.
    # LASSO drop events can reduce support, so extend until the cap is reached.
    steps = cap
    while True:
        alphas, _, coefs = lars_path(
            z,
            centered,
            Gram=z.T @ z,
            Xy=z.T @ centered,
            method="lasso",
            max_iter=steps,
            return_path=True,
        )
        sizes = np.sum(np.abs(coefs) > 1e-10, axis=0)
        reached = np.flatnonzero(sizes >= cap)
        if len(reached):
            stop = reached[0] + 1
            alphas, coefs = alphas[:stop], coefs[:, :stop]
            break
        if alphas[-1] < 1e-12 or coefs.shape[1] < steps + 1:
            break
        steps += cap
        if steps > 5000:
            raise RuntimeError("LASSO path failed to reach its support cap or zero penalty")
    rss = np.sum((centered[:, None] - z @ coefs) ** 2, axis=0)
    support = np.sum(np.abs(coefs) > 1e-10, axis=0)
    c_t = np.log(np.log(max(d, 3)))
    bic = np.log(np.maximum(rss / n, 1e-30)) + support * np.log(n) / n * c_t
    bic[support > cap] = np.inf
    chosen = int(np.argmin(bic))
    coefficients = pd.Series(coefs[:, chosen] / scales.to_numpy(), index=x.columns)
    prediction = float(y.mean() + (forecast_row[x.columns] - means) @ coefficients)
    selected = coefficients[coefficients.abs() > 1e-10]
    return (
        prediction,
        selected,
        {
            "n_train": n,
            "d": d,
            "alpha": float(alphas[chosen]),
            "bic": float(bic[chosen]),
            "cap": cap,
            "selected": len(selected),
        },
    )


def bic_regression(features, response, forecast_row, groups):
    """OLS BIC selection on identical complete rows for every candidate lag order."""
    x, y, row = complete_design(features, response, forecast_row)
    values, future = x.to_numpy(), row.to_numpy()
    locations = {name: j for j, name in enumerate(x.columns)}
    best = None
    for columns in groups:
        if not set(columns).issubset(locations):
            continue
        indices = [locations[name] for name in columns]
        design = np.column_stack([np.ones(len(x)), values[:, indices]])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < design.shape[1] or len(y) <= rank + 2:
            continue
        rss = np.sum((y - design @ coefficients) ** 2)
        bic = np.log(max(rss / len(y), 1e-30)) + rank * np.log(len(y)) / len(y)
        prediction = np.r_[1.0, future[indices]] @ coefficients
        if best is None or bic < best[0]:
            best = bic, prediction, len(columns)
    if best is None:
        if forecast_row.isna().any() and not any(
            set(columns).issubset(locations) for columns in groups
        ):
            return float("nan"), {
                "n_train": len(y),
                "failure_reason": "Required own-lag predictor unavailable after cleaning",
            }
        raise ValueError("No estimable BIC regression")
    return float(best[1]), {"n_train": len(y), "regressors": best[2], "bic": float(best[0])}
