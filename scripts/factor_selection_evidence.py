"""Evidence for the number of factors and the factor-VAR order, and EM fit diagnostics.

Run from the repo root:  uv run python -m scripts.factor_selection_evidence
Writes tab_80_factor_number, tab_81_var_order, tab_82_em_diagnostics, tab_83_var_order_fits,
fig_80_scree, fig_81_abc, and the macros the methodology section quotes.

Panel for the criteria: the 47 monthly series at the first forecast date, with the run's own release
mask, outlier cleaning and standardization (src/forecasting/experiment.py, D-18, D-19). Principal
components need a complete panel, so the criteria use the months through the last one published for
every series (origin minus three) and fill the few outlier cells by the benchmark's ALS (D-65).
The quarterly subset is not part of this panel (D-20 asks for it as a further check).
These are our implementations of the published criteria, not the authors' code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.api import VAR

from src._style import PALETTE, SIZES, record_number, save_fig, save_table
from src.forecasting.experiment import clean_window, prepare
from src.forecasting.initialization import BalancedStartEM
from src.forecasting.pca import balanced_pca
from src.forecasting.var_order import OrderPEM, companion_radius

PROFILE = "config/forecast_paper.yaml"
KMAX = 15  # D-20: at most 15 factors, as in the EA-MD-QD paper
MAX_VAR_LAG = 6
ABC_DRAWS = 200  # D-20
ABC_SHARE = 0.75  # D-20: smallest subsample is 3/4 of the series
ABC_GRID = np.arange(0.0, 3.0001, 0.01)
ABC_MIN_WIDTH = 0.05  # a stable value of r must hold over a c-interval at least this wide
RUN = "outputs/forecasts/paper_experiment_nonzero_iqr"  # D-71: every date started cold
DESIGN = yaml.safe_load(Path("config/research_design.yaml").read_text())
MONITOR = DESIGN["factor_model"]["fit_monitoring"]


def first_window(origin=None):
    """Standardized monthly panel known at the first forecast date, and its EM model."""
    design, settings, _, panel, info, lags, origins = prepare(PROFILE, "M")
    origin = origins[0] if origin is None else pd.Period(origin, "M")
    _, _, z, _, _, _ = clean_window(panel, lags, origin)
    r = design["factor_model"]["selected_r"]
    nm = int((info.Frequency == "M").sum())
    delay = int(lags.iloc[:nm].max())
    factors, loading, index, _ = balanced_pca(z, r, delay)
    x = z.loc[index].to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, factors @ loading.T)
    model = BalancedStartEM(z, nm, r, initial_variance=settings["initial_factor_variance"])
    return origin, x, factors, model


def residual_variance(x, kmax=KMAX):
    """V(k): mean squared residual after k principal components, k = 0..kmax."""
    s = np.linalg.svd(x, compute_uv=False)
    tail = np.cumsum((s**2)[::-1])[::-1]  # tail[k] = sum_{j >= k} s_j^2
    squares = np.append(tail, 0.0)
    return squares[: kmax + 1] / x.size, (s**2) / x.size


def bai_ng(x):
    """Bai and Ng (2002) IC1, IC2, IC3: the minimizing k in 0..KMAX."""
    t, n = x.shape
    v, _ = residual_variance(x)
    k = np.arange(KMAX + 1)
    base = (n + t) / (n * t)
    m = min(n, t)
    ic = {
        "IC1": np.log(v) + k * base * np.log(1 / base),
        "IC2": np.log(v) + k * base * np.log(m),
        "IC3": np.log(v) + k * np.log(m) / m,
    }
    return {name: int(np.argmin(values)) for name, values in ic.items()}


def ahn_horenstein(x):
    """Ahn and Horenstein (2013) eigenvalue ratio and growth ratio: the maximizing k in 1..KMAX."""
    _, mu = residual_variance(x)
    tail = np.cumsum(mu[::-1])[::-1]  # tail[k] = sum_{j >= k} mu_j (0-based)
    k = np.arange(1, KMAX + 1)
    er = mu[k - 1] / mu[k]
    gr = np.log(tail[k - 1] / tail[k]) / np.log(tail[k] / tail[k + 1])
    return {"ER": int(np.argmax(er)) + 1, "GR": int(np.argmax(gr)) + 1}


def abc_penalty(m, t, variant):
    """The three penalties p(n, T) of Bai and Ng (2002), as used by Alessi et al. (2010)."""
    base = (m + t) / (m * t)
    if variant == 1:
        return base * np.log(m * t / (m + t))
    if variant == 2:
        return base * np.log(min(m, t))
    return np.log(min(m, t)) / min(m, t)


def abc_choice(x, rng, variant=1):
    """Alessi, Barigozzi and Capasso (2010) criterion for one random ordering of the series.

    Nested subpanels of n, n-1, ..., 3n/4 series; for each penalty multiplier c the minimizer of
    log V(k) + k c p(n, T). The answer is r on the first c-interval, wider than ABC_MIN_WIDTH,
    where all subpanels agree (zero spread) on a value below KMAX.
    """
    t, n = x.shape
    order = rng.permutation(n)
    sizes = np.arange(n, int(np.ceil(ABC_SHARE * n)) - 1, -1)
    k = np.arange(KMAX + 1)
    chosen = np.empty((len(sizes), len(ABC_GRID)), dtype=int)
    for a, m in enumerate(sizes):
        v, _ = residual_variance(x[:, order[:m]])
        penalty = abc_penalty(m, t, variant)
        chosen[a] = (np.log(v) + ABC_GRID[:, None] * penalty * k).argmin(axis=1)
    stable = (chosen.std(axis=0) == 0) & (chosen[0] < KMAX)
    width = round(ABC_MIN_WIDTH / (ABC_GRID[1] - ABC_GRID[0]))
    for i in range(len(ABC_GRID) - width):
        if stable[i : i + width + 1].all() and (chosen[0, i : i + width + 1] == chosen[0, i]).all():
            return int(chosen[0, i])
    return None


def abc_mode(x, seed, variant=1):
    rng = np.random.default_rng(seed)
    draws = [abc_choice(x, rng, variant) for _ in range(ABC_DRAWS)]
    values, counts = np.unique([d for d in draws if d is not None], return_counts=True)
    return int(values[counts.argmax()]), counts.max() / ABC_DRAWS, draws


def abc_path(x, seed, variant=1):
    """Full-sample number of factors for each c, and the share of draws in which all subpanels agree."""
    t, n = x.shape
    k = np.arange(KMAX + 1)
    v, _ = residual_variance(x)
    full = (np.log(v) + ABC_GRID[:, None] * abc_penalty(n, t, variant) * k).argmin(axis=1)
    rng = np.random.default_rng(seed)
    sizes = np.arange(n, int(np.ceil(ABC_SHARE * n)) - 1, -1)
    agree = np.zeros(len(ABC_GRID))
    for _ in range(ABC_DRAWS):
        order = rng.permutation(n)
        chosen = np.empty((len(sizes), len(ABC_GRID)), dtype=int)
        for a, m in enumerate(sizes):
            vm, _ = residual_variance(x[:, order[:m]])
            penalty = abc_penalty(m, t, variant)
            chosen[a] = (np.log(vm) + ABC_GRID[:, None] * penalty * k).argmin(axis=1)
        agree += chosen.std(axis=0) == 0
    return full, agree / ABC_DRAWS


def plateau(full, r):
    """First and last c of the longest run of the full-sample path equal to r."""
    runs, start = [], 0
    for i in range(1, len(full) + 1):
        if i == len(full) or full[i] != full[start]:
            if full[start] == r:
                runs.append((ABC_GRID[start], ABC_GRID[i - 1]))
            start = i
    return max(runs, key=lambda run: run[1] - run[0])


def abc_figure(x, seed, selected):
    """The number of factors against the penalty multiplier c, and how often subpanels agree."""
    full, agree = abc_path(x, seed)
    low, high = plateau(full, selected)
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(SIZES["column"][0], 3.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    for axis in (top, bottom):
        axis.axvspan(low, high, color=PALETTE[2], alpha=0.25, linewidth=0)
    top.step(ABC_GRID, full, where="post", color=PALETTE[0], linewidth=1.2)
    top.set_ylim(0, KMAX + 1)
    top.set_yticks([1, 3, 5, 10, 15])
    top.set_title("(a) Number of factors selected")
    top.set_ylabel("Factors")
    bottom.plot(ABC_GRID, agree, color=PALETTE[1], linewidth=1.2)
    bottom.set_ylim(0, 1.02)
    bottom.set_yticks([0, 0.5, 1])
    bottom.set_title("(b) Share of orderings in which all subsets agree")
    bottom.set_ylabel("Share")
    bottom.set_xlabel("Penalty multiplier $c$")
    bottom.set_xlim(1.0, 3.0)
    bottom.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    fig.tight_layout()
    one = ABC_GRID[np.argmax(full == 1)]
    record_number("AbcCFrom", f"{low:.1f}")
    record_number("AbcCTo", f"{high:.1f}")
    record_number("AbcOneFrom", f"{one:.1f}")
    on_plateau = (ABC_GRID >= low) & (ABC_GRID <= high)
    record_number("AbcAgreeThreeMax", f"{100 * agree[on_plateau].max():.0f}")
    record_number("AbcAgreeOneMax", f"{100 * agree[full == 1].max():.0f}")
    return fig


def var_orders(factors, trend):
    """Lag orders chosen by AIC, HQIC and BIC up to MAX_VAR_LAG on a common sample."""
    chosen = VAR(factors).select_order(maxlags=MAX_VAR_LAG, trend=trend).selected_orders
    return {"AIC": chosen["aic"], "HQIC": chosen["hqic"], "BIC": chosen["bic"]}


def tables():
    origin, x, pca_factors, model = first_window()
    seed = DESIGN["project"]["random_seed"]
    bn, ah = bai_ng(x), ahn_horenstein(x)
    abc, share, draws = abc_mode(x, seed)
    other = {v: abc_mode(x, seed, v)[:2] for v in (2, 3)}
    number = pd.DataFrame(
        {
            "Criterion": [
                "Bai and Ng, IC1",
                "Bai and Ng, IC2",
                "Bai and Ng, IC3",
                "Ahn and Horenstein, ER",
                "Ahn and Horenstein, GR",
                "Alessi et al., penalty 1 (used)",
                "Alessi et al., penalty 2",
                "Alessi et al., penalty 3",
            ],
            "Factors": [
                bn["IC1"],
                bn["IC2"],
                bn["IC3"],
                ah["ER"],
                ah["GR"],
                abc,
                other[2][0],
                other[3][0],
            ],
            "Share of orderings": ["---"] * 5
            + [f"{100 * share:.0f}%", f"{100 * other[2][1]:.0f}%", f"{100 * other[3][1]:.0f}%"],
        }
    )
    em = model.fit(maxiter=3000)
    em_factors = em.posterior.mean.reshape(-1, model.r)[4:]
    pca = var_orders(pca_factors, "c")
    smoothed = var_orders(em_factors, "n")
    order = pd.DataFrame({"PCA factors": pca, "EM-smoothed factors": smoothed})
    for name, value in [
        ("NFactorsIcOne", bn["IC1"]),
        ("NFactorsIcTwo", bn["IC2"]),
        ("NFactorsIcThree", bn["IC3"]),
        ("NFactorsEr", ah["ER"]),
        ("NFactorsGr", ah["GR"]),
        ("NFactorsAbc", abc),
        ("NFactorsMax", KMAX),
        ("NAbcDraws", ABC_DRAWS),
        ("VarOrderAicEm", smoothed["AIC"]),
        ("VarOrderBicEm", smoothed["BIC"]),
        ("VarOrderAicPca", pca["AIC"]),
        ("VarOrderBicPca", pca["BIC"]),
    ]:
        record_number(name, str(value))
    record_number("ShareAbc", f"{100 * share:.0f}")
    record_number("NFactorsAbcTwo", str(other[2][0]))
    record_number("ShareAbcTwo", f"{100 * other[2][1]:.0f}")
    record_number("NFactorsAbcThree", str(other[3][0]))
    record_number("ShareAbcThree", f"{100 * other[3][1]:.0f}")
    record_number("ShareAbcOne", f"{100 * sum(d == 1 for d in draws) / ABC_DRAWS:.0f}")
    figure = abc_figure(x, seed, abc)
    return origin, x.shape, number, order, figure


def scree_figure(x):
    """Variance share of the leading principal components and the eigenvalue ratios."""
    t, n = x.shape
    r = DESIGN["factor_model"]["selected_r"]
    _, mu = residual_variance(x)
    share = 100 * mu / mu.sum()
    edge = 100 * (1 + np.sqrt(n / t)) ** 2 / n  # Marchenko-Pastur upper edge, pure noise
    k = np.arange(1, 11)
    colors = [PALETTE[0] if i <= r else "#BBBBBB" for i in k]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    left.bar(k, share[:10], color=colors)
    left.axhline(edge, color="#555555", linestyle="--", linewidth=0.8)
    left.text(10.4, edge + 0.5, f"noise edge {edge:.1f}%", ha="right", color="#333333")
    left.set_xlabel("Principal component")
    left.set_ylabel("Share of variance (%)")
    right.bar(k, mu[:10] / mu[1:11], color=colors)
    right.axhline(1, color="#333333", linewidth=0.6)
    right.set_xlabel("$k$")
    right.set_ylabel(r"Eigenvalue ratio $\mu_k/\mu_{k+1}$")
    for axis in (left, right):
        axis.set_xticks(k)
    fig.tight_layout()
    record_number("NPcsAboveNoise", str(int((share > edge).sum())))
    return fig


def order_fits(orders=(1, 2, 3)):
    """Cold EM fits of the monthly panel for each VAR order, at the first date and each December."""
    design, settings, _, panel, info, lags, origins = prepare(PROFILE, "M")
    r = design["factor_model"]["selected_r"]
    nm = int((info.Frequency == "M").sum())
    rows = []
    for origin in [origins[0]] + [o for o in origins if o.month == 12]:
        _, _, z, _, _, _ = clean_window(panel, lags, origin)
        for p in orders:
            fit = OrderPEM(z, nm, r, settings["initial_factor_variance"], p).fit(maxiter=3000)
            if not fit.converged:
                raise RuntimeError(f"VAR({p}) fit did not converge at {origin}")
            llf = fit.posterior.llf
            if origin == origins[0] and p == orders[0]:
                record_number("LagPenaltyFirst", f"{0.5 * r * r * np.log(len(z)):.0f}")
            rows.append(
                {
                    "origin": str(origin),
                    "p": p,
                    "llf": llf,
                    "root": companion_radius(fit.params.ar, r),
                    # the VAR coefficients are the only parameters that differ across orders
                    "bic": -2 * llf + r * r * p * np.log(len(z)),
                }
            )
    fits = pd.DataFrame(rows)
    first = fits[fits["origin"] == fits["origin"].iloc[0]].set_index("p")
    best = fits.loc[fits.groupby("origin")["bic"].idxmin(), "p"].value_counts()
    table = pd.DataFrame(
        {
            "Log-likelihood, first date": first["llf"].map(lambda v: f"{v:.1f}"),
            "Largest root, first date": first["root"].map(lambda v: f"{v:.2f}"),
            "Largest root, any date": fits.groupby("p")["root"].max().map(lambda v: f"{v:.2f}"),
            "Dates with the lowest BIC": [int(best.get(p, 0)) for p in orders],
        }
    )
    table.index = [f"VAR({p})" for p in orders]
    llf = fits.pivot(index="origin", columns="p", values="llf")
    record_number("NLagDates", str(len(llf)))
    record_number("LagGainOneTwoMin", f"{(llf[2] - llf[1]).min():.0f}")
    record_number("LagGainOneTwoMax", f"{(llf[2] - llf[1]).max():.0f}")
    record_number("LagGainTwoThreeMin", f"{(llf[3] - llf[2]).min():.0f}")
    record_number("LagGainTwoThreeMax", f"{(llf[3] - llf[2]).max():.0f}")
    record_number("LagOneRootMax", f"{fits[fits['p'] == 1]['root'].max():.2f}")
    record_number("LagTwoRootMax", f"{fits[fits['p'] == 2]['root'].max():.2f}")
    for name, p in [("One", 1), ("Two", 2), ("Three", 3)]:
        record_number(f"LagBicDates{name}", str(int(best.get(p, 0))))
    record_number("LagBicFirst", str(int(first["bic"].idxmin())))
    return table


def em_diagnostics():
    fits = pd.read_csv(f"{RUN}/fit_monitoring.csv")
    rows = {}
    for name, g in [*fits.groupby("information_set", sort=False), ("All", fits)]:
        rows[name] = {
            "Fits": len(g),
            "Not converged": int((~g["converged"]).sum()),
            "Median iterations": g["iterations"].median(),
            "Maximum iterations": g["iterations"].max(),
            "Largest VAR root": g["rho"].max(),
            "Flagged": int(g["flagged"].sum()),
        }
    table = pd.DataFrame(rows).T
    counts = ["Fits", "Not converged", "Median iterations", "Maximum iterations"]
    counts += ["Flagged"]
    table[counts] = table[counts].map(lambda v: f"{v:.0f}")
    table["Largest VAR root"] = table["Largest VAR root"].map(lambda v: f"{v:.2f}")
    return table


def main():
    origin, shape, number, order, abc_fig = tables()
    t, n = shape
    save_table(
        number,
        "tab_80_factor_number",
        notes=(
            f"The {n} monthly series at the first forecast date ({origin}), {t} months; at most "
            f"{KMAX} factors. ER, GR: eigenvalue and growth ratios. Penalty 1 to 3: the penalties "
            "of IC1 to IC3, scaled by the constant of Alessi et al. Share of orderings: share of "
            f"{ABC_DRAWS} random orderings of the series that return the number shown. "
            "\\textit{Source:} EA-MD-QD, own calculations."
        ),
        used_in="paper: methodology, number of factors",
        float_format="%.0f",
        index=False,
        rule_before=["Ahn and Horenstein, ER", "Alessi et al., penalty 1 (used)"],
    )
    save_table(
        order,
        "tab_81_var_order",
        notes=(
            f"Lag order of a VAR on the three factors at the first forecast date ({origin}), by "
            f"criterion, up to {MAX_VAR_LAG} lags. PCA factors: principal components of the "
            "balanced panel, with an intercept, as in the PCA models; EM-smoothed factors: those "
            "of the state-space model, without intercept. The model uses two lags. "
            "\\textit{Source:} EA-MD-QD, own calculations."
        ),
        used_in="paper: appendix, factor VAR order",
        float_format="%.0f",
    )
    save_fig(
        scree_figure(first_window()[1]),
        "fig_80_scree",
        used_in="paper: appendix, number of factors",
    )
    save_fig(abc_fig, "fig_81_abc", used_in="paper: methodology, number of factors")
    save_table(
        order_fits(),
        "tab_83_var_order_fits",
        notes=(
            "EM fits of the \\NSeriesMonthly{} monthly series, started from principal components, "
            "at the first forecast date and at each December through 2025 (\\NLagDates{} dates), "
            "by order of the factor VAR. Largest root: of the companion matrix. BIC: $-2$ times "
            "the log-likelihood of the state-space model plus the number of VAR coefficients "
            "times the log of the number of months. The model uses two lags. \\textit{Source:} "
            "EA-MD-QD, own calculations."
        ),
        used_in="paper: appendix, factor VAR order",
        float_format="%.2f",
        column_format="lrrrr",
    )
    save_table(
        em_diagnostics(),
        "tab_82_em_diagnostics",
        notes=(
            "One EM fit per forecast date and information set, each started from principal "
            "components. Largest VAR root: of the companion matrix. A fit is flagged if it does "
            f"not converge, if its largest root reaches {MONITOR['flag_rho']} or if it fails "
            "numerically. \\textit{Source:} EA-MD-QD, own calculations."
        ),
        used_in="paper: appendix, EM diagnostics",
        float_format="%.2f",
        column_format="lrrrrrr",
    )


if __name__ == "__main__":
    main()
