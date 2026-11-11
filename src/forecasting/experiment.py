"""Expanding-origin runner. No full-sample fitting or tuning is performed here."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.api import VAR
from threadpoolctl import threadpool_limits

from src.data.ea_md_qd import FILE, SNAPSHOT, available, load, release_lags
from src.forecasting.design import (
    bic_regression,
    dictionary,
    sparse_predict,
    target_month,
    training_sample,
)
from src.forecasting.dfm import MixedFrequencyEM, Parameters, pca_fit, radius
from src.forecasting.initialization import BalancedStartEM
from src.forecasting.pca import balanced_pca, carried_common
from src.preprocessing.transform import transform

MODELS = [
    "AR",
    "PCR",
    "AR_PCR",
    "PCA_VAR",
    "sparse_only",
    "PCA_VAR_LASSO",
    "MF_factor_only",
    "MF_factor_own_AR",
    "FarmPredict_M",
    "FarmPredict_MQ",
]


def load_settings(profile="config/forecast_analysis.yaml"):
    design_path = Path("config/research_design.yaml")
    profile_path = Path(profile)
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    settings = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if "inherits" in settings:
        parent = Path(settings["inherits"])
        settings = {**yaml.safe_load(parent.read_text()), **settings}
    if settings["horizon"] != 1:
        raise ValueError("This runner implements h=1 only")
    if (
        design["factor_model"]["selected_r"] != 3
        or design["factor_forecast"]["factor_var_order"] != 2
    ):
        raise ValueError("This execution profile requires the agreed r=3, VAR(2)")
    return (
        design,
        settings,
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [design_path, profile_path, FILE]
        },
    )


def prepare(profile="config/forecast_analysis.yaml", information_set=None):
    design, settings, hashes = load_settings(profile)
    raw, info = load()
    monthly = list(info.index[info.Frequency == "M"])
    quarterly = settings["quarterly_series"]
    if len(monthly) != 47 or len(quarterly) != 16 or len(set(quarterly)) != 16:
        raise ValueError("Expected exactly 47 monthly and 16 unique quarterly series")
    if not (info.loc[quarterly, "Frequency"] == "Q").all():
        raise ValueError("Quarterly whitelist contains a nonquarterly series")
    if not (info.loc[quarterly, "TR3"] == 2).all():
        raise ValueError("Recheck quarterly measurement equations after transformation changes")
    if information_set is not None:
        from src.data.external import load_external

        external, manifest = load_external()
        if information_set not in ["M", "MQ", "ME", "MQE"]:
            raise ValueError("Unknown paper information set")
        settings["information_set"] = information_set
        settings["models"] = ["MF_factor_only", "MF_factor_own_AR", "sparse_only", "FarmPredict_MQ"]
        if information_set in ["M", "ME"]:
            settings["models"] += ["PCR", "PCA_VAR", "PCA_VAR_LASSO"]
        if information_set == "M":
            settings["models"] += ["RW", "AR"]
        if "Q" not in information_set:
            quarterly = []
        settings["quarterly_series"] = quarterly
    names = monthly + quarterly
    lags = release_lags(raw, info)[names]
    start = design["research"]["forecast"]["start_date"]
    panel = transform(raw[names], info).loc[start:]
    if information_set is not None:
        settings["panel_monthly"] = monthly.copy()
        if "E" in information_set:
            members = list(external)
            panel = pd.concat(
                [panel[monthly], external.reindex(panel.index), panel[quarterly]], axis=1
            )
            external_info = pd.DataFrame(index=members, columns=info.columns)
            external_info["Frequency"] = "M"
            info = pd.concat([info, external_info])
            lags = pd.concat(
                [
                    lags,
                    pd.Series({k: v["release_lag_months"] for k, v in manifest["series"].items()}),
                ]
            )
            monthly = monthly + members
            names = monthly + quarterly
            lags = lags.reindex(names)
        hashes.update(
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("data/external/factor_sparse").rglob("*")
                if p.is_file()
            }
        )
        parent = Path(settings["inherits"])
        hashes[str(parent)] = hashlib.sha256(parent.read_bytes()).hexdigest()
    targets = settings.get("targets", monthly)
    if not set(targets).issubset(monthly):
        raise ValueError("Targets must belong to the monthly panel")
    origins = []
    for origin in pd.period_range(settings["test_start"], SNAPSHOT - 1, freq="M"):
        # All models and all targets share origins and fully observed scoring values.
        if all(
            target_month(origin, lags[name]) in panel.index
            and pd.notna(panel.loc[target_month(origin, lags[name]), name])
            for name in targets
        ):
            origins.append(origin)
    if not origins:
        raise ValueError("No common, scoreable test origins")
    return design, settings, hashes, panel, info.loc[names], lags, pd.PeriodIndex(origins, freq="M")


def clean_window(panel, lags, origin):
    """Source release mask first, then in-window D-18 cleaning and D-19 scaling."""
    published = available(panel, lags, origin)
    # D-72: median and IQR of the nonzero values. Zero-inflated changes (iron-ore
    # and EU gas prices before 2010, rounded unemployment changes) make the IQR of
    # all values zero or near zero, so the 10-IQR rule flagged nearly every
    # nonzero move. A series without a positive nonzero IQR is never flagged.
    nonzero = published.where(published.abs() > 1e-9)
    median = nonzero.median()
    iqr = nonzero.quantile(0.75) - nonzero.quantile(0.25)
    outliers = (published - median).abs().gt(10 * iqr, axis="columns") & (iqr > 0)
    outliers.loc[(published.index.year >= 2020) & (published.index.year <= 2021)] = False
    cleaned = published.mask(outliers)
    mean, scale = cleaned.mean(), cleaned.std(ddof=0)
    if mean.isna().any() or (scale <= 1e-12).any() or scale.isna().any():
        raise ValueError("Undefined in-window mean or nonpositive series variance")
    return published, cleaned, (cleaned - mean) / scale, mean, scale, outliers


def fit_var(factors, maxlag=8):
    """Compare lag orders on a common BIC estimation sample via statsmodels."""
    model = VAR(factors)
    p = model.select_order(maxlags=maxlag).selected_orders["bic"]
    return model.fit(p), int(p)


def factor_common_at(factors, loadings, fitted_var, index, date, target):
    if date <= index[-1]:
        f = factors[index.get_loc(date)]
    else:
        steps = (date - index[-1]).n
        if fitted_var.k_ar:
            f = fitted_var.forecast(factors[-fitted_var.k_ar :], steps)[-1]
        else:
            f = np.asarray(fitted_var.intercept)
    return float(loadings[target] @ f)


def own_features(panel, info, lags, origins, target):
    return dictionary(panel[[target]], info.Frequency, lags, origins)[0]


def starting_point(settings, previous):
    """D-71: 'cold' starts every date from the balanced-end PCA start and ignores the previous fit.

    'warm' (the default, D-45) chains each date's EM from the preceding date's estimates.
    """
    mode = settings.get("reestimation_start", "warm")
    if mode not in ("warm", "cold"):
        raise ValueError(f"Unknown reestimation_start: {mode}")
    return None if mode == "cold" else previous


def run_origin(panel, info, lags, origin, design, settings, previous=None, ordinal=0):
    """One shared DFM and PCA fit, then the same per-target code path for all 47."""
    previous = starting_point(settings, previous)
    selected_models = settings.get("models", MODELS)
    start_time = time.perf_counter()
    published, cleaned, z, mean, scale, outliers = clean_window(panel, lags, origin)
    monitoring = design["factor_model"]["fit_monitoring"]
    r = design["factor_model"]["selected_r"]
    nm = int((info.Frequency == "M").sum())
    initializer = settings.get("dfm_initialization", "original")
    if initializer not in ["original", "balanced_t_minus_3"]:
        raise ValueError(f"Unknown experimental DFM initializer: {initializer}")
    em_class = BalancedStartEM if initializer == "balanced_t_minus_3" else MixedFrequencyEM
    if initializer == "balanced_t_minus_3" and int(lags.iloc[:nm].max()) != 3:
        raise ValueError("The experimental initializer requires a maximum monthly delay of 3")
    model = em_class(z, nm, r, initial_variance=settings["initial_factor_variance"])
    warm = None
    if previous is not None:
        old, old_scale, _ = previous
        ratios = old_scale.to_numpy() / scale.to_numpy()
        warm = Parameters(
            old.loadings * ratios[:, None],
            old.variances * ratios**2,
            old.ar.copy(),
            old.innovation.copy(),
        )
    cap = monitoring["em_cap_cold_first_fit"] if warm is None else monitoring["em_cap_warm_fit"]
    crashed, cold_check_gap = False, None
    try:
        fit = model.fit(warm, maxiter=cap)
    except (np.linalg.LinAlgError, RuntimeError, ValueError):
        if warm is None:
            raise
        crashed = True
        fit = model.fit(maxiter=monitoring["em_cap_cold_first_fit"])
    if warm is not None and not fit.converged:
        fit = model.fit(maxiter=monitoring["em_cap_cold_first_fit"])
    if warm is not None and (ordinal + 1) % monitoring["cold_check_every_n_dates"] == 0:
        cold = model.fit(maxiter=monitoring["em_cap_cold_first_fit"])
        cold_check_gap = abs(radius(cold.params.ar) - radius(fit.params.ar))
    rho = radius(fit.params.ar)
    flagged = (
        not fit.converged
        or rho >= monitoring["flag_rho"]
        or crashed
        or (cold_check_gap is not None and cold_check_gap > monitoring["cold_check_rho_gap"])
    )
    # Training responses may be cleaned; scoring always uses the published panel.
    policy = settings.get("training_outlier_policy", "published_labels_and_own_lags")
    if policy not in ["cleaned_inputs_and_labels", "published_labels_and_own_lags"]:
        raise ValueError(f"Unknown training outlier policy: {policy}")
    training_values = cleaned if policy == "cleaned_inputs_and_labels" else published
    common = fit.common() * scale + mean
    residual_labels = training_values - common
    residual_inputs = cleaned - common
    origins = z.index
    w_mq, _ = dictionary(residual_inputs, info.Frequency, lags, origins)
    w_m = w_mq.loc[:, w_mq.columns.str.contains("__M", regex=False)]
    w_raw, _ = dictionary(cleaned, info.Frequency, lags, origins)
    # PCR and EM retain their original PCA estimates; only the two VAR
    # benchmarks adopt the balanced-end method requested on 7 November.
    pca_iterations, var_order = None, None
    if set(selected_models) & {"PCR", "AR_PCR", "PCA_VAR", "PCA_VAR_LASSO"}:
        pcr_f, _, _ = pca_fit(z.iloc[:, :nm], r)
        f, loading, factor_index, pca_iterations = balanced_pca(
            z.iloc[:, :nm], r, int(lags.iloc[:nm].max())
        )
        var, var_order = fit_var(f, design["factor_forecast"]["max_lag"])
        pca_common = carried_common(f, loading, var, factor_index, z.index)
        pca_common.columns = z.columns[:nm]
        pca_common = pca_common * scale.iloc[:nm] + mean.iloc[:nm]
        pca_residuals = cleaned.iloc[:, :nm] - pca_common
        pca_labels = training_values.iloc[:, :nm] - pca_common
        w_pca, _ = dictionary(pca_residuals, info.Frequency, lags, origins)
        # PCR has a scalar target regression on PCA factor lags, distinct from PCA+VAR.
        # The factor reference date is at/before the latest fully released monthly row.
        # This prevents a historical target from entering its own contemporaneous PC regressor.
        f_index = pd.DataFrame(pcr_f, index=origins, columns=[f"F{k + 1}" for k in range(r)])
        factor_features = pd.DataFrame(index=origins)
        factor_delay = int(lags.iloc[:nm].max())
        for lag in range(8):
            for name in f_index:
                factor_features[f"{name}__lag{lag + 1}"] = f_index[name].shift(factor_delay + lag)
    records, selections, tuning = [], [], []
    for j, name in enumerate(z.columns[:nm]):
        if name not in settings.get("targets", z.columns[:nm]):
            continue
        target_date = target_month(origin, lags[name])
        forecasts = {}
        unavailable = {}

        def estimate(label, function, *args, unavailable=unavailable):
            try:
                result = function(*args)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                if "information_set" not in settings:
                    raise
                unavailable[label] = f"{type(error).__name__}: {error}"
                if function is sparse_predict:
                    return np.nan, pd.Series(dtype=float), {"failure_reason": unavailable[label]}
                return np.nan, {"failure_reason": unavailable[label]}
            if "failure_reason" in result[-1]:
                unavailable[label] = result[-1]["failure_reason"]
            return result

        mf_common = float(mean[name] + scale[name] * fit.common_at(target_date, name))
        pc_common = 0.0
        if set(selected_models) & {"PCA_VAR", "PCA_VAR_LASSO"}:
            pc_common = float(
                mean[name]
                + scale[name] * factor_common_at(f, loading, var, factor_index, target_date, j)
            )
        own = own_features(training_values, info, lags, origins, name)
        own_groups = [list(own.columns[:p]) for p in range(1, 5)]
        x, y = training_sample(own, training_values[name], lags[name], origin)
        if "AR" in selected_models:
            forecasts["AR"], detail = estimate(
                "AR", bic_regression, x, y, own.loc[origin], own_groups
            )
            tuning.append({"target": name, "model": "AR", **detail})
        if "RW" in selected_models:
            forecasts["RW"] = float(published[name].dropna().iloc[-1])
        for benchmark, with_ar in [("PCR", False), ("AR_PCR", True)]:
            if benchmark not in selected_models:
                continue
            features = pd.concat([factor_features, own], axis=1) if with_ar else factor_features
            groups = []
            for q in range(1, 9):
                pcs = list(factor_features.columns[: q * r])
                if with_ar:
                    groups.extend([pcs + list(own.columns[:p]) for p in range(1, 5)])
                else:
                    groups.append(pcs)
            x, y = training_sample(features, training_values[name], lags[name], origin)
            forecasts[benchmark], detail = estimate(
                benchmark, bic_regression, x, y, features.loc[origin], groups
            )
            tuning.append({"target": name, "model": benchmark, **detail})
        forecasts["MF_factor_only"] = mf_common
        forecasts["PCA_VAR"] = pc_common
        own_u = own_features(residual_labels, info, lags, origins, name)
        x, y = training_sample(own_u, residual_labels[name], lags[name], origin)
        u, detail = estimate(
            "MF_factor_own_AR", bic_regression, x, y, own_u.loc[origin], own_groups
        )
        forecasts["MF_factor_own_AR"] = mf_common + u
        tuning.append({"target": name, "model": "MF_factor_own_AR", **detail})
        for label, features, response, base in [
            ("FarmPredict_M", w_m, residual_labels[name], mf_common),
            ("FarmPredict_MQ", w_mq, residual_labels[name], mf_common),
            (
                "PCA_VAR_LASSO",
                w_pca if "PCA_VAR_LASSO" in selected_models else None,
                pca_labels[name] if "PCA_VAR_LASSO" in selected_models else None,
                pc_common,
            ),
            ("sparse_only", w_raw, training_values[name], 0.0),
        ]:
            if label not in selected_models:
                continue
            x, y = training_sample(features, response, lags[name], origin)
            prediction, selected, detail = estimate(
                label, sparse_predict, x, y, features.loc[origin]
            )
            forecasts[label] = base + prediction
            tuning.append({"target": name, "model": label, **detail})
            for term, coefficient in selected.items():
                variable, position = term.rsplit("__", 1)
                selections.append(
                    {
                        "target": name,
                        "model": label,
                        "term": term,
                        "predictor": variable,
                        "frequency": position[0],
                        "released_position": int(position[1:]),
                        "coefficient": coefficient,
                    }
                )
        actual = panel.loc[target_date, name]
        for label in selected_models:
            value = forecasts[label]
            if (
                not np.isfinite(value)
                and label not in unavailable
                and "information_set" not in settings
            ):
                raise RuntimeError(f"Nonfinite forecast: {origin} {name} {label}")
            records.append(
                {
                    "target": name,
                    "target_month": str(target_date),
                    "model": label,
                    "actual": float(actual),
                    "forecast": value,
                    "error": float(actual - value),
                    "flagged": flagged,
                    "status": "unavailable" if label in unavailable else "ok",
                    "failure_reason": unavailable.get(label, ""),
                }
            )
    log = {
        "origin": str(origin),
        "n_train_months": len(z),
        "iterations": fit.iterations,
        "converged": fit.converged,
        "llf": fit.posterior.llf,
        "rho": rho,
        "crashed": crashed,
        "cold_check_rho_gap": cold_check_gap,
        "flagged": flagged,
        "pca_iterations": pca_iterations,
        "pca_var_order": var_order,
        "input_outliers": int(outliers.sum().sum()),
        **(
            {
                "data_availability": {
                    name: {
                        "published": int(published[name].notna().sum()),
                        "cleaned": int(cleaned[name].notna().sum()),
                        "outliers": int(outliers[name].sum()),
                    }
                    for name in panel
                }
            }
            if "information_set" in settings
            else {}
        ),
        "em_min_likelihood_increment": float(np.min(np.diff(fit.likelihood))),
        "seconds": time.perf_counter() - start_time,
    }
    for rows in [records, selections, tuning]:
        for row in rows:
            row["origin"] = str(origin)
    return records, selections, tuning, log, (fit.params, scale, mean)


def run(
    profile="config/forecast_analysis.yaml",
    output="outputs/forecasts/analysis_final",
    limit=None,
    end=None,
    resume=False,
    information_set=None,
):
    design, settings, hashes, panel, info, lags, origins = prepare(profile, information_set)
    if end:
        origins = origins[origins <= pd.Period(end, "M")]
    if limit is not None:
        origins = origins[:limit]
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "PyYAML"]
    }
    metadata = {
        "panel_members": list(panel.columns),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "methodology_commit": "5ca397a",
        "uncommitted_implementation": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "classification": settings["classification"],
        "design": design,
        "profile": settings,
        "input_hashes": hashes,
        "python": platform.python_version(),
        "packages": versions,
        "origins": list(map(str, origins)),
        "release_lags": lags.to_dict(),
        "vintage": "09.2026, revised and full-sample seasonally adjusted; pseudo-real-time",
        "estimator": "exact factor-only Gaussian EM; quarterly WN integrated with covariance K",
        "smoother": "information-form Gaussian smoother, validated against Kalman",
        "completion": "RUNNING",
    }
    source_paths = (
        sorted(Path("src/forecasting").glob("*.py"))
        + list(Path("src/data").glob("*.py"))
        + list(Path("src/preprocessing").glob("*.py"))
        + [Path("scripts/analysis.py"), Path("uv.lock")]
    )
    metadata["source_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
    }
    metadata_path = output / "metadata.json"
    if resume and metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ["input_hashes", "source_hashes", "profile", "origins", "packages"]:
            if saved.get(key) != metadata[key]:
                raise ValueError(f"Resume provenance mismatch: {key}; use a new run directory")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    info.assign(release_lag_months=lags).to_csv(output / "selected_series.csv")
    previous = None
    with threadpool_limits(limits=1):
        for ordinal, origin in enumerate(origins):
            folder = output / str(origin)
            checkpoint = folder / "parameters.npz"
            if (
                resume
                and checkpoint.exists()
                and all(
                    (folder / name).exists()
                    for name in ["fit.json", "forecasts.csv", "selected.csv", "tuning.csv"]
                )
            ):
                with np.load(checkpoint) as archive:
                    previous = (
                        Parameters(
                            archive["loadings"],
                            archive["variances"],
                            archive["ar"],
                            archive["innovation"],
                        ),
                        pd.Series(archive["scale"], index=panel.columns),
                        pd.Series(archive["mean"], index=panel.columns),
                    )
                print(f"[{ordinal + 1}/{len(origins)}] {origin}: reused checkpoint", flush=True)
                continue
            print(f"[{ordinal + 1}/{len(origins)}] {origin}: fitting", flush=True)
            try:
                records, selected, tuning, log, previous = run_origin(
                    panel, info, lags, origin, design, settings, previous, ordinal
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                if information_set is None:
                    raise
                reason = f"panel_estimation_failure: {type(error).__name__}: {error}"
                folder.mkdir(exist_ok=True)
                rows = []
                for target in settings["targets"]:
                    date = target_month(origin, lags[target])
                    for label in settings["models"]:
                        rows.append(
                            {
                                "origin": str(origin),
                                "target": target,
                                "target_month": str(date),
                                "model": label,
                                "forecast": np.nan,
                                "actual": panel.loc[date, target],
                                "error": np.nan,
                                "flagged": True,
                                "status": "unavailable",
                                "failure_reason": reason,
                            }
                        )
                pd.DataFrame(rows).to_csv(folder / "forecasts.csv", index=False)
                pd.DataFrame(
                    columns=[
                        "origin",
                        "target",
                        "model",
                        "term",
                        "predictor",
                        "frequency",
                        "released_position",
                        "coefficient",
                    ]
                ).to_csv(folder / "selected.csv", index=False)
                pd.DataFrame(columns=["origin", "target", "model"]).to_csv(
                    folder / "tuning.csv", index=False
                )
                (folder / "fit.json").write_text(
                    json.dumps({"origin": str(origin), "flagged": True, "failure_reason": reason})
                )
                print(reason, flush=True)
                # No parameter checkpoint: resume retries failed origins from the
                # preceding successful fit, rather than accepting a broken fit.
                continue
            folder.mkdir(exist_ok=True)
            pd.DataFrame(records).to_csv(folder / "forecasts.csv", index=False)
            pd.DataFrame(
                selected,
                columns=[
                    "origin",
                    "target",
                    "model",
                    "term",
                    "predictor",
                    "frequency",
                    "released_position",
                    "coefficient",
                ],
            ).to_csv(folder / "selected.csv", index=False)
            pd.DataFrame(tuning).to_csv(folder / "tuning.csv", index=False)
            (folder / "fit.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            p, scale, mean = previous
            np.savez_compressed(
                folder / "parameters.npz",
                loadings=p.loadings,
                variances=p.variances,
                ar=p.ar,
                innovation=p.innovation,
                scale=scale.to_numpy(),
                mean=mean.to_numpy(),
            )
            print(
                f"  {log['seconds']:.1f}s; iterations={log['iterations']}; "
                f"rho={log['rho']:.4f}; flagged={log['flagged']}",
                flush=True,
            )
    metadata["completion"] = "COMPLETE"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output
