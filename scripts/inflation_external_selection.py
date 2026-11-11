"""Audit saved direct-LASSO selections and core factor/sparse channels; no refits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fixed_factor_mqe_nonzero import verify_saved_files
from src import _style
from src.data.external import IDS
from src.forecasting.design import complete_design, dictionary, training_sample
from src.forecasting.experiment import clean_window, prepare

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/inflation_external_selection")
TARGETS = ["HICPOV_EA", "HICPNG_EA", "HICPNEF_EA"]
PANELS = ["M", "MQ", "ME", "MQE"]
FIXED = {
    "ME": Path("outputs/forecasts/fixed_dfm_external_sparse_nonzero_iqr"),
    "MQE": Path("outputs/reports/nonzero_robustness/mqe_ablation"),
}


def group(predictor, target, frequency):
    if predictor == target:
        return "own_target"
    if predictor in IDS:
        return "external"
    return "quarterly" if frequency == "Q" else "other_monthly"


def samples(frame):
    crisis = frame.origin.between("2021-01", "2023-12")
    return [("primary", frame), ("outside_crisis", frame[~crisis]), ("crisis", frame[crisis])]


def main():
    if OUT.exists():
        raise FileExistsError("Use a new report directory; saved reports are immutable")
    paths = [
        ROOT / name
        for name in ["forecasts.csv", "coverage.csv", "selected_predictors.csv", "tuning.csv"]
    ]
    paths += [folder / "forecasts.csv" for folder in FIXED.values()]
    paths += [Path(__file__), Path("scripts/fixed_factor_mqe_nonzero.py")]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    forecasts = pd.read_csv(ROOT / "forecasts.csv")
    coverage = pd.read_csv(ROOT / "coverage.csv")
    selected = pd.read_csv(ROOT / "selected_predictors.csv")
    tuning = pd.read_csv(ROOT / "tuning.csv")
    dates = {}
    for target in TARGETS:
        values = coverage[
            coverage.target.eq(target) & coverage["sample"].eq("primary")
        ].common_dates.unique()
        assert len(values) == 1
        dates[target] = set(values[0].split(";"))
    assert all(value == dates[TARGETS[0]] for value in dates.values())
    origins = sorted(dates[TARGETS[0]])
    assert len(origins) == 120 and not any(x.startswith("2020") for x in origins)
    forecast_rows, contribution_rows, support_rows = [], [], []
    newline_checks, input_hashes = {}, {}
    for panel_name in PANELS:
        meta = json.loads((ROOT / panel_name / "metadata.json").read_text())
        newline_checks[panel_name] = verify_saved_files(meta)
        _, settings, inputs, panel, info, lags, _ = prepare(
            "config/forecast_paper.yaml", panel_name
        )
        input_hashes[panel_name] = inputs
        assert settings["training_outlier_policy"] == "cleaned_inputs_and_labels"
        for month in origins:
            origin = pd.Period(month, "M")
            _, cleaned, z, _, _, _ = clean_window(panel, lags, origin)
            features, refs = dictionary(cleaned, info.Frequency, lags, z.index)
            for target in TARGETS:
                keys = (
                    selected.origin.eq(month)
                    & selected.target.eq(target)
                    & selected.information_set.eq(panel_name)
                )
                chosen = selected[keys & selected.model.eq("LASSO")]
                tuned = tuning[
                    tuning.origin.eq(month)
                    & tuning.target.eq(target)
                    & tuning.information_set.eq(panel_name)
                    & tuning.model.eq("LASSO")
                ].iloc[0]
                x, y = training_sample(features, cleaned[target], lags[target], origin)
                x, y, row = complete_design(x, y, features.loc[origin])
                assert len(y) == int(tuned.n_train) and len(chosen) == int(tuned.selected)
                components = dict.fromkeys(
                    ["own_target", "external", "quarterly", "other_monthly"], 0.0
                )
                for item in chosen.itertuples():
                    reference = refs.loc[origin, item.term]
                    assert str(reference) == item.reference_month
                    assert reference + int(lags[item.predictor]) <= origin
                    value = item.coefficient * (row[item.term] - x[item.term].mean())
                    category = group(item.predictor, target, item.frequency)
                    components[category] += value
                    contribution_rows.append(
                        {
                            "origin": month,
                            "target": target,
                            "panel": panel_name,
                            "term": item.term,
                            "predictor": item.predictor,
                            "group": category,
                            "coefficient": item.coefficient,
                            "reference_month": str(reference),
                            "contribution": value,
                        }
                    )
                saved = forecasts[
                    forecasts.origin.eq(month)
                    & forecasts.target.eq(target)
                    & forecasts.information_set.eq(panel_name)
                    & forecasts.model.eq("LASSO")
                ].iloc[0]
                prediction = float(y.mean() + sum(components.values()))
                np.testing.assert_allclose(prediction, saved.forecast, rtol=1e-8, atol=1e-8)
                forecast_rows.append(
                    dict(
                        origin=month,
                        target=target,
                        panel=panel_name,
                        actual=saved.actual,
                        forecast=saved.forecast,
                        intercept=float(y.mean()),
                        n_train=len(y),
                        selected=len(chosen),
                        alpha=tuned.alpha,
                        reconstruction_error=prediction - saved.forecast,
                        **components,
                    )
                )
                for model in ["LASSO", "FarmPredict"] if target == "HICPNEF_EA" else ["LASSO"]:
                    s = selected[keys & selected.model.eq(model)].copy()
                    counts = {name: 0 for name in components}
                    for item in s.itertuples():
                        counts[group(item.predictor, target, item.frequency)] += 1
                    support_rows.append(
                        dict(
                            origin=month,
                            target=target,
                            panel=panel_name,
                            model=model,
                            selected=len(s),
                            **counts,
                        )
                    )
        print(
            f"Reconstructed {panel_name}: {len(origins) * len(TARGETS)} direct forecasts",
            flush=True,
        )
    monthly = pd.DataFrame(forecast_rows)
    contributions = pd.DataFrame(contribution_rows)
    support = pd.DataFrame(support_rows)
    support_summary, frequencies, contribution_summary, transitions = [], [], [], []
    for sample, frame in samples(support):
        for keys, g in frame.groupby(["target", "panel", "model"]):
            target, panel_name, model = keys
            support_summary.append(
                dict(
                    sample=sample,
                    target=target,
                    panel=panel_name,
                    model=model,
                    n=len(g),
                    empty_pct=100 * g.selected.eq(0).mean(),
                    mean_selected=g.selected.mean(),
                    **{
                        name + "_pct": 100 * g[name].gt(0).mean()
                        for name in ["own_target", "external", "quarterly", "other_monthly"]
                    },
                )
            )
            s = selected[
                selected.target.eq(target)
                & selected.information_set.eq(panel_name)
                & selected.model.eq(model)
                & selected.origin.isin(g.origin)
            ]
            for (term, predictor, position), term_rows in s.groupby(
                ["term", "predictor", "released_position"]
            ):
                frequencies.append(
                    {
                        "sample": sample,
                        "target": target,
                        "panel": panel_name,
                        "model": model,
                        "term": term,
                        "predictor": predictor,
                        "position": position,
                        "n_selected": len(term_rows),
                        "selection_pct": 100 * len(term_rows) / len(g),
                    }
                )
    for sample, frame in samples(monthly):
        for (target, panel_name), g in frame.groupby(["target", "panel"]):
            for category in ["own_target", "external", "quarterly", "other_monthly"]:
                # Removing a group holds all coefficients and the intercept fixed; no refit.
                loss = (g.actual - g.forecast).pow(2).mean()
                removed_loss = (g.actual - g.forecast + g[category]).pow(2).mean()
                contribution_summary.append(
                    {
                        "sample": sample,
                        "target": target,
                        "panel": panel_name,
                        "group": category,
                        "mean_abs_contribution": g[category].abs().mean(),
                        "MSE": loss,
                        "MSE_without_group_no_refit": removed_loss,
                        "removal_MSE_change": removed_loss - loss,
                    }
                )
        for target in TARGETS:
            for base, expanded in [("M", "ME"), ("MQ", "MQE")]:
                a = frame[frame.target.eq(target) & frame.panel.eq(base)].set_index("origin")
                b = (
                    frame[frame.target.eq(target) & frame.panel.eq(expanded)]
                    .set_index("origin")
                    .loc[a.index]
                )
                s = support[support.model.eq("LASSO") & support.target.eq(target)]
                own_a = s[s.panel.eq(base)].set_index("origin").loc[a.index].own_target.gt(0)
                own_b = s[s.panel.eq(expanded)].set_index("origin").loc[a.index].own_target.gt(0)
                ext_b = s[s.panel.eq(expanded)].set_index("origin").loc[a.index].external.gt(0)
                transitions.append(
                    {
                        "sample": sample,
                        "target": target,
                        "comparison": expanded + " vs " + base,
                        "n": len(a),
                        "own_dropped": int((own_a & ~own_b).sum()),
                        "own_added": int((~own_a & own_b).sum()),
                        "own_dropped_with_external": int((own_a & ~own_b & ext_b).sum()),
                        "own_retained": int((own_a & own_b).sum()),
                        "gain_pct": 100
                        * (
                            1
                            - (b.actual - b.forecast).pow(2).mean()
                            / (a.actual - a.forecast).pow(2).mean()
                        ),
                        "n_train_base": a.n_train.mean(),
                        "n_train_expanded": b.n_train.mean(),
                    }
                )
    core = forecasts[forecasts.target.eq("HICPNEF_EA") & forecasts.origin.isin(origins)]
    scores, channels, core_monthly = [], [], []
    for sample, frame in samples(core):
        for (panel_name, model), g in frame.groupby(["information_set", "model"]):
            if model in ["DFM", "FarmPredict", "DFM_AR", "LASSO"]:
                scores.append(
                    {
                        "sample": sample,
                        "panel": panel_name,
                        "model": model,
                        "MSE": g.error.pow(2).mean(),
                    }
                )
        for base, expanded in [("M", "ME"), ("MQ", "MQE")]:
            losses = {}
            for panel_name in [base, expanded]:
                f = frame[frame.information_set.eq(panel_name)].pivot(
                    index="origin", columns="model", values="forecast"
                )
                actual = (
                    frame[frame.information_set.eq(panel_name)]
                    .groupby("origin")
                    .actual.first()
                    .loc[f.index]
                )
                correction = f.FarmPredict - f.DFM
                dfm_loss = (actual - f.DFM).pow(2)
                fp_loss = (actual - f.FarmPredict).pow(2)
                increment = dfm_loss - fp_loss
                np.testing.assert_allclose(
                    increment, 2 * (actual - f.DFM) * correction - correction**2, atol=1e-12
                )
                losses[panel_name] = (dfm_loss.mean(), fp_loss.mean(), increment.mean())
                if sample == "primary":
                    for month in f.index:
                        core_monthly.append(
                            {
                                "origin": month,
                                "panel": panel_name,
                                "common": f.loc[month, "DFM"],
                                "sparse_correction": correction[month],
                                "residual_loss_gain": increment[month],
                            }
                        )
            a, b = losses[base], losses[expanded]
            factor_change, sparse_change = a[0] - b[0], b[2] - a[2]
            np.testing.assert_allclose(a[1] - b[1], factor_change + sparse_change, atol=1e-12)
            channels.append(
                {
                    "sample": sample,
                    "comparison": expanded + " vs " + base,
                    "FP_MSE_reduction": a[1] - b[1],
                    "DFM_MSE_reduction": factor_change,
                    "sparse_gain_change": sparse_change,
                    "base_sparse_gain": a[2],
                    "expanded_sparse_gain": b[2],
                    "DFM_gain_pct": 100 * (1 - b[0] / a[0]),
                    "FP_gain_pct": 100 * (1 - b[1] / a[1]),
                }
            )
        for panel_name, folder in FIXED.items():
            fixed = pd.read_csv(folder / "forecasts.csv")
            fixed = fixed[fixed.target.eq("HICPNEF_EA") & fixed.origin.isin(frame.origin.unique())]
            assert len(fixed) == frame.origin.nunique()
            scores.append(
                {
                    "sample": sample,
                    "panel": panel_name,
                    "model": "Fixed_factors",
                    "MSE": fixed.error.pow(2).mean(),
                }
            )
    OUT.mkdir(parents=True)
    tables = {
        "direct_monthly": monthly,
        "direct_term_contributions": contributions,
        "support_monthly": support,
        "support_summary": pd.DataFrame(support_summary),
        "term_frequencies": pd.DataFrame(frequencies),
        "contribution_groups": pd.DataFrame(contribution_summary),
        "own_lag_transitions": pd.DataFrame(transitions),
        "core_scores": pd.DataFrame(scores),
        "core_channel_decomposition": pd.DataFrame(channels),
        "core_monthly": pd.DataFrame(core_monthly),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT / (name + ".csv"), index=False)
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    (OUT / "audit.json").write_text(
        json.dumps(
            {
                "sources": hashes,
                "inputs": input_hashes,
                "newline_only_checks": newline_checks,
                "reconstructed_direct_forecasts": len(monthly),
                "max_reconstruction_error": monthly.reconstruction_error.abs().max(),
                "model_refits": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "inflation_external_selection",
        [str(OUT / "audit.json")],
        "Matched direct-LASSO selection and core factor/sparse channel diagnostics",
    )


if __name__ == "__main__":
    main()
