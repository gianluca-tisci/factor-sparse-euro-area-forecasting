"""Explain saved energy-MQ LASSO forecasts without refitting any model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_mq_energy_diagnostics import markdown
from src import _style
from src.forecasting.design import complete_design, dictionary, training_sample
from src.forecasting.experiment import clean_window, prepare

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/energy_lasso_selections")
MONTHS = ["2022-04", "2022-07", "2022-08", "2022-11", "2022-12", "2023-03"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / name for name in ["forecasts.csv", "selected_predictors.csv", "tuning.csv"]]
    sources += [
        Path("config/forecast_paper.yaml"),
        Path("config/forecast_analysis.yaml"),
        Path("config/research_design.yaml"),
        Path("src/forecasting/design.py"),
        Path("src/forecasting/experiment.py"),
    ]
    hashes = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    selected = pd.read_csv(ROOT / "selected_predictors.csv")
    selected = selected[
        (selected.target == "HICPNG_EA")
        & (selected.model == "LASSO")
        & (selected.information_set == "MQ")
    ]
    tuning = pd.read_csv(ROOT / "tuning.csv")
    tuning = tuning[
        (tuning.target == "HICPNG_EA")
        & (tuning.model == "LASSO")
        & (tuning.information_set == "MQ")
    ].set_index("origin")
    forecasts = pd.read_csv(ROOT / "forecasts.csv", low_memory=False)
    forecasts = forecasts[(forecasts.target == "HICPNG_EA") & (forecasts.information_set == "MQ")]
    _, settings, input_hashes, panel, info, lags, _ = prepare("config/forecast_paper.yaml", "MQ")
    assert settings["training_outlier_policy"] == "cleaned_inputs_and_labels"
    events, terms = [], []
    for month in MONTHS:
        origin = pd.Period(month, "M")
        _, cleaned, z, _, _, _ = clean_window(panel, lags, origin)
        features, refs = dictionary(cleaned, info.Frequency, lags, z.index)
        x, y = training_sample(features, cleaned.HICPNG_EA, lags.HICPNG_EA, origin)
        x, y, row = complete_design(x, y, features.loc[origin])
        chosen = selected[selected.origin == month]
        assert len(chosen) == int(tuning.loc[month, "selected"])
        assert len(y) == int(tuning.loc[month, "n_train"])
        prediction = float(y.mean())
        for item in chosen.itertuples():
            reference = refs.loc[origin, item.term]
            assert str(reference) == item.reference_month
            assert reference + int(lags[item.predictor]) <= origin
            contribution = item.coefficient * (row[item.term] - x[item.term].mean())
            prediction += contribution
            terms.append(
                {
                    "origin": month,
                    "term": item.term,
                    "predictor": item.predictor,
                    "reference_month": item.reference_month,
                    "available_month": item.assumed_available_month,
                    "coefficient": item.coefficient,
                    "observed_predictor": row[item.term],
                    "training_predictor_mean": x[item.term].mean(),
                    "centered_contribution": contribution,
                }
            )
        f = forecasts[forecasts.origin == month].set_index("model")
        np.testing.assert_allclose(prediction, f.loc["LASSO", "forecast"], rtol=1e-8, atol=1e-8)
        assert f.target_month.eq(month).all()
        events.append(
            {
                "origin": month,
                "actual": f.actual.iloc[0],
                "LASSO": f.loc["LASSO", "forecast"],
                "DFM": f.loc["DFM", "forecast"],
                "FarmPredict_DFM": f.loc["FarmPredict", "forecast"],
                "training_response_mean": y.mean(),
                "n_train": len(y),
                "selected": len(chosen),
                "alpha": tuning.loc[month, "alpha"],
                "reconstruction_difference": prediction - f.loc["LASSO", "forecast"],
            }
        )
    events, terms = pd.DataFrame(events), pd.DataFrame(terms)
    period = selected[selected.origin.between("2022-01", "2023-12")]
    freq = (
        period.groupby(["term", "predictor", "frequency"])
        .origin.nunique()
        .rename("n_origins")
        .reset_index()
    )
    freq["percent_of_24_origins"] = 100 * freq.n_origins / 24
    freq = freq.sort_values("n_origins", ascending=False)
    nulls = sorted(
        set(pd.period_range("2022-01", "2023-12", freq="M").astype(str)) - set(period.origin)
    )
    assert int(
        tuning.loc[pd.period_range("2022-01", "2023-12", freq="M").astype(str), "selected"]
        .eq(0)
        .sum()
    ) == len(nulls)
    assert period.frequency.eq("M").all()
    events.to_csv(OUT / "episode_forecasts.csv", index=False)
    terms.to_csv(OUT / "selected_terms_and_contributions.csv", index=False)
    freq.to_csv(OUT / "selection_frequencies_2022_2023.csv", index=False)
    text = [
        "# Energy-MQ: direct LASSO selections during the cumulative-MSE reversal",
        (
            "Saved cold paper run, no 2020 scoring months. Focus: six high-loss-difference episodes in 2022-2023. "
            "This is the direct LASSO on transformed observations, not the sparse regression on DFM residuals. "
            "No EM or LASSO estimation was run; coefficients are read from the saved selection file."
        ),
        "## Episode forecasts",
        markdown(events.round(6)),
        "## Selected predictors and centered contributions",
        markdown(terms.round(6)),
        (
            "Prediction = training-response mean + sum of coefficient times (available predictor minus "
            "its mean in the effective complete-row training sample). Contributions are in 100*dlog units. "
            "Every episode reconstructs its saved LASSO forecast within 1e-8. Coefficient size alone is not "
            "a comparable importance measure across differently scaled predictors."
        ),
        (
            "HICPOV is headline HICP inflation; HICPNG is energy HICP inflation; PPINRG is energy producer-price "
            "inflation. M1 means the latest released observation: HICP predictors refer to the previous month; "
            "PPINRG refers to two months earlier under the assumed publication calendar."
        ),
        "## Interpretation",
        (
            "In April 2022 the LASSO selects lagged headline and energy inflation. Its forecast is still positive "
            "and misses the sign of the reversal, but is less positive than DFM/FarmPredict. In July and August "
            "it selects no regressors: modified BIC chooses the intercept-only model, yielding the effective "
            "training mean. This is strong shrinkage rather than successful selection of a leading crisis indicator."
        ),
        (
            "In November, headline inflation and energy producer prices both contribute positively, so the LASSO "
            "again misses the negative sign but makes a smaller error. In December 2022, the October decline in "
            "energy producer prices makes the forecast slightly negative. In March 2023, January energy producer "
            "prices and February consumer energy inflation both contribute negatively. The sparse model remains "
            "far from the realized decline, but is closer than the still-positive DFM-based forecasts."
        ),
        (
            "No quarterly predictor is selected in these six episodes. This does not prove quarterly candidates "
            "have no effect: they can change complete-case training rows and the modified-BIC penalty. "
            "The result reflects conservative predictions plus, in some episodes, lagged energy-price signals. "
            "These are algebraic forecast contributions, not causal effects or an ablation identifying the "
            "predictive value of each series. The episodes were chosen after seeing forecast errors, so this "
            "is an explanatory diagnostic rather than independent validation."
        ),
        "## Selection frequencies across all 24 origins in 2022-2023",
        markdown(freq.round(3)),
        "Intercept-only origins: " + ", ".join(nulls) + ".",
        (
            "Source: saved paper_experiment forecasts, tuning and selections; available transformed observations "
            "reconstructed with the unchanged publication and cleaning functions. No LaTeX edits."
        ),
    ]
    (OUT / "README.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    for path in sources:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.as_posix()]
    (OUT / "provenance.json").write_text(
        json.dumps({"sources": hashes, "reconstruction_inputs": input_hashes}, indent=2),
        encoding="utf-8",
    )
    _style.record_output(
        "energy_lasso_selections",
        [str(OUT / "README.md")],
        "Energy MQ episode diagnostic; saved coefficients only",
    )
    print(
        f"Reconstructed {len(events)} forecasts; archived {len(terms)} terms; null origins: {nulls}"
    )


if __name__ == "__main__":
    main()
