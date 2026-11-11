"""Render the saved inflation selection audit and code-generated paper numbers."""

import hashlib
import json

import pandas as pd

from scripts.inflation_external_selection import OUT
from scripts.report_mq_energy_diagnostics import markdown
from src import _style


def main():
    support = pd.read_csv(OUT / "support_summary.csv")
    transitions = pd.read_csv(OUT / "own_lag_transitions.csv")
    frequency = pd.read_csv(OUT / "term_frequencies.csv")
    groups = pd.read_csv(OUT / "contribution_groups.csv")
    channels = pd.read_csv(OUT / "core_channel_decomposition.csv")
    scores = pd.read_csv(OUT / "core_scores.csv")
    primary = support[support["sample"].eq("primary")]
    direct = primary[primary.model.eq("LASSO")]
    core = primary[primary.model.eq("FarmPredict")]
    core_scores = scores.pivot(index=["sample", "panel"], columns="model", values="MSE")
    primary_transitions = transitions[transitions["sample"].eq("primary")]
    macros = {"SelectionOrigins": str(int(direct.n.iloc[0]))}
    for target, label in [
        ("HICPOV_EA", "Headline"),
        ("HICPNG_EA", "Energy"),
        ("HICPNEF_EA", "Core"),
    ]:
        row = primary_transitions[
            primary_transitions.target.eq(target) & primary_transitions.comparison.eq("ME vs M")
        ].iloc[0]
        macros[f"Selection{label}OwnM"] = str(int(row.own_dropped))
        row = primary_transitions[
            primary_transitions.target.eq(target) & primary_transitions.comparison.eq("MQE vs MQ")
        ].iloc[0]
        macros[f"Selection{label}OwnMQ"] = str(int(row.own_dropped))
    for panel in ["M", "ME", "MQ", "MQE"]:
        row = core[core.panel.eq(panel)].iloc[0]
        macros[f"SelectionCoreEmpty{panel}"] = f"{row.empty_pct:.1f}"
        v = core_scores.loc[("primary", panel)]
        macros[f"SelectionCoreSparseGain{panel}"] = f"{100 * (1 - v.FarmPredict / v.DFM):.2f}"
        macros[f"SelectionCoreSparseWorse{panel}"] = f"{100 * (v.FarmPredict / v.DFM - 1):.2f}"
    row = channels[channels["sample"].eq("primary") & channels.comparison.eq("ME vs M")].iloc[0]
    macros["SelectionCoreFactorGain"] = f"{row.DFM_gain_pct:.2f}"
    macros["SelectionCoreFPGain"] = f"{row.FP_gain_pct:.2f}"
    for panel in ["ME", "MQE"]:
        v = core_scores.loc[("primary", panel)]
        base = "M" if panel == "ME" else "MQ"
        a = core_scores.loc[("primary", base)]
        macros[f"SelectionCoreFixedGain{panel}"] = (
            f"{100 * (1 - v.Fixed_factors / a.FarmPredict):.2f}"
        )
    previous = _style.OUT
    try:
        _style.OUT = OUT
        for name, value in macros.items():
            _style.record_number(name, value)
        _style.save_table(
            direct[
                ["target", "panel", "own_target_pct", "external_pct", "empty_pct", "mean_selected"]
            ],
            "tab_102_direct_inflation_selection",
            index=False,
            float_format="%.2f",
            notes="Primary nonzero-IQR matched dates, excluding 2020. Selection percentages count origins with at least one term in the group. Source: saved selections.",
        )
        _style.save_table(
            core_scores.reset_index().round(6),
            "tab_103_core_prediction_channels",
            index=False,
            notes="Matched primary dates and sensitivity excluding 2021--2023; 2020 excluded throughout. Fixed factors retain M for ME and MQ for MQE. Source: saved forecasts and ablations.",
        )
    finally:
        _style.OUT = previous
    document = [
        "# External predictors: direct LASSO and core FarmPredict selections",
        "Current nonzero-IQR run. The primary sample comprises 120 matched forecast origins per inflation target, excluding 2020. The outside-crisis sample additionally excludes 2021--2023 (84 origins). The crisis sample is 2021--2023 (36 origins). Evaluation exclusions do not remove observations from subsequent expanding training windows. No model or tuning parameter was refitted.",
        "## Direct LASSO: do external indicators replace own-target lags?",
        markdown(primary_transitions.round(4)),
        "For headline and energy, own-lag selection disappears in the expanded dictionaries, while external predictors enter on every origin. For core, own lags were already rare in M; its improvement should be described as added external signals, not mainly as replacement of its own persistence. MQ core uses own lags more frequently, which also disappear in MQE. This is a matched selection pattern, not a causal or uniquely identified substitution: correlated candidates, effective training rows and the modified-BIC penalty also change.",
        markdown(direct.round(4)),
        "## Which signals enter, at which released positions?",
        "RETA_006 denotes retail selling-price expectations over the next three months; INDU_006 industry selling-price expectations. MCOILBRENTEU is the monthly Brent price change. External M1 is the latest released observation under the assumed calendar: RETA_006/INDU_006 M1 refer to the origin month, Brent M1 to the previous month. Core RETA_006 M2 refers to the previous month. Thus M1 does not mean the same reference month for all series. Historical release/vintage limitations remain those of the baseline pseudo-real-time exercise.",
    ]
    for model, targets in [
        ("LASSO", ["HICPOV_EA", "HICPNG_EA", "HICPNEF_EA"]),
        ("FarmPredict", ["HICPNEF_EA"]),
    ]:
        for target in targets:
            for panel in ["M", "MQ", "ME", "MQE"]:
                f = frequency[
                    frequency["sample"].eq("primary")
                    & frequency.model.eq(model)
                    & frequency.target.eq(target)
                    & frequency.panel.eq(panel)
                ]
                document += [
                    f"### {model}: {target}, {panel}",
                    markdown(
                        f.sort_values(["n_selected", "term"], ascending=[False, True])
                        .head(12)
                        .round(3)
                    ),
                ]
    document += [
        "## Forecast contributions, beyond support counts",
        "Each direct forecast is reconstructed as the effective training-response mean plus the sum of coefficient times (available predictor minus its effective training mean). All 1,440 reconstructions match saved forecasts within 1e-8; reference and availability dates and effective training counts are checked. Contributions are in the target's transformed units. Coefficient size alone is not an importance ranking.",
        markdown(groups[groups["sample"].eq("primary") & groups.group.eq("external")].round(6)),
        "MSE_without_group_no_refit deletes only that group's centered forecast contribution, retaining the saved coefficients and intercept. Its deterioration shows that external contributions matter algebraically; it is not the forecast from a re-estimated model without E. These deletions do not provide causal effects or independent validation.",
        "## Core: common forecast versus sparse correction",
        markdown(core_scores.reset_index().round(6)),
        markdown(core.round(4)),
        markdown(channels.round(8)),
        "The channel identity is exact: FP MSE reduction = DFM MSE reduction + change in the incremental sparse gain, where sparse gain = DFM MSE minus FP MSE. In both primary comparisons the common-only forecast improves more than the full hybrid, and the sparse gain changes from positive to negative. The same signs hold outside the crisis. Therefore the full ME/MQE improvement is not an improvement in the incremental residual correction; that correction partially offsets the common-forecast improvement. This accounting does not isolate a structural causal effect of changing the factor panel.",
        "The ME residual LASSO is mostly empty. Its few selected terms are IGDSX M4 (non-energy industrial-goods HICP inflation), IRT3M_EACC M1 (short-term interest rate), and RETA_006 M2. MQE also uses long-term interest-rate residuals and retail-price expectations, but the aggregate correction still worsens its expanded DFM. Empty support means an intercept-only residual forecast, not necessarily an exactly zero correction.",
        "Fixed-factor external residuals improve the original M/MQ hybrids, including outside the crisis. Hence external residuals can be useful with the old factor representation. They still do not beat the expanded-panel DFM in the primary sample, and outside the crisis do not beat the expanded hybrid. The full factor and residual channels are not additive interventions: residuals, supports, penalties and complete-case samples change together. The supported conclusion is that the best core forecasts use an improved common forecast, not that external residual signals are universally useless.",
        "## Outside-crisis sensitivity",
        markdown(transitions[transitions["sample"].eq("outside_crisis")].round(4)),
        "The direct-LASSO improvement and absence of own-target lags persist outside the crisis for each target in both comparisons. This does not establish significance; no additional hypothesis tests were run.",
        "## Reproduction and files",
        "Run `python -m scripts.inflation_external_selection` in a fresh output directory, then `python -m scripts.inflation_external_selection_report`. Full term frequencies (including released positions), per-origin supports, centered contributions, channel decompositions and sensitivity results are in the CSVs beside this report. `audit.json` records source hashes, accepted LF/CRLF-only differences and maximum forecast reconstruction error. Original forecasts, model settings and estimators remain unchanged.",
    ]
    (OUT / "analysis.md").write_text("\n\n".join(document) + "\n", encoding="utf-8")
    paths = list(OUT.glob("*.csv")) + [OUT / "audit.json"]
    (OUT / "report_audit.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2),
        encoding="utf-8",
    )
    _style.record_output(
        "inflation_external_selection_report",
        [str(OUT / "analysis.md"), str(OUT / "numbers.tex")],
        "Results: external selection and core prediction channels",
    )


if __name__ == "__main__":
    main()
