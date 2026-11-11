# External variables outside the crisis: fixed versus expanded factors

EXPLORATORY evaluation-only sensitivity for headline, energy and core in ME/MQE. Same broad exclusion January 2021-December 2023 and narrow exclusion February 2022-December 2023 as codex/inflation-crisis-sensitivity. Every sample also excludes 2020 target months. Primary evaluation remains the reference; no changes to training or model fitting.

## Fixed-factor ablation versus direct external-panel LASSO

| target | sample | panel | numerator | benchmark | n | MSE | benchmark_MSE | final_ratio | reduction_percent |
|---|---|---|---|---|---|---|---|---|---|
| HICPOV_EA | primary | ME | ME:Fixed_factor | ME:LASSO | 120 | 0.075275 | 0.065402 | 1.150963 | -15.096259 |
| HICPOV_EA | primary | MQE | MQE:Fixed_factor | MQE:LASSO | 120 | 0.075487 | 0.063384 | 1.190936 | -19.093585 |
| HICPOV_EA | outside_energy_crisis | ME | ME:Fixed_factor | ME:LASSO | 84 | 0.022042 | 0.018884 | 1.167248 | -16.724787 |
| HICPOV_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | MQE:LASSO | 84 | 0.022003 | 0.018262 | 1.204867 | -20.486669 |
| HICPOV_EA | outside_war_phase | ME | ME:Fixed_factor | ME:LASSO | 97 | 0.037492 | 0.032288 | 1.161176 | -16.117621 |
| HICPOV_EA | outside_war_phase | MQE | MQE:Fixed_factor | MQE:LASSO | 97 | 0.037298 | 0.031677 | 1.177445 | -17.744465 |
| HICPNG_EA | primary | ME | ME:Fixed_factor | ME:LASSO | 120 | 3.84023 | 2.956302 | 1.298998 | -29.899788 |
| HICPNG_EA | primary | MQE | MQE:Fixed_factor | MQE:LASSO | 120 | 3.841062 | 2.925525 | 1.312948 | -31.294767 |
| HICPNG_EA | outside_energy_crisis | ME | ME:Fixed_factor | ME:LASSO | 84 | 1.432362 | 1.441523 | 0.993645 | 0.635535 |
| HICPNG_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | MQE:LASSO | 84 | 1.429163 | 1.443971 | 0.989745 | 1.025532 |
| HICPNG_EA | outside_war_phase | ME | ME:Fixed_factor | ME:LASSO | 97 | 1.48121 | 1.650395 | 0.897489 | 10.251138 |
| HICPNG_EA | outside_war_phase | MQE | MQE:Fixed_factor | MQE:LASSO | 97 | 1.482926 | 1.686912 | 0.879077 | 12.092306 |
| HICPNEF_EA | primary | ME | ME:Fixed_factor | ME:LASSO | 120 | 0.016163 | 0.018572 | 0.870287 | 12.971326 |
| HICPNEF_EA | primary | MQE | MQE:Fixed_factor | MQE:LASSO | 120 | 0.015782 | 0.017783 | 0.887503 | 11.249744 |
| HICPNEF_EA | outside_energy_crisis | ME | ME:Fixed_factor | ME:LASSO | 84 | 0.007286 | 0.006996 | 1.041581 | -4.158121 |
| HICPNEF_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | MQE:LASSO | 84 | 0.007382 | 0.00684 | 1.079349 | -7.934899 |
| HICPNEF_EA | outside_war_phase | ME | ME:Fixed_factor | ME:LASSO | 97 | 0.014489 | 0.014665 | 0.988022 | 1.197792 |
| HICPNEF_EA | outside_war_phase | MQE | MQE:Fixed_factor | MQE:LASSO | 97 | 0.014619 | 0.014376 | 1.016901 | -1.690112 |

Positive reductions favor the ablation. MSE changes between samples also reflect different target volatility; relative performance on matched dates is the relevant comparison. Original and expanded model menus, including PCA + VAR + LASSO in ME, are ranked in winners.csv. FarmPredict is a family including both PCA and DFM variants; the two plotted numerators concern only DFM variants.

The exclusions change scoring only. Crisis observations remain in subsequent expanding training windows. This is not a counterfactual economy without crisis, or an independent validation sample. The broad window is an operational envelope, not a uniform official break date for every target. No significance or causal claim follows from descriptive MSE differences.

Both panels use the same dates. The fixed-factor case preserves the original M or MQ common forecast and projects E on its saved factors. It changes sparse candidates and potentially complete training rows. All plotted endpoints are checked against independently computed full retained-sample MSE ratios.

Reproduce: python -m scripts.external_crisis_sensitivity. No model refits or LaTeX edits.
