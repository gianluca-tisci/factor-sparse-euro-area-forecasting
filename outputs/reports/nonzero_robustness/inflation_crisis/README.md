# Inflation outside the energy-crisis episode (D-73)

EXPLORATORY evaluation-only sensitivity. Three inflation targets in M, MQ, ME and MQE, separately. The primary experiment remains the reference. 2020 target months are excluded in every sample.

## Fixed exclusion windows

Main sensitivity: January 2021-December 2023, a broad calendar-year envelope for the 2021-2023 energy crisis described in Commission COM(2025)72. Boundary sensitivity: February 2022-December 2023, covering the war escalation and subsequent adjustment. These are operational bounds, not a universal official break date. They were fixed before computing this sensitivity and applied uniformly.

Sources: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A52025DC0072 ; https://www.ecb.europa.eu/press/economic-bulletin/focus/2022/html/ecb.ebbox202204_01~68ef3c3dc6.en.html

## Winners by information set and sample

| sample | target | panel | n | winner | winner_MSE | runner_up | runner_up_MSE | reduction_percent |
|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | M | 120 | DFM_AR | 0.074418 | DFM | 0.075013 | 0.793611 |
| primary | HICPOV_EA | MQ | 120 | DFM | 0.075377 | DFM_AR | 0.075399 | 0.028871 |
| primary | HICPOV_EA | ME | 120 | LASSO | 0.065402 | PCR | 0.078711 | 16.908755 |
| primary | HICPOV_EA | MQE | 120 | LASSO | 0.063384 | AR | 0.084029 | 24.568163 |
| outside_energy_crisis | HICPOV_EA | M | 84 | LASSO | 0.019277 | AR | 0.019792 | 2.600188 |
| outside_energy_crisis | HICPOV_EA | MQ | 84 | AR | 0.019792 | LASSO | 0.020255 | 2.285897 |
| outside_energy_crisis | HICPOV_EA | ME | 84 | LASSO | 0.018884 | AR | 0.019792 | 4.587189 |
| outside_energy_crisis | HICPOV_EA | MQE | 84 | LASSO | 0.018262 | AR | 0.019792 | 7.730239 |
| outside_war_phase | HICPOV_EA | M | 97 | DFM_AR | 0.03633 | DFM | 0.037692 | 3.613731 |
| outside_war_phase | HICPOV_EA | MQ | 97 | DFM_AR | 0.036186 | DFM | 0.037589 | 3.730934 |
| outside_war_phase | HICPOV_EA | ME | 97 | LASSO | 0.032288 | FarmPredict | 0.036205 | 10.820066 |
| outside_war_phase | HICPOV_EA | MQE | 97 | LASSO | 0.031677 | FarmPredict | 0.03493 | 9.311109 |
| primary | HICPNG_EA | M | 120 | AR | 3.118783 | LASSO | 3.293357 | 5.300816 |
| primary | HICPNG_EA | MQ | 120 | AR | 3.118783 | LASSO | 3.134804 | 0.511066 |
| primary | HICPNG_EA | ME | 120 | LASSO | 2.956302 | AR | 3.118783 | 5.209756 |
| primary | HICPNG_EA | MQE | 120 | LASSO | 2.925525 | AR | 3.118783 | 6.196561 |
| outside_energy_crisis | HICPNG_EA | M | 84 | AR | 1.377238 | DFM_AR | 1.407816 | 2.17207 |
| outside_energy_crisis | HICPNG_EA | MQ | 84 | AR | 1.377238 | DFM_AR | 1.406887 | 2.107472 |
| outside_energy_crisis | HICPNG_EA | ME | 84 | AR | 1.377238 | LASSO | 1.441523 | 4.459565 |
| outside_energy_crisis | HICPNG_EA | MQE | 84 | AR | 1.377238 | DFM_AR | 1.43664 | 4.134831 |
| outside_war_phase | HICPNG_EA | M | 97 | DFM_AR | 1.434124 | FarmPredict | 1.475048 | 2.774398 |
| outside_war_phase | HICPNG_EA | MQ | 97 | DFM_AR | 1.434055 | DFM | 1.474649 | 2.752812 |
| outside_war_phase | HICPNG_EA | ME | 97 | AR | 1.519373 | DFM_AR | 1.569677 | 3.20475 |
| outside_war_phase | HICPNG_EA | MQE | 97 | AR | 1.519373 | DFM_AR | 1.531756 | 0.808399 |
| primary | HICPNEF_EA | M | 120 | PCR | 0.017074 | DFM_AR | 0.017126 | 0.303467 |
| primary | HICPNEF_EA | MQ | 120 | DFM_AR | 0.017125 | AR | 0.01836 | 6.72568 |
| primary | HICPNEF_EA | ME | 120 | DFM_AR | 0.014396 | DFM | 0.014423 | 0.192458 |
| primary | HICPNEF_EA | MQE | 120 | DFM_AR | 0.0145 | DFM | 0.014515 | 0.101573 |
| outside_energy_crisis | HICPNEF_EA | M | 84 | DFM_AR | 0.007501 | AR | 0.007541 | 0.528039 |
| outside_energy_crisis | HICPNEF_EA | MQ | 84 | LASSO | 0.007258 | DFM_AR | 0.007476 | 2.918594 |
| outside_energy_crisis | HICPNEF_EA | ME | 84 | DFM_AR | 0.006981 | LASSO | 0.006996 | 0.215029 |
| outside_energy_crisis | HICPNEF_EA | MQE | 84 | LASSO | 0.00684 | DFM_AR | 0.007228 | 5.37049 |
| outside_war_phase | HICPNEF_EA | M | 97 | DFM_AR | 0.014541 | FarmPredict | 0.015023 | 3.203625 |
| outside_war_phase | HICPNEF_EA | MQ | 97 | DFM_AR | 0.014568 | FarmPredict | 0.015154 | 3.863792 |
| outside_war_phase | HICPNEF_EA | ME | 97 | DFM_AR | 0.012816 | DFM | 0.012921 | 0.810363 |
| outside_war_phase | HICPNEF_EA | MQE | 97 | DFM_AR | 0.013105 | DFM | 0.013133 | 0.208536 |

FarmPredict includes both saved identifiers FarmPredict (DFM+LASSO) and PCA_VAR_LASSO (PCA+VAR+LASSO). The PCA variant is estimated only in M and ME. Margins are descriptive, not significance tests. MSE levels across different scoring samples also reflect changed target volatility, not just model quality.

No models are refitted or retuned. Exclusions affect scoring only; crisis data remain in later expanding training samples. This is not a counterfactual economy without the crisis. The check was motivated by observed primary errors and is not an independent validation sample. Cumulated losses resume after each gap using the pre-gap accumulated values; they are not restarted.

Reproduce: python -m scripts.inflation_crisis_sensitivity. No LaTeX or estimator configuration changes.
