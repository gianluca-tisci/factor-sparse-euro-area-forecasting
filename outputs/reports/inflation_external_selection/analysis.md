# External predictors: direct LASSO and core FarmPredict selections

Current nonzero-IQR run. The primary sample comprises 120 matched forecast origins per inflation target, excluding 2020. The outside-crisis sample additionally excludes 2021--2023 (84 origins). The crisis sample is 2021--2023 (36 origins). Evaluation exclusions do not remove observations from subsequent expanding training windows. No model or tuning parameter was refitted.

## Direct LASSO: do external indicators replace own-target lags?

| sample | target | comparison | n | own_dropped | own_added | own_dropped_with_external | own_retained | gain_pct | n_train_base | n_train_expanded |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | ME vs M | 120 | 101 | 0 | 101 | 0 | 22.5029 | 227.825 | 225.4333 |
| primary | HICPOV_EA | MQE vs MQ | 120 | 97 | 0 | 97 | 0 | 25.4183 | 224.825 | 222.4333 |
| primary | HICPNG_EA | ME vs M | 120 | 81 | 0 | 81 | 0 | 10.2344 | 227.825 | 225.4333 |
| primary | HICPNG_EA | MQE vs MQ | 120 | 87 | 0 | 87 | 0 | 6.676 | 224.825 | 222.4333 |
| primary | HICPNEF_EA | ME vs M | 120 | 2 | 0 | 2 | 0 | 22.9482 | 227.825 | 225.4333 |
| primary | HICPNEF_EA | MQE vs MQ | 120 | 32 | 0 | 32 | 0 | 22.8814 | 224.825 | 222.4333 |

For headline and energy, own-lag selection disappears in the expanded dictionaries, while external predictors enter on every origin. For core, own lags were already rare in M; its improvement should be described as added external signals, not mainly as replacement of its own persistence. MQ core uses own lags more frequently, which also disappear in MQE. This is a matched selection pattern, not a causal or uniquely identified substitution: correlated candidates, effective training rows and the modified-BIC penalty also change.

| sample | target | panel | model | n | empty_pct | mean_selected | own_target_pct | external_pct | quarterly_pct | other_monthly_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | M | LASSO | 120 | 3.3333 | 2.9417 | 1.6667 | 0.0 | 0.0 | 96.6667 |
| primary | HICPNEF_EA | ME | LASSO | 120 | 0.0 | 4.525 | 0.0 | 64.1667 | 0.0 | 100.0 |
| primary | HICPNEF_EA | MQ | LASSO | 120 | 9.1667 | 4.8333 | 26.6667 | 0.0 | 42.5 | 87.5 |
| primary | HICPNEF_EA | MQE | LASSO | 120 | 0.8333 | 4.8083 | 0.0 | 60.0 | 50.0 | 98.3333 |
| primary | HICPNG_EA | M | LASSO | 120 | 17.5 | 2.0167 | 67.5 | 0.0 | 0.0 | 73.3333 |
| primary | HICPNG_EA | ME | LASSO | 120 | 0.0 | 1.9 | 0.0 | 100.0 | 0.0 | 36.6667 |
| primary | HICPNG_EA | MQ | LASSO | 120 | 20.0 | 1.4417 | 72.5 | 0.0 | 0.0 | 60.8333 |
| primary | HICPNG_EA | MQE | LASSO | 120 | 0.0 | 1.9083 | 0.0 | 100.0 | 0.0 | 35.0 |
| primary | HICPOV_EA | M | LASSO | 120 | 14.1667 | 3.5833 | 84.1667 | 0.0 | 0.0 | 84.1667 |
| primary | HICPOV_EA | ME | LASSO | 120 | 0.0 | 2.825 | 0.0 | 100.0 | 0.0 | 20.8333 |
| primary | HICPOV_EA | MQ | LASSO | 120 | 17.5 | 3.15 | 80.8333 | 0.0 | 20.8333 | 45.8333 |
| primary | HICPOV_EA | MQE | LASSO | 120 | 0.0 | 2.8667 | 0.0 | 100.0 | 5.0 | 14.1667 |

## Which signals enter, at which released positions?

RETA_006 denotes retail selling-price expectations over the next three months; INDU_006 industry selling-price expectations. MCOILBRENTEU is the monthly Brent price change. External M1 is the latest released observation under the assumed calendar: RETA_006/INDU_006 M1 refer to the origin month, Brent M1 to the previous month. Core RETA_006 M2 refers to the previous month. Thus M1 does not mean the same reference month for all series. Historical release/vintage limitations remain those of the baseline pseudo-real-time exercise.

### LASSO: HICPOV_EA, M

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | M | LASSO | HICPOV_EA__M1 | HICPOV_EA | 1 | 101 | 84.167 |
| primary | HICPOV_EA | M | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 101 | 84.167 |
| primary | HICPOV_EA | M | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 43 | 35.833 |
| primary | HICPOV_EA | M | LASSO | PPIDCOG_EA__M3 | PPIDCOG_EA | 3 | 42 | 35.0 |
| primary | HICPOV_EA | M | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 38 | 31.667 |
| primary | HICPOV_EA | M | LASSO | PPIING_EA__M4 | PPIING_EA | 4 | 37 | 30.833 |
| primary | HICPOV_EA | M | LASSO | HICPSV_EA__M1 | HICPSV_EA | 1 | 31 | 25.833 |
| primary | HICPOV_EA | M | LASSO | PPIING_EA__M3 | PPIING_EA | 3 | 11 | 9.167 |
| primary | HICPOV_EA | M | LASSO | PPINDCOG_EA__M1 | PPINDCOG_EA | 1 | 8 | 6.667 |
| primary | HICPOV_EA | M | LASSO | PPICOG_EA__M3 | PPICOG_EA | 3 | 7 | 5.833 |
| primary | HICPOV_EA | M | LASSO | PPIDCOG_EA__M4 | PPIDCOG_EA | 4 | 3 | 2.5 |
| primary | HICPOV_EA | M | LASSO | PPIING_EA__M1 | PPIING_EA | 1 | 3 | 2.5 |

### LASSO: HICPOV_EA, MQ

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | MQ | LASSO | HICPOV_EA__M1 | HICPOV_EA | 1 | 97 | 80.833 |
| primary | HICPOV_EA | MQ | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 53 | 44.167 |
| primary | HICPOV_EA | MQ | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 44 | 36.667 |
| primary | HICPOV_EA | MQ | LASSO | PPIDCOG_EA__M3 | PPIDCOG_EA | 3 | 42 | 35.0 |
| primary | HICPOV_EA | MQ | LASSO | PPIING_EA__M4 | PPIING_EA | 4 | 37 | 30.833 |
| primary | HICPOV_EA | MQ | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 37 | 30.833 |
| primary | HICPOV_EA | MQ | LASSO | ESC_EA__Q1 | ESC_EA | 1 | 25 | 20.833 |
| primary | HICPOV_EA | MQ | LASSO | HICPSV_EA__M1 | HICPSV_EA | 1 | 24 | 20.0 |
| primary | HICPOV_EA | MQ | LASSO | PPIING_EA__M3 | PPIING_EA | 3 | 6 | 5.0 |
| primary | HICPOV_EA | MQ | LASSO | HICPNG_EA__M1 | HICPNG_EA | 1 | 3 | 2.5 |
| primary | HICPOV_EA | MQ | LASSO | PPICOG_EA__M3 | PPICOG_EA | 3 | 3 | 2.5 |
| primary | HICPOV_EA | MQ | LASSO | PPIING_EA__M1 | PPIING_EA | 1 | 3 | 2.5 |

### LASSO: HICPOV_EA, ME

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | ME | LASSO | RETA_006__M1 | RETA_006 | 1 | 120 | 100.0 |
| primary | HICPOV_EA | ME | LASSO | MCOILBRENTEU__M1 | MCOILBRENTEU | 1 | 89 | 74.167 |
| primary | HICPOV_EA | ME | LASSO | INDU_006__M1 | INDU_006 | 1 | 30 | 25.0 |
| primary | HICPOV_EA | ME | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 18 | 15.0 |
| primary | HICPOV_EA | ME | LASSO | PCOALAUUSDM__M2 | PCOALAUUSDM | 2 | 15 | 12.5 |
| primary | HICPOV_EA | ME | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 14 | 11.667 |
| primary | HICPOV_EA | ME | LASSO | PNGASEUUSDM__M2 | PNGASEUUSDM | 2 | 13 | 10.833 |
| primary | HICPOV_EA | ME | LASSO | WB_FOOD__M3 | WB_FOOD | 3 | 8 | 6.667 |
| primary | HICPOV_EA | ME | LASSO | IGDSX__M4 | IGDSX | 4 | 7 | 5.833 |
| primary | HICPOV_EA | ME | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 7 | 5.833 |
| primary | HICPOV_EA | ME | LASSO | PPINDCOG_EA__M1 | PPINDCOG_EA | 1 | 5 | 4.167 |
| primary | HICPOV_EA | ME | LASSO | PSOILUSDM__M4 | PSOILUSDM | 4 | 3 | 2.5 |

### LASSO: HICPOV_EA, MQE

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPOV_EA | MQE | LASSO | RETA_006__M1 | RETA_006 | 1 | 120 | 100.0 |
| primary | HICPOV_EA | MQE | LASSO | MCOILBRENTEU__M1 | MCOILBRENTEU | 1 | 84 | 70.0 |
| primary | HICPOV_EA | MQE | LASSO | WB_FOOD__M3 | WB_FOOD | 3 | 49 | 40.833 |
| primary | HICPOV_EA | MQE | LASSO | INDU_006__M1 | INDU_006 | 1 | 32 | 26.667 |
| primary | HICPOV_EA | MQE | LASSO | PCOALAUUSDM__M2 | PCOALAUUSDM | 2 | 22 | 18.333 |
| primary | HICPOV_EA | MQE | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 11 | 9.167 |
| primary | HICPOV_EA | MQE | LASSO | HFCE_EA__Q1 | HFCE_EA | 1 | 5 | 4.167 |
| primary | HICPOV_EA | MQE | LASSO | PNGASEUUSDM__M2 | PNGASEUUSDM | 2 | 5 | 4.167 |
| primary | HICPOV_EA | MQE | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 4 | 3.333 |
| primary | HICPOV_EA | MQE | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 3 | 2.5 |
| primary | HICPOV_EA | MQE | LASSO | PPINDCOG_EA__M1 | PPINDCOG_EA | 1 | 3 | 2.5 |
| primary | HICPOV_EA | MQE | LASSO | PPIDCOG_EA__M3 | PPIDCOG_EA | 3 | 2 | 1.667 |

### LASSO: HICPNG_EA, M

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNG_EA | M | LASSO | HICPNG_EA__M1 | HICPNG_EA | 1 | 81 | 67.5 |
| primary | HICPNG_EA | M | LASSO | SHIX_EA__M1 | SHIX_EA | 1 | 50 | 41.667 |
| primary | HICPNG_EA | M | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 38 | 31.667 |
| primary | HICPNG_EA | M | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 35 | 29.167 |
| primary | HICPNG_EA | M | LASSO | ICONFIX_EA__M1 | ICONFIX_EA | 1 | 13 | 10.833 |
| primary | HICPNG_EA | M | LASSO | HICPOV_EA__M1 | HICPOV_EA | 1 | 8 | 6.667 |
| primary | HICPNG_EA | M | LASSO | PPIING_EA__M1 | PPIING_EA | 1 | 8 | 6.667 |
| primary | HICPNG_EA | M | LASSO | CURR_EACC__M3 | CURR_EACC | 3 | 7 | 5.833 |
| primary | HICPNG_EA | M | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 1 | 0.833 |
| primary | HICPNG_EA | M | LASSO | PPIDCOG_EA__M3 | PPIDCOG_EA | 3 | 1 | 0.833 |

### LASSO: HICPNG_EA, MQ

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNG_EA | MQ | LASSO | HICPNG_EA__M1 | HICPNG_EA | 1 | 87 | 72.5 |
| primary | HICPNG_EA | MQ | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 38 | 31.667 |
| primary | HICPNG_EA | MQ | LASSO | SHIX_EA__M1 | SHIX_EA | 1 | 34 | 28.333 |
| primary | HICPNG_EA | MQ | LASSO | HICPOV_EA__M1 | HICPOV_EA | 1 | 9 | 7.5 |
| primary | HICPNG_EA | MQ | LASSO | ICONFIX_EA__M1 | ICONFIX_EA | 1 | 3 | 2.5 |
| primary | HICPNG_EA | MQ | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 1 | 0.833 |
| primary | HICPNG_EA | MQ | LASSO | PPIDCOG_EA__M3 | PPIDCOG_EA | 3 | 1 | 0.833 |

### LASSO: HICPNG_EA, ME

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNG_EA | ME | LASSO | MCOILBRENTEU__M1 | MCOILBRENTEU | 1 | 120 | 100.0 |
| primary | HICPNG_EA | ME | LASSO | INDU_006__M1 | INDU_006 | 1 | 54 | 45.0 |
| primary | HICPNG_EA | ME | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 38 | 31.667 |
| primary | HICPNG_EA | ME | LASSO | PNGASEUUSDM__M2 | PNGASEUUSDM | 2 | 6 | 5.0 |
| primary | HICPNG_EA | ME | LASSO | SHIX_EA__M1 | SHIX_EA | 1 | 5 | 4.167 |
| primary | HICPNG_EA | ME | LASSO | WB_FOOD__M3 | WB_FOOD | 3 | 2 | 1.667 |
| primary | HICPNG_EA | ME | LASSO | PCOTTINDUSDM__M4 | PCOTTINDUSDM | 4 | 1 | 0.833 |
| primary | HICPNG_EA | ME | LASSO | PPICOG_EA__M1 | PPICOG_EA | 1 | 1 | 0.833 |
| primary | HICPNG_EA | ME | LASSO | RETA_006__M1 | RETA_006 | 1 | 1 | 0.833 |

### LASSO: HICPNG_EA, MQE

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNG_EA | MQE | LASSO | MCOILBRENTEU__M1 | MCOILBRENTEU | 1 | 120 | 100.0 |
| primary | HICPNG_EA | MQE | LASSO | INDU_006__M1 | INDU_006 | 1 | 53 | 44.167 |
| primary | HICPNG_EA | MQE | LASSO | PPINRG_EA__M1 | PPINRG_EA | 1 | 38 | 31.667 |
| primary | HICPNG_EA | MQE | LASSO | PNGASEUUSDM__M2 | PNGASEUUSDM | 2 | 7 | 5.833 |
| primary | HICPNG_EA | MQE | LASSO | RETA_006__M1 | RETA_006 | 1 | 5 | 4.167 |
| primary | HICPNG_EA | MQE | LASSO | SHIX_EA__M1 | SHIX_EA | 1 | 4 | 3.333 |
| primary | HICPNG_EA | MQE | LASSO | PCOTTINDUSDM__M4 | PCOTTINDUSDM | 4 | 2 | 1.667 |

### LASSO: HICPNEF_EA, M

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | M | LASSO | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 72 | 60.0 |
| primary | HICPNEF_EA | M | LASSO | PPICOG_EA__M4 | PPICOG_EA | 4 | 41 | 34.167 |
| primary | HICPNEF_EA | M | LASSO | PPICOG_EA__M3 | PPICOG_EA | 3 | 40 | 33.333 |
| primary | HICPNEF_EA | M | LASSO | PPIDCOG_EA__M4 | PPIDCOG_EA | 4 | 23 | 19.167 |
| primary | HICPNEF_EA | M | LASSO | HICPSV_EA__M1 | HICPSV_EA | 1 | 20 | 16.667 |
| primary | HICPNEF_EA | M | LASSO | TRNDCOG_EA__M1 | TRNDCOG_EA | 1 | 16 | 13.333 |
| primary | HICPNEF_EA | M | LASSO | TRNNDCOG_EA__M2 | TRNNDCOG_EA | 2 | 15 | 12.5 |
| primary | HICPNEF_EA | M | LASSO | UNEU25_EA__M3 | UNEU25_EA | 3 | 15 | 12.5 |
| primary | HICPNEF_EA | M | LASSO | LTIRT_EACC__M2 | LTIRT_EACC | 2 | 14 | 11.667 |
| primary | HICPNEF_EA | M | LASSO | IPNDCOG_EA__M3 | IPNDCOG_EA | 3 | 13 | 10.833 |
| primary | HICPNEF_EA | M | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 12 | 10.0 |
| primary | HICPNEF_EA | M | LASSO | PPIDCOG_EA__M2 | PPIDCOG_EA | 2 | 12 | 10.0 |

### LASSO: HICPNEF_EA, MQ

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | MQ | LASSO | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 62 | 51.667 |
| primary | HICPNEF_EA | MQ | LASSO | ESC_EA__Q1 | ESC_EA | 1 | 44 | 36.667 |
| primary | HICPNEF_EA | MQ | LASSO | PPICOG_EA__M4 | PPICOG_EA | 4 | 41 | 34.167 |
| primary | HICPNEF_EA | MQ | LASSO | PPICOG_EA__M3 | PPICOG_EA | 3 | 40 | 33.333 |
| primary | HICPNEF_EA | MQ | LASSO | PPIDCOG_EA__M1 | PPIDCOG_EA | 1 | 39 | 32.5 |
| primary | HICPNEF_EA | MQ | LASSO | PPIDCOG_EA__M2 | PPIDCOG_EA | 2 | 37 | 30.833 |
| primary | HICPNEF_EA | MQ | LASSO | PPIDCOG_EA__M4 | PPIDCOG_EA | 4 | 37 | 30.833 |
| primary | HICPNEF_EA | MQ | LASSO | IPNDCOG_EA__M3 | IPNDCOG_EA | 3 | 34 | 28.333 |
| primary | HICPNEF_EA | MQ | LASSO | HICPNEF_EA__M3 | HICPNEF_EA | 3 | 32 | 26.667 |
| primary | HICPNEF_EA | MQ | LASSO | HICPNEF_EA__M4 | HICPNEF_EA | 4 | 32 | 26.667 |
| primary | HICPNEF_EA | MQ | LASSO | TRNDCOG_EA__M1 | TRNDCOG_EA | 1 | 29 | 24.167 |
| primary | HICPNEF_EA | MQ | LASSO | HICPSV_EA__M1 | HICPSV_EA | 1 | 27 | 22.5 |

### LASSO: HICPNEF_EA, ME

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | ME | LASSO | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 71 | 59.167 |
| primary | HICPNEF_EA | ME | LASSO | TRNDCOG_EA__M1 | TRNDCOG_EA | 1 | 56 | 46.667 |
| primary | HICPNEF_EA | ME | LASSO | RETA_006__M2 | RETA_006 | 2 | 54 | 45.0 |
| primary | HICPNEF_EA | ME | LASSO | IPNDCOG_EA__M3 | IPNDCOG_EA | 3 | 48 | 40.0 |
| primary | HICPNEF_EA | ME | LASSO | PPICOG_EA__M4 | PPICOG_EA | 4 | 40 | 33.333 |
| primary | HICPNEF_EA | ME | LASSO | RETA_006__M4 | RETA_006 | 4 | 38 | 31.667 |
| primary | HICPNEF_EA | ME | LASSO | IPDCOG_EA__M2 | IPDCOG_EA | 2 | 37 | 30.833 |
| primary | HICPNEF_EA | ME | LASSO | RETA_006__M3 | RETA_006 | 3 | 34 | 28.333 |
| primary | HICPNEF_EA | ME | LASSO | TRNNDCOG_EA__M2 | TRNNDCOG_EA | 2 | 24 | 20.0 |
| primary | HICPNEF_EA | ME | LASSO | TRNCAG_EA__M2 | TRNCAG_EA | 2 | 22 | 18.333 |
| primary | HICPNEF_EA | ME | LASSO | PPICAG_EA__M2 | PPICAG_EA | 2 | 21 | 17.5 |
| primary | HICPNEF_EA | ME | LASSO | IGDSX__M4 | IGDSX | 4 | 19 | 15.833 |

### LASSO: HICPNEF_EA, MQE

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | MQE | LASSO | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 70 | 58.333 |
| primary | HICPNEF_EA | MQE | LASSO | RETA_006__M2 | RETA_006 | 2 | 54 | 45.0 |
| primary | HICPNEF_EA | MQE | LASSO | IPDCOG_EA__M2 | IPDCOG_EA | 2 | 51 | 42.5 |
| primary | HICPNEF_EA | MQE | LASSO | ESC_EA__Q1 | ESC_EA | 1 | 50 | 41.667 |
| primary | HICPNEF_EA | MQE | LASSO | IPNDCOG_EA__M3 | IPNDCOG_EA | 3 | 47 | 39.167 |
| primary | HICPNEF_EA | MQE | LASSO | TRNDCOG_EA__M1 | TRNDCOG_EA | 1 | 42 | 35.0 |
| primary | HICPNEF_EA | MQE | LASSO | HFCE_EA__Q1 | HFCE_EA | 1 | 41 | 34.167 |
| primary | HICPNEF_EA | MQE | LASSO | PPICOG_EA__M4 | PPICOG_EA | 4 | 39 | 32.5 |
| primary | HICPNEF_EA | MQE | LASSO | RETA_006__M4 | RETA_006 | 4 | 39 | 32.5 |
| primary | HICPNEF_EA | MQE | LASSO | TRNNDCOG_EA__M2 | TRNNDCOG_EA | 2 | 23 | 19.167 |
| primary | HICPNEF_EA | MQE | LASSO | RETA_006__M3 | RETA_006 | 3 | 22 | 18.333 |
| primary | HICPNEF_EA | MQE | LASSO | TRNCAG_EA__M2 | TRNCAG_EA | 2 | 18 | 15.0 |

### FarmPredict: HICPNEF_EA, M

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | M | FarmPredict | LTIRT_EACC__M1 | LTIRT_EACC | 1 | 47 | 39.167 |
| primary | HICPNEF_EA | M | FarmPredict | KCONFIX_EA__M1 | KCONFIX_EA | 1 | 29 | 24.167 |
| primary | HICPNEF_EA | M | FarmPredict | HICPIN_EA__M1 | HICPIN_EA | 1 | 15 | 12.5 |
| primary | HICPNEF_EA | M | FarmPredict | PPICOG_EA__M4 | PPICOG_EA | 4 | 15 | 12.5 |
| primary | HICPNEF_EA | M | FarmPredict | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 14 | 11.667 |
| primary | HICPNEF_EA | M | FarmPredict | CCI_EA__M4 | CCI_EA | 4 | 13 | 10.833 |
| primary | HICPNEF_EA | M | FarmPredict | CCONFIX_EA__M2 | CCONFIX_EA | 2 | 7 | 5.833 |
| primary | HICPNEF_EA | M | FarmPredict | IRT3M_EACC__M4 | IRT3M_EACC | 4 | 4 | 3.333 |
| primary | HICPNEF_EA | M | FarmPredict | CCONFIX_EA__M1 | CCONFIX_EA | 1 | 1 | 0.833 |
| primary | HICPNEF_EA | M | FarmPredict | HICPOV_EA__M1 | HICPOV_EA | 1 | 1 | 0.833 |

### FarmPredict: HICPNEF_EA, MQ

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | MQ | FarmPredict | LTIRT_EACC__M1 | LTIRT_EACC | 1 | 47 | 39.167 |
| primary | HICPNEF_EA | MQ | FarmPredict | KCONFIX_EA__M1 | KCONFIX_EA | 1 | 24 | 20.0 |
| primary | HICPNEF_EA | MQ | FarmPredict | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 17 | 14.167 |
| primary | HICPNEF_EA | MQ | FarmPredict | HICPIN_EA__M1 | HICPIN_EA | 1 | 11 | 9.167 |
| primary | HICPNEF_EA | MQ | FarmPredict | PPICOG_EA__M4 | PPICOG_EA | 4 | 11 | 9.167 |
| primary | HICPNEF_EA | MQ | FarmPredict | CCONFIX_EA__M2 | CCONFIX_EA | 2 | 6 | 5.0 |
| primary | HICPNEF_EA | MQ | FarmPredict | CCI_EA__M4 | CCI_EA | 4 | 5 | 4.167 |
| primary | HICPNEF_EA | MQ | FarmPredict | IRT3M_EACC__M4 | IRT3M_EACC | 4 | 1 | 0.833 |
| primary | HICPNEF_EA | MQ | FarmPredict | M2_EACC__M1 | M2_EACC | 1 | 1 | 0.833 |

### FarmPredict: HICPNEF_EA, ME

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | ME | FarmPredict | IGDSX__M4 | IGDSX | 4 | 3 | 2.5 |
| primary | HICPNEF_EA | ME | FarmPredict | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 3 | 2.5 |
| primary | HICPNEF_EA | ME | FarmPredict | RETA_006__M2 | RETA_006 | 2 | 1 | 0.833 |

### FarmPredict: HICPNEF_EA, MQE

| sample | target | panel | model | term | predictor | position | n_selected | selection_pct |
|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | MQE | FarmPredict | LTIRT_EACC__M1 | LTIRT_EACC | 1 | 12 | 10.0 |
| primary | HICPNEF_EA | MQE | FarmPredict | RETA_006__M2 | RETA_006 | 2 | 10 | 8.333 |
| primary | HICPNEF_EA | MQE | FarmPredict | IRT3M_EACC__M1 | IRT3M_EACC | 1 | 5 | 4.167 |
| primary | HICPNEF_EA | MQE | FarmPredict | IGDSX__M4 | IGDSX | 4 | 3 | 2.5 |
| primary | HICPNEF_EA | MQE | FarmPredict | RETA_006__M1 | RETA_006 | 1 | 2 | 1.667 |
| primary | HICPNEF_EA | MQE | FarmPredict | PPINDCOG_EA__M4 | PPINDCOG_EA | 4 | 1 | 0.833 |

## Forecast contributions, beyond support counts

Each direct forecast is reconstructed as the effective training-response mean plus the sum of coefficient times (available predictor minus its effective training mean). All 1,440 reconstructions match saved forecasts within 1e-8; reference and availability dates and effective training counts are checked. Contributions are in the target's transformed units. Coefficient size alone is not an importance ranking.

| sample | target | panel | group | mean_abs_contribution | MSE | MSE_without_group_no_refit | removal_MSE_change |
|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | M | external | 0.0 | 0.024104 | 0.024104 | 0.0 |
| primary | HICPNEF_EA | ME | external | 0.03086 | 0.018572 | 0.027358 | 0.008786 |
| primary | HICPNEF_EA | MQ | external | 0.0 | 0.023059 | 0.023059 | 0.0 |
| primary | HICPNEF_EA | MQE | external | 0.029847 | 0.017783 | 0.026696 | 0.008913 |
| primary | HICPNG_EA | M | external | 0.0 | 3.293357 | 3.293357 | 0.0 |
| primary | HICPNG_EA | ME | external | 0.309832 | 2.956302 | 3.422147 | 0.465845 |
| primary | HICPNG_EA | MQ | external | 0.0 | 3.134804 | 3.134804 | 0.0 |
| primary | HICPNG_EA | MQE | external | 0.308308 | 2.925525 | 3.428817 | 0.503292 |
| primary | HICPOV_EA | M | external | 0.0 | 0.084392 | 0.084392 | 0.0 |
| primary | HICPOV_EA | ME | external | 0.072205 | 0.065402 | 0.096353 | 0.030952 |
| primary | HICPOV_EA | MQ | external | 0.0 | 0.084986 | 0.084986 | 0.0 |
| primary | HICPOV_EA | MQE | external | 0.076711 | 0.063384 | 0.097856 | 0.034471 |

MSE_without_group_no_refit deletes only that group's centered forecast contribution, retaining the saved coefficients and intercept. Its deterioration shows that external contributions matter algebraically; it is not the forecast from a re-estimated model without E. These deletions do not provide causal effects or independent validation.

## Core: common forecast versus sparse correction

| sample | panel | DFM | DFM_AR | FarmPredict | Fixed_factors | LASSO |
|---|---|---|---|---|---|---|
| crisis | M | 0.045923 | 0.039584 | 0.043918 | nan | 0.061994 |
| crisis | ME | 0.031372 | 0.031698 | 0.032494 | 0.036875 | 0.045584 |
| crisis | MQ | 0.046513 | 0.039639 | 0.043839 | nan | 0.059928 |
| crisis | MQE | 0.031251 | 0.03147 | 0.03164 | 0.035382 | 0.043317 |
| outside_crisis | M | 0.008581 | 0.007501 | 0.0081 | nan | 0.007865 |
| outside_crisis | ME | 0.00716 | 0.006981 | 0.007236 | 0.007286 | 0.006996 |
| outside_crisis | MQ | 0.008647 | 0.007476 | 0.00822 | nan | 0.007258 |
| outside_crisis | MQE | 0.007343 | 0.007228 | 0.007365 | 0.007382 | 0.00684 |
| primary | M | 0.019784 | 0.017126 | 0.018846 | nan | 0.024104 |
| primary | ME | 0.014423 | 0.014396 | 0.014814 | 0.016163 | 0.018572 |
| primary | MQ | 0.020007 | 0.017125 | 0.018906 | nan | 0.023059 |
| primary | MQE | 0.014515 | 0.0145 | 0.014648 | 0.015782 | 0.017783 |

| sample | target | panel | model | n | empty_pct | mean_selected | own_target_pct | external_pct | quarterly_pct | other_monthly_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | HICPNEF_EA | M | FarmPredict | 120 | 46.6667 | 1.2167 | 0.0 | 0.0 | 0.0 | 53.3333 |
| primary | HICPNEF_EA | ME | FarmPredict | 120 | 96.6667 | 0.0583 | 0.0 | 3.3333 | 0.0 | 2.5 |
| primary | HICPNEF_EA | MQ | FarmPredict | 120 | 50.0 | 1.025 | 0.0 | 0.0 | 0.0 | 50.0 |
| primary | HICPNEF_EA | MQE | FarmPredict | 120 | 79.1667 | 0.275 | 0.0 | 11.6667 | 0.0 | 13.3333 |

| sample | comparison | FP_MSE_reduction | DFM_MSE_reduction | sparse_gain_change | base_sparse_gain | expanded_sparse_gain | DFM_gain_pct | FP_gain_pct |
|---|---|---|---|---|---|---|---|---|
| primary | ME vs M | 0.00403228 | 0.0053603 | -0.00132803 | 0.00093794 | -0.00039009 | 27.09448057 | 21.39614449 |
| primary | MQE vs MQ | 0.00425795 | 0.00549174 | -0.00123379 | 0.00110128 | -0.00013251 | 27.44920159 | 22.52212807 |
| outside_crisis | ME vs M | 0.00086425 | 0.00142139 | -0.00055714 | 0.00048066 | -7.647e-05 | 16.56423651 | 10.66923856 |
| outside_crisis | MQE vs MQ | 0.00085472 | 0.0013044 | -0.00044968 | 0.00042727 | -2.241e-05 | 15.08444064 | 10.39798851 |
| crisis | ME vs M | 0.01142433 | 0.01455111 | -0.00312677 | 0.00200492 | -0.00112185 | 31.68565109 | 26.01263225 |
| crisis | MQE vs MQ | 0.01219884 | 0.01526221 | -0.00306338 | 0.00267398 | -0.00038939 | 32.81298454 | 27.82661235 |

The channel identity is exact: FP MSE reduction = DFM MSE reduction + change in the incremental sparse gain, where sparse gain = DFM MSE minus FP MSE. In both primary comparisons the common-only forecast improves more than the full hybrid, and the sparse gain changes from positive to negative. The same signs hold outside the crisis. Therefore the full ME/MQE improvement is not an improvement in the incremental residual correction; that correction partially offsets the common-forecast improvement. This accounting does not isolate a structural causal effect of changing the factor panel.

The ME residual LASSO is mostly empty. Its few selected terms are IGDSX M4 (non-energy industrial-goods HICP inflation), IRT3M_EACC M1 (short-term interest rate), and RETA_006 M2. MQE also uses long-term interest-rate residuals and retail-price expectations, but the aggregate correction still worsens its expanded DFM. Empty support means an intercept-only residual forecast, not necessarily an exactly zero correction.

Fixed-factor external residuals improve the original M/MQ hybrids, including outside the crisis. Hence external residuals can be useful with the old factor representation. They still do not beat the expanded-panel DFM in the primary sample, and outside the crisis do not beat the expanded hybrid. The full factor and residual channels are not additive interventions: residuals, supports, penalties and complete-case samples change together. The supported conclusion is that the best core forecasts use an improved common forecast, not that external residual signals are universally useless.

## Outside-crisis sensitivity

| sample | target | comparison | n | own_dropped | own_added | own_dropped_with_external | own_retained | gain_pct | n_train_base | n_train_expanded |
|---|---|---|---|---|---|---|---|---|---|---|
| outside_crisis | HICPOV_EA | ME vs M | 84 | 78 | 0 | 78 | 0 | 2.04 | 218.8571 | 216.2976 |
| outside_crisis | HICPOV_EA | MQE vs MQ | 84 | 73 | 0 | 73 | 0 | 9.8394 | 215.8571 | 213.2976 |
| outside_crisis | HICPNG_EA | ME vs M | 84 | 59 | 0 | 59 | 0 | 10.8827 | 218.8571 | 216.2976 |
| outside_crisis | HICPNG_EA | MQE vs MQ | 84 | 59 | 0 | 59 | 0 | 10.0355 | 215.8571 | 213.2976 |
| outside_crisis | HICPNEF_EA | ME vs M | 84 | 2 | 0 | 2 | 0 | 11.0503 | 218.8571 | 216.2976 |
| outside_crisis | HICPNEF_EA | MQE vs MQ | 84 | 24 | 0 | 24 | 0 | 5.7672 | 215.8571 | 213.2976 |

The direct-LASSO improvement and absence of own-target lags persist outside the crisis for each target in both comparisons. This does not establish significance; no additional hypothesis tests were run.

## Reproduction and files

Run `python -m scripts.inflation_external_selection` in a fresh output directory, then `python -m scripts.inflation_external_selection_report`. Full term frequencies (including released positions), per-origin supports, centered contributions, channel decompositions and sensitivity results are in the CSVs beside this report. `audit.json` records source hashes, accepted LF/CRLF-only differences and maximum forecast reconstruction error. Original forecasts, model settings and estimators remain unchanged.
