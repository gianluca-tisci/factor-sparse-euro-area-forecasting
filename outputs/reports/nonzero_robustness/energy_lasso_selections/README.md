# Energy-MQ: direct LASSO selections during the cumulative-MSE reversal

Saved cold paper run, no 2020 scoring months. Focus: six high-loss-difference episodes in 2022-2023. This is the direct LASSO on transformed observations, not the sparse regression on DFM residuals. No EM or LASSO estimation was run; coefficients are read from the saved selection file.

## Episode forecasts

| origin | actual | LASSO | DFM | FarmPredict_DFM | training_response_mean | n_train | selected | alpha | reconstruction_difference |
|---|---|---|---|---|---|---|---|---|---|
| 2022-04 | -1.932942 | 2.203044 | 6.847841 | 6.851207 | 0.334124 | 243 | 2 | 0.468576 | 0.0 |
| 2022-07 | 0.178845 | 0.345 | 4.412309 | 4.413479 | 0.345 | 246 | 0 | 0.585359 | 0.0 |
| 2022-08 | 0.202149 | 0.344328 | 3.676456 | 3.677709 | 0.344328 | 247 | 0 | 0.576424 | 0.0 |
| 2022-11 | -1.000479 | 0.660957 | 4.735762 | 4.733375 | 0.369413 | 250 | 2 | 0.489652 | 0.0 |
| 2022-12 | -4.704085 | -0.055642 | 3.279545 | 3.277543 | 0.363955 | 251 | 1 | 0.485209 | 0.0 |
| 2023-03 | -3.889089 | -0.771838 | 1.846571 | 1.842621 | 0.331361 | 254 | 2 | 0.388485 | 0.0 |

## Selected predictors and centered contributions

| origin | term | predictor | reference_month | available_month | coefficient | observed_predictor | training_predictor_mean | centered_contribution |
|---|---|---|---|---|---|---|---|---|
| 2022-04 | HICPOV_EA__M1 | HICPOV_EA | 2022-03 | 2022-04 | 0.499605 | 1.792041 | 0.14349 | 0.823625 |
| 2022-04 | HICPNG_EA__M1 | HICPNG_EA | 2022-03 | 2022-04 | 0.106622 | 10.071896 | 0.268158 | 1.045295 |
| 2022-11 | PPINRG_EA__M1 | PPINRG_EA | 2022-09 | 2022-11 | 0.046347 | 1.42936 | 0.512456 | 0.042496 |
| 2022-11 | HICPOV_EA__M1 | HICPOV_EA | 2022-10 | 2022-11 | 0.200414 | 1.405226 | 0.162552 | 0.249049 |
| 2022-12 | PPINRG_EA__M1 | PPINRG_EA | 2022-10 | 2022-12 | 0.057284 | -6.808733 | 0.516109 | -0.419597 |
| 2023-03 | PPINRG_EA__M1 | PPINRG_EA | 2023-01 | 2023-03 | 0.105727 | -9.354263 | 0.47842 | -1.039577 |
| 2023-03 | HICPNG_EA__M1 | HICPNG_EA | 2023-02 | 2023-03 | 0.041403 | -1.223939 | 0.312724 | -0.063622 |

Prediction = training-response mean + sum of coefficient times (available predictor minus its mean in the effective complete-row training sample). Contributions are in 100*dlog units. Every episode reconstructs its saved LASSO forecast within 1e-8. Coefficient size alone is not a comparable importance measure across differently scaled predictors.

HICPOV is headline HICP inflation; HICPNG is energy HICP inflation; PPINRG is energy producer-price inflation. M1 means the latest released observation: HICP predictors refer to the previous month; PPINRG refers to two months earlier under the assumed publication calendar.

## Interpretation

In April 2022 the LASSO selects lagged headline and energy inflation. Its forecast is still positive and misses the sign of the reversal, but is less positive than DFM/FarmPredict. In July and August it selects no regressors: modified BIC chooses the intercept-only model, yielding the effective training mean. This is strong shrinkage rather than successful selection of a leading crisis indicator.

In November, headline inflation and energy producer prices both contribute positively, so the LASSO again misses the negative sign but makes a smaller error. In December 2022, the October decline in energy producer prices makes the forecast slightly negative. In March 2023, January energy producer prices and February consumer energy inflation both contribute negatively. The sparse model remains far from the realized decline, but is closer than the still-positive DFM-based forecasts.

No quarterly predictor is selected in these six episodes. This does not prove quarterly candidates have no effect: they can change complete-case training rows and the modified-BIC penalty. The result reflects conservative predictions plus, in some episodes, lagged energy-price signals. These are algebraic forecast contributions, not causal effects or an ablation identifying the predictive value of each series. The episodes were chosen after seeing forecast errors, so this is an explanatory diagnostic rather than independent validation.

## Selection frequencies across all 24 origins in 2022-2023

| term | predictor | frequency | n_origins | percent_of_24_origins |
|---|---|---|---|---|
| HICPNG_EA__M1 | HICPNG_EA | M | 16 | 66.667 |
| PPINRG_EA__M1 | PPINRG_EA | M | 14 | 58.333 |
| HICPOV_EA__M1 | HICPOV_EA | M | 4 | 16.667 |
| ICONFIX_EA__M1 | ICONFIX_EA | M | 2 | 8.333 |
| SHIX_EA__M1 | SHIX_EA | M | 2 | 8.333 |
| PPIDCOG_EA__M3 | PPIDCOG_EA | M | 1 | 4.167 |

Intercept-only origins: 2022-05, 2022-06, 2022-07, 2022-08, 2022-09, 2022-10.

Source: saved paper_experiment forecasts, tuning and selections; available transformed observations reconstructed with the unchanged publication and cleaning functions. No LaTeX edits.
