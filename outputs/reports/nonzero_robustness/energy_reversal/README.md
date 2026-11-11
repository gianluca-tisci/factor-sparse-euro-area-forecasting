# Why direct LASSO catches up in energy inflation

Saved primary ME/MQE forecasts; 2020 scoring months excluded. Descriptive post-hoc diagnosis, not a causal test or new validation sample.

## Period MSE and win counts

| panel | phase | model | n | MSE | LASSO_MSE | reduction_vs_LASSO_percent | LASSO_win_months | intercept_only_LASSO_months |
|---|---|---|---|---|---|---|---|---|
| ME | pre_2022_no_2020 | FarmPredict | 72 | 1.530466 | 1.441147 | -6.197775 | 34 | 0 |
| ME | crisis_2022_2023 | FarmPredict | 24 | 17.548858 | 8.53109 | -105.704763 | 16 | 0 |
| ME | post_2023 | FarmPredict | 24 | 1.841617 | 1.926979 | 4.42982 | 9 | 0 |
| ME | pre_2022_no_2020 | Fixed_factor | 72 | 1.274133 | 1.441147 | 11.588945 | 36 | 0 |
| ME | crisis_2022_2023 | Fixed_factor | 24 | 13.477274 | 8.53109 | -57.978335 | 13 | 0 |
| ME | post_2023 | Fixed_factor | 24 | 1.901476 | 1.926979 | 1.32348 | 12 | 0 |
| MQE | pre_2022_no_2020 | FarmPredict | 72 | 1.502335 | 1.489831 | -0.839271 | 33 | 0 |
| MQE | crisis_2022_2023 | FarmPredict | 24 | 18.148106 | 8.238419 | -120.286256 | 14 | 0 |
| MQE | post_2023 | FarmPredict | 24 | 1.882932 | 1.919714 | 1.91601 | 13 | 0 |
| MQE | pre_2022_no_2020 | Fixed_factor | 72 | 1.288754 | 1.489831 | 13.496664 | 35 | 0 |
| MQE | crisis_2022_2023 | Fixed_factor | 24 | 13.460024 | 8.238419 | -63.381141 | 15 | 0 |
| MQE | post_2023 | Fixed_factor | 24 | 1.879024 | 1.919714 | 2.119587 | 11 | 0 |

## Reversal episodes and the October counterexample

| panel | origin | actual | LASSO | DFM | FarmPredict | Fixed_factor | selected_slopes | selected_terms |
|---|---|---|---|---|---|---|---|---|
| ME | 2022-04 | -1.932942 | 3.296922 | 10.121607 | 10.123971 | 6.779471 | 4 | PPICOG_EA__M1;MCOILBRENTEU__M1;INDU_006__M1;RETA_006__M1 |
| ME | 2022-07 | 0.178845 | 1.485022 | 6.531771 | 6.532115 | 4.370722 | 2 | MCOILBRENTEU__M1;INDU_006__M1 |
| ME | 2022-08 | 0.202149 | 1.039959 | 5.656285 | 5.65744 | 3.61551 | 2 | MCOILBRENTEU__M1;INDU_006__M1 |
| ME | 2022-10 | 4.467192 | 0.971962 | 5.878312 | 5.675238 | 3.961831 | 2 | MCOILBRENTEU__M1;INDU_006__M1 |
| ME | 2022-11 | -1.000479 | 1.59881 | 2.911517 | 2.896328 | 4.711926 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |
| ME | 2022-12 | -4.704085 | 0.526337 | 2.419039 | 2.404922 | 3.251744 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |
| ME | 2023-03 | -3.889089 | -0.520117 | 1.023405 | 1.007778 | 1.279006 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |
| MQE | 2022-04 | -1.932942 | 3.207883 | 10.172054 | 10.174043 | 6.850995 | 3 | MCOILBRENTEU__M1;INDU_006__M1;RETA_006__M1 |
| MQE | 2022-07 | 0.178845 | 1.65008 | 7.007232 | 7.007398 | 4.41337 | 3 | MCOILBRENTEU__M1;INDU_006__M1;RETA_006__M1 |
| MQE | 2022-08 | 0.202149 | 1.019589 | 6.212 | 6.212519 | 3.677599 | 2 | MCOILBRENTEU__M1;INDU_006__M1 |
| MQE | 2022-10 | 4.467192 | 0.95374 | 5.955387 | 5.814891 | 4.030314 | 2 | MCOILBRENTEU__M1;INDU_006__M1 |
| MQE | 2022-11 | -1.000479 | 1.564923 | 2.948507 | 2.929458 | 4.733536 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |
| MQE | 2022-12 | -4.704085 | 0.538147 | 2.463051 | 2.444792 | 3.27759 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |
| MQE | 2023-03 | -3.889089 | -0.499847 | 1.116242 | 0.179779 | 1.399143 | 4 | PPINRG_EA__M1;MCOILBRENTEU__M1;PNGASEUUSDM__M2;INDU_006__M1 |

Check the saved per-origin predictor lists for the revised nonzero-IQR run rather than transferring individual selections from the historical run. Direct LASSO has nonzero slopes throughout 2022-2023 in ME/MQE, as verified against saved tuning counts. Selection alone does not establish contribution direction or causal importance; that requires predictor values, centering and coefficient products or an ablation.

LASSO forecasts are often much less positive than the expanded-panel DFM forecasts. When realized inflation becomes negative, a smaller positive forecast can make a much smaller squared error without predicting the correct sign. July and August 2022 also reward less positive forecasts despite slightly positive realizations. It is inaccurate to call every such forecast zero, an unconditional training mean or evidence of stronger shrinkage: ME/MQE retain selected signals, and near-zero predictions can also reflect predictor realizations and offsetting terms.

October 2022 is a counterexample for the original expanded-panel FarmPredict comparison: the large positive realization is closer to its high forecast than to direct LASSO. The report retains every scored month, including losses. A few large favorable squared-error differences can outweigh more numerous small losses; cumulative rankings depend on error magnitudes, not just win counts. The crisis sign table separates negative and nonnegative realized changes.

Fixed-factor ablation changes the common forecast relative to original ME/MQE, but still leaves a largely factor-driven prediction. A LASSO residual correction does not shrink the entire common forecast. Both the original correction (FarmPredict minus DFM) and the fixed correction are exported monthly, so this observation is directly inspectable.

Outside-crisis results show whether relative losses concentrate in the excluded episode, not why it occurred. Crisis data remain in later training samples. Inspect external_crisis/comparisons.csv in the parent report for the primary, broad and narrow exclusions. No LaTeX changes or refits.
