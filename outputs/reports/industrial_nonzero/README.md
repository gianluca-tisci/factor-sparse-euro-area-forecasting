# Industrial-production robustness (D-75)

EXPLORATORY: ten pre-specified targets, M and MQ only. Same settings as the latest cold primary run. Primary scoring through 2025-12, excluding 2020 target months. No LaTeX changes. Each target uses common dates across all M/MQ specifications. Report every sector, including losses. The industrial aggregates overlap and are not independent replications. This extension follows inspection of the earlier 47-target exercise on the same data; it is exploratory robustness, not an untouched validation sample.

## Winners and runner-up margins

| target | panel | winner | winner_MSE | runner_up | runner_up_MSE | reduction_percent | best_or_tied | n |
|---|---|---|---|---|---|---|---|---|
| HICPOV_EA | M | DFM_AR | 0.074418 | DFM | 0.075013 | 0.793611 | DFM_AR | 120 |
| HICPOV_EA | MQ | DFM | 0.075377 | DFM_AR | 0.075399 | 0.028871 | DFM | 120 |
| HICPNG_EA | M | AR | 3.118783 | LASSO | 3.293357 | 5.300816 | AR | 120 |
| HICPNG_EA | MQ | AR | 3.118783 | LASSO | 3.134804 | 0.511066 | AR | 120 |
| HICPNEF_EA | M | PCR | 0.017074 | DFM_AR | 0.017126 | 0.303467 | PCR | 120 |
| HICPNEF_EA | MQ | DFM_AR | 0.017125 | AR | 0.01836 | 6.72568 | DFM_AR | 120 |
| IPMN_EA | M | PCA_VAR_LASSO | 1.618389 | PCA_VAR | 1.639705 | 1.299981 | PCA_VAR_LASSO | 121 |
| IPMN_EA | MQ | LASSO | 1.699408 | AR | 1.798446 | 5.506847 | LASSO | 121 |
| IPCAG_EA | M | PCA_VAR_LASSO | 6.039369 | PCA_VAR | 6.328173 | 4.563792 | PCA_VAR_LASSO | 121 |
| IPCAG_EA | MQ | AR | 6.576981 | LASSO | 6.848046 | 3.958296 | AR | 121 |
| IPCOG_EA | M | LASSO | 4.174574 | AR | 4.266385 | 2.151968 | LASSO | 121 |
| IPCOG_EA | MQ | LASSO | 4.183438 | AR | 4.266385 | 1.944202 | LASSO | 121 |
| IPDCOG_EA | M | AR | 2.854842 | LASSO | 3.375267 | 15.418772 | AR | 121 |
| IPDCOG_EA | MQ | AR | 2.854842 | LASSO | 3.395942 | 15.93372 | AR | 121 |
| IPNDCOG_EA | M | AR | 5.279331 | LASSO | 5.294586 | 0.288113 | AR | 121 |
| IPNDCOG_EA | MQ | AR | 5.279331 | LASSO | 5.294588 | 0.288158 | AR | 121 |
| IPING_EA | M | PCR | 0.860938 | PCA_VAR | 0.935557 | 7.975838 | PCR | 121 |
| IPING_EA | MQ | AR | 0.943302 | LASSO | 1.063339 | 11.288679 | AR | 121 |
| IPNRG_EA | M | AR | 3.108188 | DFM_AR | 3.253984 | 4.480524 | AR | 121 |
| IPNRG_EA | MQ | AR | 3.108188 | DFM_AR | 3.224818 | 3.616629 | AR | 121 |

FarmPredict includes DFM+LASSO (saved identifier FarmPredict) and PCA+VAR+LASSO (PCA_VAR_LASSO, estimated only in M). Read winner_reductions.csv for every comparator. P-values in paired_comparisons.csv are unadjusted; rankings are descriptive.

Compressed forecast/selection/tuning snapshots retain all attempted origins for provenance; the scores, contrasts and winner tables use only the primary sample. See docs/analysis/industrial_robustness_validation.md for execution and Windows portability checks.

Validated all 7280 original-target forecast cells against the primary run to tolerance 1e-10. The saved primary forecasts and estimator files were not changed.
