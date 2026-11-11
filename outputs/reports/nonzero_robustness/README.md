# Robustness refreshed under nonzero-IQR cleaning (D-72)

Current-run counterparts of the three merged historical analyses. Historical directories remain untouched. All primary forecasts come from paper_experiment_nonzero_iqr; fixed M factors reuse its existing nonzero-IQR ablation, and fixed MQ factors use the new mqe_ablation directory. The broad and narrow crisis windows, publication delays, lag menus and modified-BIC selection are unchanged. 2020 scoring months are excluded throughout.

## Outside January 2021-December 2023: winners

| target | sample | panel | menu | winner | MSE | n | runner_up | runner_up_MSE | reduction_vs_second_percent |
|---|---|---|---|---|---|---|---|---|---|
| HICPOV_EA | outside_energy_crisis | ME | including_fixed_factor | ME:LASSO | 0.018884 | 84 | shared:AR | 0.019792 | 4.587189 |
| HICPOV_EA | outside_energy_crisis | MQE | including_fixed_factor | MQE:LASSO | 0.018262 | 84 | shared:AR | 0.019792 | 7.730239 |
| HICPNG_EA | outside_energy_crisis | ME | including_fixed_factor | shared:AR | 1.377238 | 84 | ME:Fixed_factor | 1.432362 | 3.848488 |
| HICPNG_EA | outside_energy_crisis | MQE | including_fixed_factor | shared:AR | 1.377238 | 84 | MQE:Fixed_factor | 1.429163 | 3.63328 |
| HICPNEF_EA | outside_energy_crisis | ME | including_fixed_factor | ME:DFM_AR | 0.006981 | 84 | ME:LASSO | 0.006996 | 0.215029 |
| HICPNEF_EA | outside_energy_crisis | MQE | including_fixed_factor | MQE:LASSO | 0.00684 | 84 | MQE:DFM_AR | 0.007228 | 5.37049 |

The old outside-crisis conclusions do not all survive: external LASSO leads headline and MQE core. Core ME still favors DFM+AR, and energy still favors AR. Compare the original model menus separately from the menus including a fixed-factor alternative; PCA-based FarmPredict is included wherever estimated.

## Energy fixed-factor comparisons outside the crisis

| target | sample | panel | numerator | benchmark | n | MSE | benchmark_MSE | final_ratio | reduction_percent |
|---|---|---|---|---|---|---|---|---|---|
| HICPNG_EA | outside_energy_crisis | ME | ME:Fixed_factor | ME:LASSO | 84 | 1.432362 | 1.441523 | 0.993645 | 0.635535 |
| HICPNG_EA | outside_energy_crisis | ME | ME:Fixed_factor | shared:AR | 84 | 1.432362 | 1.377238 | 1.040025 | -4.002525 |
| HICPNG_EA | outside_energy_crisis | ME | ME:Fixed_factor | M:DFM | 84 | 1.432362 | 1.427751 | 1.003229 | -0.322946 |
| HICPNG_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | MQE:LASSO | 84 | 1.429163 | 1.443971 | 0.989745 | 1.025532 |
| HICPNG_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | shared:AR | 84 | 1.429163 | 1.377238 | 1.037703 | -3.770264 |
| HICPNG_EA | outside_energy_crisis | MQE | MQE:Fixed_factor | MQ:DFM | 84 | 1.429163 | 1.426862 | 1.001612 | -0.161236 |

The ablations retain small gains over external direct LASSO, but no longer improve on the same-base factor-only DFM outside the broad crisis window. This is materially different from the historical 0.43% improvements. Positive reductions in the table favor the ablation. These differences are descriptive, not significance tests.

## Execution and preservation

Industrial robustness uses the saved per-origin M/MQ factor parameters because adding target labels does not change the estimation panel. All ten targets and their sparse/PCA/univariate forecasts are recomputed. The initial cold rerun checkpoints are preserved in outputs/forecasts/industrial_nonzero; 7,430 forecast cells from these cold fits matched the checkpoint-based execution within 1e-10, with maximum absolute difference 5.4e-13. The completed industrial report independently checks every original-target forecast against the current primary run. Its metadata records checkpoint reuse, real saved EM diagnostics and source hashes; no likelihood trace or new fit is fabricated.

The full industrial execution is in outputs/forecasts/industrial_nonzero_cached and its shareable report/archive in outputs/reports/industrial_nonzero. No changes to src/forecasting, existing estimation configs, raw data or primary forecasts are needed. New configs and new output directories record this execution. The crisis exclusions affect scoring only; later training still includes crisis observations.

## Files and reproduction

Run scripts.fixed_factor_mqe_nonzero, scripts.refresh_nonzero_reports, scripts.industrial_nonzero_cached and scripts.compare_nonzero_reports in that dependency order (industrial work can run independently). The raw regeneration commands refuse to overwrite completed directories. Figures for crisis sensitivity and fixed-factor comparisons are in this directory's figures folder. Galleries: index.html, inflation_crisis/index.html and external_crisis/index.html. Energy episode selections and contributions are in energy_lasso_selections/ and energy_reversal/. Historical-to-current numerical changes are in changes_*.csv.
| fig_147_manufacturing_pca_m_original | outputs\reports\nonzero_robustness\figures\fig_147_manufacturing_pca_m_original.png, outputs\reports\nonzero_robustness\figures\fig_147_manufacturing_pca_m_original.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| fig_148_manufacturing_pca_mq_original | outputs\reports\nonzero_robustness\figures\fig_148_manufacturing_pca_mq_original.png, outputs\reports\nonzero_robustness\figures\fig_148_manufacturing_pca_mq_original.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| fig_149_manufacturing_pca_me_original | outputs\reports\nonzero_robustness\figures\fig_149_manufacturing_pca_me_original.png, outputs\reports\nonzero_robustness\figures\fig_149_manufacturing_pca_me_original.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| fig_150_manufacturing_pca_mqe_original | outputs\reports\nonzero_robustness\figures\fig_150_manufacturing_pca_mqe_original.png, outputs\reports\nonzero_robustness\figures\fig_150_manufacturing_pca_mqe_original.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| fig_151_manufacturing_pca_me_fixed | outputs\reports\nonzero_robustness\figures\fig_151_manufacturing_pca_me_fixed.png, outputs\reports\nonzero_robustness\figures\fig_151_manufacturing_pca_me_fixed.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| fig_152_manufacturing_pca_mqe_fixed | outputs\reports\nonzero_robustness\figures\fig_152_manufacturing_pca_mqe_fixed.png, outputs\reports\nonzero_robustness\figures\fig_152_manufacturing_pca_mqe_fixed.pdf | scripts\plot_manufacturing_nonzero.py | Manufacturing with mandatory PCA+VAR+LASSO comparator |
| manufacturing_pca_comparisons | outputs\reports\nonzero_robustness\manufacturing_pca\index.html, outputs\reports\nonzero_robustness\manufacturing_pca\README.md | scripts\plot_manufacturing_nonzero.py | Revised manufacturing comparisons; PCA+VAR+LASSO always included |
