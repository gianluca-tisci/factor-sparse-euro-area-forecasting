# Outputs

| id | files | script | used in |
|---|---|---|---|
| results_reorganized | outputs/reports/results_reorganized/audit.json | scripts/reorganized_results_exhibits.py | New Results structure; current nonzero-IQR outputs |
| robustness_table_display | outputs/reports/robustness_table_display/tables/tab_101_robustness_summary.tex | scripts/robustness_table_display.py | Appendix robustness table: including-2020 rows omitted |
| ladder_steps_display | outputs/reports/ladder_steps_display/figures/fig_198_ladder_steps.pdf, outputs/reports/ladder_steps_display/audit.json | scripts/ladder_steps_display.py | Figure display restricted to M/MQ/ME for inflation, M/MQ for manufacturing |
| monthly_external_tables | outputs/reports/monthly_external_tables/audit.json | scripts/monthly_external_tables.py | External discussion restricted to M/ME; full saved tables retained |
| inflation_external_selection | outputs/reports/inflation_external_selection/audit.json | scripts/inflation_external_selection.py | Matched direct-LASSO selection and core factor/sparse channel diagnostics |
| core_common_share_comparison | outputs/reports/core_common_share_comparison/analysis.md, outputs/reports/core_common_share_comparison/numbers.tex | scripts/core_common_share_comparison.py | Results 4.2.2: M/ME core common variance shares, distinct from forecast accuracy |
| manufacturing_common_share_comparison | outputs/reports/manufacturing_common_share_comparison/analysis.md, outputs/reports/manufacturing_common_share_comparison/numbers.tex | scripts/manufacturing_common_share_comparison.py | Results 4.2.1: M/MQ manufacturing common variance shares, distinct from forecast accuracy |
| results_section_nonzero | outputs/reports/results_section_nonzero/audit.json, outputs/reports/results_section_nonzero/numbers.tex | scripts/results_section_nonzero.py | Current D-72 nonzero-IQR Results rewrite; explicit provenance |
| tab_01_data | outputs/tables/tab_01_data.tex | scripts/descriptives.py | paper: data section |
| tab_03_series | outputs/tables/tab_03_series.tex | scripts/descriptives.py | paper: appendix, series list |
| tab_02_target | outputs/tables/tab_02_target.tex | scripts/descriptives.py | paper: data section, targets |
| fig_01_targets | outputs/figures/fig_01_targets.pdf | scripts/descriptives.py | paper: data section |
| fig_03_release_timing | outputs/figures/fig_03_release_timing.pdf | scripts/descriptives.py | saved, not in the paper: publication delays at one date |
| tab_80_factor_number | outputs/tables/tab_80_factor_number.tex | scripts/factor_selection_evidence.py | paper: methodology, number of factors |
| tab_81_var_order | outputs/tables/tab_81_var_order.tex | scripts/factor_selection_evidence.py | paper: appendix, factor VAR order |
| fig_80_scree | outputs/figures/fig_80_scree.pdf | scripts/factor_selection_evidence.py | paper: appendix, number of factors |
| fig_81_abc | outputs/figures/fig_81_abc.pdf | scripts/factor_selection_evidence.py | paper: methodology, number of factors |
| tab_83_var_order_fits | outputs/tables/tab_83_var_order_fits.tex | scripts/factor_selection_evidence.py | paper: appendix, factor VAR order |
| tab_82_em_diagnostics | outputs/tables/tab_82_em_diagnostics.tex | scripts/factor_selection_evidence.py | paper: appendix, EM diagnostics |
| fig_83_correlation | outputs/figures/fig_83_correlation.pdf | scripts/sparse_diagnostics.py | paper: methodology, Figure fig:residual-correlation (3.3.3) |
| tab_84_residual_dependence | outputs/tables/tab_84_residual_dependence.tex | scripts/sparse_diagnostics.py | not in the paper; its numbers are quoted in the notes of fig_83 (3.3.3) |
| fig_90_sparse_step | outputs/figures/fig_90_sparse_step.pdf | scripts/results_figures.py | draft: results, question 1 |
| fig_91_information_sets | outputs/figures/fig_91_information_sets.pdf | scripts/results_figures.py | draft: results, question 2 |
| fig_92_mse_heatmap | outputs/figures/fig_92_mse_heatmap.pdf | scripts/results_figures.py | draft: results, full MSE matrix |
| tab_90_headline_mse | outputs/tables/tab_90_headline_mse.tex | scripts/paper_draft_report.py | Results: primary matrix |
| tab_91_energy_mse | outputs/tables/tab_91_energy_mse.tex | scripts/paper_draft_report.py | Results: primary matrix |
| tab_92_core_mse | outputs/tables/tab_92_core_mse.tex | scripts/paper_draft_report.py | Results: primary matrix |
| tab_93_manufacturing_mse | outputs/tables/tab_93_manufacturing_mse.tex | scripts/paper_draft_report.py | Results: primary matrix |
| tab_94_fixed_factor_ablation | outputs/tables/tab_94_fixed_factor_ablation.tex | scripts/paper_draft_report.py | Results: separate ablation |
| inflation_external_selection_report | outputs/reports/inflation_external_selection/analysis.md, outputs/reports/inflation_external_selection/numbers.tex | scripts/inflation_external_selection_report.py | Results: external selection and core prediction channels |
| results_ladder | outputs/reports/results_ladder/numbers.tex, outputs/reports/results_ladder/selection.csv | scripts/results_ladder_exhibits.py | Results 4.1-4.2: ladder and information-set exhibits and macros |
