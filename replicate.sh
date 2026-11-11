#!/usr/bin/env bash
# Rebuild every figure, table and number in the paper from the saved forecasts (about 2 minutes).
# Run from the repository root after `uv sync`. Re-estimating the forecasts: README.md, step 3.
#
# The scripts refuse to overwrite saved results, so the committed exhibits are first moved to
# outputs/.previous/ (git-ignored); scripts/check_replication.py then compares every rebuilt
# table and number with the committed one.
set -euo pipefail

if [ -e outputs/.previous ]; then
  echo "outputs/.previous exists from an earlier run: move or delete it first." >&2
  exit 1
fi
mkdir -p outputs/.previous/reports/nonzero_robustness
for d in outputs/reports/*; do
  case "$(basename "$d")" in
    industrial_nonzero | nonzero_robustness) ;;  # inputs computed by re-estimation (step 3)
    *) mv "$d" outputs/.previous/reports/ ;;
  esac
done
for f in outputs/reports/nonzero_robustness/*; do
  [ "$(basename "$f")" = mqe_ablation ] || mv "$f" outputs/.previous/reports/nonzero_robustness/
done
mv outputs/figures outputs/tables outputs/.previous/
mkdir -p outputs/figures outputs/tables
cp outputs/numbers.tex outputs/.previous/  # rewritten macro by macro, so it stays in place

run() { echo ">> $1"; uv run python -m "scripts.$1"; }

# Section 4 robustness reports (crisis windows, energy selections, fixed-factor variants)
run refresh_nonzero_reports
run reorganized_results_exhibits
run robustness_table_display

# Section 4 main results
run results_ladder_exhibits
run ladder_steps_display
run monthly_external_tables
run inflation_external_selection
run inflation_external_selection_report
run core_common_share_comparison
run manufacturing_common_share_comparison
run results_section_nonzero
run paper_three_sets           # restricts the reported diagnostics and robustness tables to M, MQ, ME

# Sections 2-3 and the appendix: data, number of factors, sparse step, design numbers
run descriptives
run factor_selection_evidence
run sparse_diagnostics
run results_figures
run design_numbers
run paper_draft_report
run paper_three_sets_residuals

echo
run check_replication
