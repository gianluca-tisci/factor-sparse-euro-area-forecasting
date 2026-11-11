"""Run the release-aware expanding-window forecasting experiment.

From the repository root: uv run python -m scripts.analysis
Smoke test: uv run python -m scripts.analysis --limit 2 --output scratch/forecast_smoke
Report saved forecasts: uv run python -m scripts.analysis --report-only --output <run>
"""

import argparse
import json

from src.forecasting.experiment import run
from src.forecasting.report import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="config/forecast_analysis.yaml")
    parser.add_argument("--output", default="outputs/forecasts/analysis_final")
    parser.add_argument("--limit", type=int, help="Development: only first N test origins")
    parser.add_argument("--end", help="Last forecast origin, YYYY-MM")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse matching complete checkpoints")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.report_only:
        run(args.profile, args.output, args.limit, args.end, args.resume)
    print(json.dumps(report(args.output), indent=2))


if __name__ == "__main__":
    main()
