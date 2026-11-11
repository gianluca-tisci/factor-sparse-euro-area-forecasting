"""Run the proposed paper matrix using the approved forecasting implementation."""

import argparse

from src.forecasting.paper import report_paper, run_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/forecasts/paper_experiment_nonzero_iqr")
    parser.add_argument("--profile", default="config/forecast_paper.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--end")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    print(
        report_paper(args.output)
        if args.report_only
        else run_paper(args.output, args.limit, args.end, args.resume, args.profile)
    )


if __name__ == "__main__":
    main()
