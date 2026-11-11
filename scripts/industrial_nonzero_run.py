"""Re-estimate the approved industrial robustness under the current D-72 rule."""

from concurrent.futures import ProcessPoolExecutor

from scripts.industrial_nonzero import OUTPUT, PANELS, PROFILE, report
from src.forecasting.experiment import run


def panel_run(panel):
    return run(str(PROFILE), str(OUTPUT / panel), information_set=panel)


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError("Use a new run directory; no overwriting or implicit resume")
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(panel_run, PANELS))
    report()
