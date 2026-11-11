"""Run the approved ten-target M/MQ robustness in a new directory (D-75)."""

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.report_mq_energy_diagnostics import markdown
from src import _style
from src.forecasting import paper
from src.forecasting.experiment import run

PROFILE = Path("config/forecast_industrial_nonzero.yaml")
OUTPUT = Path("outputs/forecasts/industrial_nonzero")
REPORT = Path("outputs/reports/industrial_nonzero")
PANELS = ["M", "MQ"]


def specifications():
    return [("shared", model) for model in ["RW", "AR"]] + [
        (panel, model)
        for panel in PANELS
        for model in paper.MODELS[2:]
        if panel == "M" or model not in ["PCR", "PCA_VAR", "PCA_VAR_LASSO"]
    ]


def report():
    targets = yaml.safe_load(PROFILE.read_text())["targets"]
    rows, selections, tuning, fits = [], [], [], []
    baseline = Path("outputs/forecasts/paper_experiment_nonzero_iqr/forecasts.csv")
    baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    for panel in PANELS:
        root = OUTPUT / panel
        meta = json.loads((root / "metadata.json").read_text())
        assert meta["completion"] == "COMPLETE"
        assert meta["profile"]["targets"] == targets
        assert meta["profile"]["reestimation_start"] == "cold"
        primary_meta = json.loads(
            (
                Path("outputs/forecasts/paper_experiment_nonzero_iqr") / panel / "metadata.json"
            ).read_text()
        )
        for key, value in primary_meta["profile"].items():
            if key not in ["targets", "classification"]:
                assert meta["profile"][key] == value, (panel, key)
        assert meta["panel_members"] == primary_meta["panel_members"]
        for name, digest in meta["source_hashes"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
        for origin in meta["origins"]:
            folder = root / origin
            frame = pd.read_csv(folder / "forecasts.csv")
            frame["model"] = frame.model.replace(paper.ALIASES)
            frame["information_set"] = np.where(frame.model.isin(["AR", "RW"]), "shared", panel)
            frame["valid"] = np.isfinite(frame.forecast) & np.isfinite(frame.actual)
            rows.append(frame)
            fits.append({"information_set": panel, **json.loads((folder / "fit.json").read_text())})
            for filename, dest in [("selected.csv", selections), ("tuning.csv", tuning)]:
                item = pd.read_csv(folder / filename)
                item["information_set"] = panel
                dest.append(item)
    forecasts = pd.concat(rows, ignore_index=True)
    # Reuse the primary scoring function with this explicitly approved scope only.
    old_targets, old_specifications = paper.TARGETS, paper.specifications
    paper.TARGETS, paper.specifications = targets, specifications
    try:
        expected = set(paper.specifications())
        for _, group in forecasts.groupby(["target", "origin"]):
            assert set(zip(group.information_set, group.model)) == expected
        scores, pairs, coverage, _interactions = paper.evaluate_paper(forecasts)
    finally:
        paper.TARGETS, paper.specifications = old_targets, old_specifications
    original = pd.read_csv(baseline, low_memory=False)
    keys = ["origin", "target", "information_set", "model"]
    original = original[original.information_set.isin(["shared", "M", "MQ"])]
    matched = original.merge(
        forecasts, on=keys, suffixes=("_primary", "_robustness"), validate="one_to_one"
    )
    assert len(matched) == len(original)
    np.testing.assert_allclose(
        matched.forecast_primary,
        matched.forecast_robustness,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_allclose(matched.actual_primary, matched.actual_robustness)
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == baseline_hash
    primary = scores[scores["sample"] == "primary"].copy()
    scores = primary
    pairs = pairs[pairs["sample"] == "primary"]
    coverage = coverage[coverage["sample"] == "primary"]
    winners = []
    gains = []
    for target in targets:
        for panel in PANELS:
            subset = primary[
                (primary.target == target) & primary.information_set.isin(["shared", panel])
            ]
            ranked = subset.sort_values("MSE")
            assert ranked.n.nunique() == 1
            winner, runner = ranked.iloc[0], ranked.iloc[1]
            winners.append(
                {
                    "target": target,
                    "panel": panel,
                    "winner": winner.model,
                    "winner_MSE": winner.MSE,
                    "runner_up": runner.model,
                    "runner_up_MSE": runner.MSE,
                    "reduction_percent": 100 * (1 - winner.MSE / runner.MSE),
                    "best_or_tied": ";".join(
                        ranked.loc[
                            np.isclose(ranked.MSE, winner.MSE, rtol=1e-10, atol=1e-12), "model"
                        ]
                    ),
                    "n": int(winner.n),
                }
            )
            for row in subset.itertuples():
                gains.append(
                    {
                        "target": target,
                        "panel": panel,
                        "winner": winner.model,
                        "comparator": row.model,
                        "reduction_percent": 100 * (1 - winner.MSE / row.MSE),
                    }
                )
    winners = pd.DataFrame(winners)
    REPORT.mkdir(parents=True, exist_ok=True)
    tables = {
        "scores": scores,
        "paired_comparisons": pairs,
        "coverage": coverage,
        "winners": winners,
        "winner_reductions": pd.DataFrame(gains),
        "m_vs_mq": pairs[
            pairs.augmented.str.startswith("MQ:") & pairs.baseline.str.startswith("M:")
        ].assign(reduction_percent=lambda x: 100 * (1 - x.MSE_ratio)),
        "fit_monitoring": pd.DataFrame(fits),
    }
    for name, table in tables.items():
        table.to_csv(REPORT / f"{name}.csv", index=False)
    for name, table in {
        "forecasts": forecasts,
        "selected": pd.concat(selections, ignore_index=True),
        "tuning": pd.concat(tuning, ignore_index=True),
    }.items():
        (REPORT / f"{name}.csv.gz").write_bytes(
            gzip.compress(table.to_csv(index=False).encode(), mtime=0)
        )
    text = [
        "# Industrial-production robustness (D-75)",
        (
            "EXPLORATORY: ten pre-specified targets, M and MQ only. Same settings as the latest cold primary run. "
            "Primary scoring through 2025-12, excluding 2020 target months. No LaTeX changes. "
            "Each target uses common dates across all M/MQ specifications. Report every sector, including losses. "
            "The industrial aggregates overlap and are not independent replications. "
            "This extension follows inspection of the earlier 47-target exercise on the same data; "
            "it is exploratory robustness, not an untouched validation sample."
        ),
        "## Winners and runner-up margins",
        markdown(winners.round(6)),
        (
            "FarmPredict includes DFM+LASSO (saved identifier FarmPredict) and PCA+VAR+LASSO "
            "(PCA_VAR_LASSO, estimated only in M). Read winner_reductions.csv for every comparator. "
            "P-values in paired_comparisons.csv are unadjusted; rankings are descriptive."
        ),
        (
            "Compressed forecast/selection/tuning snapshots retain all attempted origins for provenance; "
            "the scores, contrasts and winner tables use only the primary sample. "
            "See docs/analysis/industrial_robustness_validation.md for execution and Windows portability checks."
        ),
        (
            f"Validated all {len(matched)} original-target forecast cells against the primary run to tolerance 1e-10. "
            "The saved primary forecasts and estimator files were not changed."
        ),
    ]
    (REPORT / "README.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    audit = {
        "baseline_sha256": baseline_hash,
        "original_cells_verified": len(matched),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parallel_driver_sha256": hashlib.sha256(
            Path("scripts/industrial_nonzero_run.py").read_bytes()
        ).hexdigest(),
        "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        "targets": targets,
        "panels": PANELS,
    }
    (REPORT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for panel in PANELS:
        (REPORT / f"metadata_{panel}.json").write_bytes(
            (OUTPUT / panel / "metadata.json").read_bytes()
        )
    with zipfile.ZipFile(REPORT / "execution_sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for panel in PANELS:
            meta = json.loads((OUTPUT / panel / "metadata.json").read_text())
            for name, digest in {**meta["source_hashes"], **meta["input_hashes"]}.items():
                archive_name = Path(name).as_posix()
                if (
                    not name.endswith((".py", ".yaml", ".lock"))
                    or archive_name in archive.namelist()
                ):
                    continue
                payload = Path(name).read_bytes()
                assert hashlib.sha256(payload).hexdigest() == digest, name
                archive.writestr(archive_name, payload)
        archive.writestr("scripts/industrial_nonzero.py", Path(__file__).read_bytes())
        archive.writestr(
            "scripts/industrial_nonzero_run.py",
            Path("scripts/industrial_nonzero_run.py").read_bytes(),
        )
        archive.writestr(
            "scripts/report_mq_energy_diagnostics.py",
            Path("scripts/report_mq_energy_diagnostics.py").read_bytes(),
        )
        archive.writestr("src/_style.py", Path("src/_style.py").read_bytes())
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in REPORT.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _style.record_output(
        "industrial_robustness", [str(REPORT / "README.md")], "EXPLORATORY D-75; no LaTeX"
    )
    print(winners.to_string(index=False), flush=True)
    print(f"Verified {len(matched)} original forecast cells; results: {REPORT}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        for panel in PANELS:
            run(str(PROFILE), str(OUTPUT / panel), resume=args.resume, information_set=panel)
    report()


if __name__ == "__main__":
    main()
