"""Compare model-implied manufacturing common shares from saved M and MQ parameters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from scripts.report_mq_energy_diagnostics import markdown
from src import _style

ROOT = Path("outputs/forecasts/paper_experiment_nonzero_iqr")
OUT = Path("outputs/reports/manufacturing_common_share_comparison")
PERIODS = {"Early": ("2015-01", "2019-12"), "Late": ("2021-01", "2025-12")}


def main():
    if OUT.exists():
        raise FileExistsError("Use a new report directory; saved reports are immutable")
    sources = [Path(__file__), Path("scripts/sparse_diagnostics.py")]
    rows = []
    for panel in ["M", "MQ"]:
        membership = ROOT / panel / "selected_series.csv"
        sources.append(membership)
        names = pd.read_csv(membership, index_col=0).index.tolist()
        i = names.index("IPMN_EA")
        for period, (start, end) in PERIODS.items():
            for origin in pd.period_range(start, end, freq="M"):
                path = ROOT / panel / str(origin) / "parameters.npz"
                sources.append(path)
                with np.load(path) as saved:
                    loadings, variance = saved["loadings"], saved["variances"]
                    ar, innovation = saved["ar"], saved["innovation"]
                r = loadings.shape[1]
                assert r == 3 and ar.shape == (r, 2 * r)
                companion = np.zeros((2 * r, 2 * r))
                companion[:r], companion[r:, :r] = ar, np.eye(r)
                shocks = np.zeros_like(companion)
                shocks[:r, :r] = innovation
                rho = np.abs(np.linalg.eigvals(companion)).max()
                assert rho < 1
                covariance = solve_discrete_lyapunov(companion, shocks)
                np.testing.assert_allclose(
                    covariance, companion @ covariance @ companion.T + shocks, rtol=1e-9, atol=1e-10
                )
                common = float(loadings[i] @ covariance[:r, :r] @ loadings[i])
                total = common + variance[i]
                share = common / total
                assert common >= 0 and variance[i] > 0 and 0 <= share <= 1
                rows.append(
                    {
                        "panel": panel,
                        "period": period,
                        "origin": str(origin),
                        "common_variance": common,
                        "idiosyncratic_variance": variance[i],
                        "share": share,
                        "rho": rho,
                    }
                )
    data = pd.DataFrame(rows)
    summary = (
        data.groupby(["period", "panel"])
        .agg(n=("share", "size"), median_share=("share", "median"))
        .reset_index()
    )
    assert summary.n.eq(60).all()
    for panel in ["M", "MQ"]:
        assert (
            data[data.panel.eq(panel)].origin.tolist() == data[data.panel.eq("M")].origin.tolist()
        )
    OUT.mkdir(parents=True)
    data.to_csv(OUT / "monthly_shares.csv", index=False)
    summary["median_percent"] = 100 * summary.median_share
    summary.to_csv(OUT / "summary.csv", index=False)
    old = _style.OUT
    try:
        _style.OUT = OUT
        for row in summary.itertuples():
            _style.record_number(f"ManufacturingCommon{row.panel}{row.period}", f"{row.median_percent:.1f}")
        table = summary.pivot(index="period", columns="panel", values="median_percent").reindex(
            ["Early", "Late"]
        )
        table.index = ["2015--2019", "2021--2025"]
        _style.save_table(
            table,
            "tab_111_manufacturing_common_shares",
            float_format="%.1f",
            notes="Median model-implied common variance share of manufacturing production, in percent, across 60 matched forecast origins per period and information set. Three factors and VAR(2); saved nonzero-IQR parameters only. Common share is not out-of-sample forecast accuracy. Source: saved paper experiment.",
        )
    finally:
        _style.OUT = old
    text = [
        "# Manufacturing inflation: common variance share in M and MQ",
        "No model is re-estimated. This repeats the measure in scripts/sparse_diagnostics.py for both panels on the same forecast origins. Each share uses the parameters estimated at that origin from its expanding training sample, not a fit to the labeled subperiod alone. The periods label forecast origins; 2020 origins are excluded, but pandemic observations remain in subsequent training samples.",
        markdown(summary.round(6)),
        "For each saved factor VAR(2), its stationary covariance is obtained from the discrete Lyapunov equation. The share is lambda_manufacturing' Sigma_F lambda_manufacturing / (lambda_manufacturing' Sigma_F lambda_manufacturing + idiosyncratic variance). It is model-implied and rotation-invariant, not a sample reconstruction R-squared or a forecasting R-squared. Target standardization cancels in the ratio. Both panels retain exactly three factors.",
        "A higher share means that the fitted model assigns more of manufacturing's variance to its common component. It does not by itself demonstrate a more accurate latent factor estimate or better out-of-sample prediction. The already observed reduction in DFM forecast MSE is separate, complementary evidence; no causal or statistical significance test of the share difference is conducted.",
        "Checks: identical origin sets; 60 origins per period and panel; stable VARs; Lyapunov residual identity; positive variances and shares in [0,1]. Reproduce with python -m scripts.manufacturing_common_share_comparison in a fresh output directory.",
    ]
    (OUT / "analysis.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT / "audit.json").write_text(
        json.dumps(
            {"sources": hashes, "saved_parameter_windows": len(data), "model_refits": 0}, indent=2
        ),
        encoding="utf-8",
    )
    _style.record_output(
        "manufacturing_common_share_comparison",
        [str(OUT / "analysis.md"), str(OUT / "numbers.tex")],
        "Results 4.2.1: M/MQ manufacturing common variance shares, distinct from forecast accuracy",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
