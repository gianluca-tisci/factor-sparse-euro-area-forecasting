# Manufacturing inflation: common variance share in M and MQ

No model is re-estimated. This repeats the measure in scripts/sparse_diagnostics.py for both panels on the same forecast origins. Each share uses the parameters estimated at that origin from its expanding training sample, not a fit to the labeled subperiod alone. The periods label forecast origins; 2020 origins are excluded, but pandemic observations remain in subsequent training samples.

| period | panel | n | median_share | median_percent |
|---|---|---|---|---|
| Early | M | 60 | 0.867655 | 86.765506 |
| Early | MQ | 60 | 0.854618 | 85.461841 |
| Late | M | 60 | 0.931255 | 93.125502 |
| Late | MQ | 60 | 0.899425 | 89.942526 |

For each saved factor VAR(2), its stationary covariance is obtained from the discrete Lyapunov equation. The share is lambda_manufacturing' Sigma_F lambda_manufacturing / (lambda_manufacturing' Sigma_F lambda_manufacturing + idiosyncratic variance). It is model-implied and rotation-invariant, not a sample reconstruction R-squared or a forecasting R-squared. Target standardization cancels in the ratio. Both panels retain exactly three factors.

A higher share means that the fitted model assigns more of manufacturing's variance to its common component. It does not by itself demonstrate a more accurate latent factor estimate or better out-of-sample prediction. The already observed reduction in DFM forecast MSE is separate, complementary evidence; no causal or statistical significance test of the share difference is conducted.

Checks: identical origin sets; 60 origins per period and panel; stable VARs; Lyapunov residual identity; positive variances and shares in [0,1]. Reproduce with python -m scripts.manufacturing_common_share_comparison in a fresh output directory.
