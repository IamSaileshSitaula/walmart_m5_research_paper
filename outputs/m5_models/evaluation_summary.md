# M5 Direct Multi-Horizon Forecasting: Fold-Provenance Audit & Evaluation Report

**Date**: 2026-10-20 11:56:52  
**Study Population**: Systematic 10% item sample (3,050 item–store series; 305 items across 10 stores)  
**Total Out-of-Sample Predictions**: 4,192,916 rows across 54 evaluation origins  
**Target Horizons**: $h = 1, \dots, 28$ direct multi-horizon (forecast lead $h$ included in predictors)  

## 1. Fold-Provenance and Temporal Boundary Audit

Every fold's predictions were generated strictly by models trained on origins dated prior to that fold's evaluation window.
Price normalization features were recomputed using fold-specific training medians (zero future price leakage).

| Fold | Role | Model Used | Max Training Target Date | Min Evaluated Target Date | Temporal Gap | Gate | N Rows | MAE Bench | MAE A | MAE B | MAE C | MAE D | $\Delta$ (B - A) | $\Delta$ (D - B) |
|---|---|---|---|---|:---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `DEV_S2012` | chronological_dev | `fold_specific_models_DEV_S2012` | 2012-10-14 (d_625) | 2012-10-26 (d_637) | +12d | **PASS (gap >= 1d)** | 602,952 | 1.4670 | 1.1169 | 1.1169 | 1.1175 | 1.1152 | -0.00% | -0.15% |
| `DEV_S2013` | chronological_dev | `fold_specific_models_DEV_S2013` | 2013-10-13 (d_989) | 2013-11-01 (d_1008) | +19d | **PASS (gap >= 1d)** | 724,808 | 1.4577 | 1.0874 | 1.0698 | 1.0724 | 1.0748 | -1.62% | +0.47% |
| `DEV_S2014` | chronological_dev | `fold_specific_models_DEV_S2014` | 2014-10-12 (d_1353) | 2014-10-31 (d_1372) | +19d | **PASS (gap >= 1d)** | 825,888 | 1.2252 | 0.9093 | 0.8969 | 0.8972 | 0.8989 | -1.37% | +0.22% |
| `DEV_ORD` | chronological_dev | `fold_specific_models_DEV_ORD` | 2015-02-01 (d_1465) | 2015-02-02 (d_1466) | +1d | **PASS (gap >= 1d)** | 760,144 | 1.3005 | 0.9588 | 0.9546 | 0.9572 | 0.9594 | -0.43% | +0.50% |
| `FINAL_TEST` | frozen_final_test | `fold_specific_models_FINAL_TEST` | 2015-10-11 (d_1717) | 2015-10-30 (d_1736) | +19d | **PASS (gap >= 1d)** | 1,279,124 | 1.3147 | 0.9699 | 0.9580 | 0.9588 | 0.9610 | -1.22% | +0.31% |

## 2. Frozen Final Test Performance (Holiday Season 2015 + Evaluation Period)

> [!IMPORTANT]
> **Standalone Untouched Test Result**: The final test fold (`FINAL_TEST`) begins at target day $d\_1736$ (2015-10-30).
> Model features, objective selection (`regression_l1`), hyperparameters, and clipping rules were frozen prior to opening these test labels.

- **Evaluation Scope**: 15 rolling forecast origins ($O \in [1735, 1913]$, targets $d \in [1736, 1941]$) spanning Christmas 2015, Thanksgiving 2015, and the official M5 public evaluation period.
- **Total Test Predictions**: 1,279,124 rows.
- **Seasonal-Naive Benchmark MAE**: **1.3147** (Bias: -0.003).
- **Model A (No Events) MAE**: **0.9699** (Bias: -0.472).
- **Model B (Same-Day Events) MAE**: **0.9580** (-1.22% vs A; Bias: -0.463).
- **Model C (Event Phase & Timing) MAE**: **0.9588** (+0.09% vs B; Bias: -0.470).
- **Model D (Same-Day Events + Event Windows) MAE**: **0.9610** (+0.31% vs B; marginal value of adding windows to same-day events).

## 3. Holiday Episode-Level Paired Comparisons

To avoid pseudo-replication from treating dependent SKU rows around the same holiday as independent evidence, performance is aggregated by holiday episode across the 4 observed seasons:

| Holiday Episode | N Rows | Mean Observed | MAE A (No Event) | MAE B (Same-Day) | MAE C (Event Phase) | MAE D (Event Windows) | $\Delta$ (B - A) % | $\Delta$ (D - B) % | D Outperforms B? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Christmas_2012` | 211,677 | 1.53 | 1.1005 | 1.1001 | 1.1027 | 1.0988 | -0.03% | -0.12% | **Yes** |
| `Christmas_2013` | 272,538 | 1.35 | 1.0508 | 1.0103 | 1.0131 | 1.0119 | -3.86% | +0.16% | No |
| `Christmas_2014` | 304,099 | 1.16 | 0.8805 | 0.8493 | 0.8487 | 0.8506 | -3.54% | +0.16% | No |
| `Christmas_2015` | 307,551 | 1.32 | 0.9558 | 0.9159 | 0.9172 | 0.9168 | -4.17% | +0.10% | No |
| `DEV_ORD_NonHoliday` | 760,144 | 1.33 | 0.9588 | 0.9546 | 0.9572 | 0.9594 | -0.43% | +0.50% | No |
| `DEV_S2012_NonHoliday` | 195,874 | 1.77 | 1.1853 | 1.1838 | 1.1852 | 1.1850 | -0.13% | +0.10% | No |
| `DEV_S2013_NonHoliday` | 186,356 | 1.60 | 1.1483 | 1.1449 | 1.1457 | 1.1559 | -0.29% | +0.97% | No |
| `DEV_S2014_NonHoliday` | 224,162 | 1.38 | 0.9869 | 0.9881 | 0.9875 | 0.9917 | +0.13% | +0.36% | No |
| `FINAL_TEST_NonHoliday` | 667,282 | 1.46 | 1.0003 | 0.9979 | 0.9991 | 1.0012 | -0.25% | +0.34% | No |
| `Thanksgiving_2012` | 195,401 | 1.53 | 1.0662 | 1.0679 | 1.0657 | 1.0630 | +0.16% | -0.46% | **Yes** |
| `Thanksgiving_2013` | 265,914 | 1.47 | 1.0823 | 1.0781 | 1.0818 | 1.0823 | -0.39% | +0.39% | No |
| `Thanksgiving_2014` | 297,627 | 1.21 | 0.8803 | 0.8767 | 0.8787 | 0.8783 | -0.41% | +0.17% | No |
| `Thanksgiving_2015` | 304,291 | 1.30 | 0.9172 | 0.9133 | 0.9127 | 0.9174 | -0.43% | +0.45% | No |

## 4. Underforecasting Bias Investigation (Mean Error ~ -0.44 Units)

> [!WARNING]
> **Underforecasting Root Cause Analysis**: All LightGBM models trained with `regression_l1` exhibit a consistent negative mean forecast error (Bias $\approx -0.44$ units).
> This is a direct mathematical consequence of L1 loss targeting the **conditional median** rather than the conditional mean on right-skewed, zero-inflated retail demand distributions.

| Segment Type | Stratum | Model | Mean Observed | Mean Forecast | Bias (Forecast - Obs) | Relative Bias % | MAE |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| Overall | All Predictions | Seasonal Naive | 1.39 | 1.41 | **+0.018** | +1.3% | 1.3411 |
| Overall | All Predictions | Model A (No Events) | 1.39 | 0.95 | **-0.439** | -31.5% | 0.9974 |
| Overall | All Predictions | Model B (Same-Day) | 1.39 | 0.96 | **-0.431** | -31.0% | 0.9875 |
| Overall | All Predictions | Model C (Event Phase) | 1.39 | 0.95 | **-0.439** | -31.6% | 0.9889 |
| Overall | All Predictions | Model D (Event Windows) | 1.39 | 0.94 | **-0.450** | -32.3% | 0.9903 |
| Volume Tier | Low Volume | Model A | 0.19 | 0.00 | **-0.187** | -99.0% | 0.1902 |
| Volume Tier | Low Volume | Model B | 0.19 | 0.00 | **-0.187** | -99.1% | 0.1901 |
| Volume Tier | Low Volume | Model C | 0.19 | 0.00 | **-0.187** | -99.0% | 0.1903 |
| Volume Tier | Low Volume | Model D | 0.19 | 0.00 | **-0.187** | -98.9% | 0.1903 |
| Volume Tier | Medium Volume | Model A | 0.60 | 0.15 | **-0.449** | -74.6% | 0.5949 |
| Volume Tier | Medium Volume | Model B | 0.60 | 0.16 | **-0.442** | -73.5% | 0.5929 |
| Volume Tier | Medium Volume | Model C | 0.60 | 0.15 | **-0.449** | -74.6% | 0.5933 |
| Volume Tier | Medium Volume | Model D | 0.60 | 0.15 | **-0.448** | -74.5% | 0.5936 |
| Volume Tier | High Volume | Model A | 3.39 | 2.71 | **-0.681** | -20.1% | 2.2071 |
| Volume Tier | High Volume | Model B | 3.39 | 2.72 | **-0.664** | -19.6% | 2.1796 |
| Volume Tier | High Volume | Model C | 3.39 | 2.70 | **-0.682** | -20.1% | 2.1830 |
| Volume Tier | High Volume | Model D | 3.39 | 2.67 | **-0.715** | -21.1% | 2.1870 |
| Event Phase | pre_1 | Model A | 1.37 | 0.84 | **-0.534** | -38.9% | 1.0161 |
| Event Phase | pre_1 | Model B | 1.37 | 0.88 | **-0.493** | -35.9% | 1.0147 |
| Event Phase | pre_1 | Model C | 1.37 | 0.85 | **-0.521** | -38.0% | 1.0171 |
| Event Phase | pre_1 | Model D | 1.37 | 0.86 | **-0.512** | -37.3% | 1.0151 |
| Event Phase | pre_7_2 | Model A | 1.42 | 0.96 | **-0.456** | -32.2% | 1.0039 |
| Event Phase | pre_7_2 | Model B | 1.42 | 0.98 | **-0.432** | -30.5% | 1.0027 |
| Event Phase | pre_7_2 | Model C | 1.42 | 0.99 | **-0.432** | -30.5% | 1.0042 |
| Event Phase | pre_7_2 | Model D | 1.42 | 0.97 | **-0.446** | -31.5% | 1.0043 |
| Event Phase | day_0 | Model A | 0.98 | 0.88 | **-0.096** | -9.8% | 0.9008 |
| Event Phase | day_0 | Model B | 0.98 | 0.73 | **-0.253** | -25.8% | 0.7772 |
| Event Phase | day_0 | Model C | 0.98 | 0.74 | **-0.236** | -24.1% | 0.7778 |
| Event Phase | day_0 | Model D | 0.98 | 0.72 | **-0.261** | -26.6% | 0.7734 |
| Event Phase | post_1_3 | Model A | 1.32 | 1.03 | **-0.283** | -21.5% | 0.9497 |
| Event Phase | post_1_3 | Model B | 1.32 | 1.05 | **-0.265** | -20.1% | 0.9499 |
| Event Phase | post_1_3 | Model C | 1.32 | 0.99 | **-0.326** | -24.7% | 0.9456 |
| Event Phase | post_1_3 | Model D | 1.32 | 1.00 | **-0.314** | -23.8% | 0.9481 |
| Event Phase | outside | Model A | 1.46 | 0.96 | **-0.496** | -34.1% | 1.0182 |
| Event Phase | outside | Model B | 1.46 | 0.98 | **-0.479** | -32.9% | 1.0154 |
| Event Phase | outside | Model C | 1.46 | 0.97 | **-0.485** | -33.3% | 1.0170 |
| Event Phase | outside | Model D | 1.46 | 0.96 | **-0.495** | -34.0% | 1.0199 |

### Operational Inventory Implications of Median Forecasting
- **Stockout Risk**: An unadjusted L1 median forecast ($\hat{y}_{50}$) will stock out on approximately 50% of replenishment cycles for intermittent items where demand exceeds the median.
- **High-Volume Items**: For high-volume items, underforecasting reaches -20% to -30% of sales volume.
- **Pre-Holiday Demand Surges**: On Day -1 (`pre_1`, Thanksgiving/Christmas Eve), demand surges while models forecast the median, creating an acute shortfall.
- **Recommendation for Inventory Simulation**: For downstream inventory simulation, median forecasts must be combined with an asymmetric quantile buffer ($\hat{y}_q$ for $q \ge 0.75$) or safety stock calculated from residual variance, rather than raw point predictions.

## 5. SKU Attention Rule: Summary of Findings

> [!NOTE]
> Detailed audit and SKU-by-SKU outcomes are saved in [sku_attention_report.md](file:///outputs/m5_models/sku_attention_report.md).
- **Calibration Scope**: Calibrated strictly on development holiday episodes (`DEV_S2012`–`DEV_S2014`) by ranking 28-day cumulative order discrepancies between Model B and Model D.
- **Review List Size**: Top 50 highest-discrepancy SKU-store pairs (1.6% of series).
- **Exploratory 2015 Evaluation**: Examines whether attending to high-discrepancy SKUs improves 2015 holiday forecasting, while rigorously reporting regressions where Model D underperforms Model B.