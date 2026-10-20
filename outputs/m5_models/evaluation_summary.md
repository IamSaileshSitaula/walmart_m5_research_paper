# M5 Direct Multi-Horizon Forecasting: Fold-Provenance Audit & Evaluation Report

**Date**: 2026-10-20 12:56:20  
**Study Population**: Systematic 10% item sample (3,050 item–store series; 305 items across 10 stores)  
**Total Out-of-Sample Predictions**: 4,192,916 rows across 54 evaluation origins  
**Target Horizons**: $h = 1, \dots, 28$ direct multi-horizon (forecast lead $h$ included in predictors)  

## 1. Fold-Provenance and Temporal Boundary Audit

Every fold's predictions were generated strictly by models trained on origins dated prior to that fold's evaluation window.
Price normalization features were recomputed using fold-specific training medians (zero future price leakage).

| Fold | Role | Model Used | Max Training Target Date | Min Evaluated Target Date | Temporal Gap | Gate | N Rows | MAE Bench | MAE A | MAE B | MAE C | MAE D | $\Delta$ (B - A) | $\Delta$ (D - B) |
|---|---|---|---|---|:---:|:---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `DEV_S2012` | chronological_dev | `fold_specific_models_DEV_S2012` | 2012-10-14 (d_625) | 2012-10-26 (d_637) | +12d | **PASS (gap >= 1d)** | 602,952 | 1.4670 | 1.1161 | 1.1147 | 1.1163 | 1.1159 | -0.13% | +0.11% |
| `DEV_S2013` | chronological_dev | `fold_specific_models_DEV_S2013` | 2013-10-13 (d_989) | 2013-11-01 (d_1008) | +19d | **PASS (gap >= 1d)** | 724,808 | 1.4577 | 1.0891 | 1.0741 | 1.0757 | 1.0709 | -1.38% | -0.29% |
| `DEV_S2014` | chronological_dev | `fold_specific_models_DEV_S2014` | 2014-10-12 (d_1353) | 2014-10-31 (d_1372) | +19d | **PASS (gap >= 1d)** | 825,888 | 1.2252 | 0.9098 | 0.9005 | 0.8979 | 0.8966 | -1.02% | -0.43% |
| `DEV_ORD` | chronological_dev | `fold_specific_models_DEV_ORD` | 2015-02-01 (d_1465) | 2015-02-02 (d_1466) | +1d | **PASS (gap >= 1d)** | 760,144 | 1.3005 | 0.9533 | 0.9523 | 0.9565 | 0.9529 | -0.11% | +0.06% |
| `FINAL_TEST` | frozen_final_test | `fold_specific_models_FINAL_TEST` | 2015-10-11 (d_1717) | 2015-10-30 (d_1736) | +19d | **PASS (gap >= 1d)** | 1,279,124 | 1.3147 | 0.9699 | 0.9573 | 0.9598 | 0.9585 | -1.30% | +0.12% |

## 2. 2015 Exploratory Test Evaluation (Holiday Season 2015 + Evaluation Period)

> [!IMPORTANT]
> **Exploratory Evaluation Status**: The final test fold (`FINAL_TEST`) evaluates rolling origins from target day $d\_1736$ (2015-10-30) onward.
> Because aggregate 2015 metrics were inspected in preliminary research rounds, this test evaluation is designated as **EXPLORATORY** rather than an untouched confirmatory test.
> Models were trained strictly on observations dated prior to test start with frozen hyperparameters, and event screening was conducted chronologically using history up to cutoff day $d\_1717$.

- **Evaluation Scope**: 15 rolling forecast origins ($O \in [1735, 1913]$, targets $d \in [1736, 1941]$) spanning Christmas 2015, Thanksgiving 2015, and the official M5 public evaluation period.
- **Total Test Predictions**: 1,279,124 rows.
- **Seasonal-Naive Benchmark MAE**: **1.3147** (Bias: -0.003).
- **Model A (No Events) MAE**: **0.9699** (Bias: -0.466).
- **Model B (Same-Day Events) MAE**: **0.9573** (-1.30% vs A; Bias: -0.457).
- **Model C (Event Phase & Timing) MAE**: **0.9598** (+0.26% vs B; Bias: -0.463).
- **Model D (Same-Day Events + Event Windows) MAE**: **0.9585** (+0.12% vs B; marginal value of adding windows to same-day events).

## 3. Holiday Episode-Level Paired Comparisons

To avoid pseudo-replication from treating dependent SKU rows around the same holiday as independent evidence, performance is aggregated by holiday episode across the 4 observed seasons:

| Holiday Episode | N Rows | Mean Observed | MAE A (No Event) | MAE B (Same-Day) | MAE C (Event Phase) | MAE D (Event Windows) | $\Delta$ (B - A) % | $\Delta$ (D - B) % | D Outperforms B? |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Christmas_2012` | 211,677 | 1.53 | 1.0990 | 1.0985 | 1.1002 | 1.0979 | -0.05% | -0.06% | **Yes** |
| `Christmas_2013` | 272,538 | 1.35 | 1.0507 | 1.0133 | 1.0210 | 1.0127 | -3.55% | -0.07% | **Yes** |
| `Christmas_2014` | 304,099 | 1.16 | 0.8813 | 0.8524 | 0.8509 | 0.8486 | -3.28% | -0.45% | **Yes** |
| `Christmas_2015` | 307,551 | 1.32 | 0.9560 | 0.9157 | 0.9183 | 0.9143 | -4.21% | -0.15% | **Yes** |
| `DEV_ORD_NonHoliday` | 760,144 | 1.33 | 0.9533 | 0.9523 | 0.9565 | 0.9529 | -0.11% | +0.06% | No |
| `DEV_S2012_NonHoliday` | 195,874 | 1.77 | 1.1862 | 1.1813 | 1.1834 | 1.1852 | -0.41% | +0.33% | No |
| `DEV_S2013_NonHoliday` | 186,356 | 1.60 | 1.1499 | 1.1503 | 1.1468 | 1.1446 | +0.03% | -0.50% | **Yes** |
| `DEV_S2014_NonHoliday` | 224,162 | 1.38 | 0.9870 | 0.9935 | 0.9870 | 0.9889 | +0.65% | -0.46% | **Yes** |
| `FINAL_TEST_NonHoliday` | 667,282 | 1.46 | 1.0003 | 0.9968 | 0.9991 | 0.9990 | -0.35% | +0.21% | No |
| `Thanksgiving_2012` | 195,401 | 1.53 | 1.0643 | 1.0654 | 1.0666 | 1.0658 | +0.10% | +0.04% | No |
| `Thanksgiving_2013` | 265,914 | 1.47 | 1.0860 | 1.0830 | 1.0819 | 1.0790 | -0.28% | -0.36% | **Yes** |
| `Thanksgiving_2014` | 297,627 | 1.21 | 0.8806 | 0.8797 | 0.8788 | 0.8762 | -0.11% | -0.40% | **Yes** |
| `Thanksgiving_2015` | 304,291 | 1.30 | 0.9171 | 0.9125 | 0.9155 | 0.9142 | -0.51% | +0.19% | No |

## 4. Underforecasting Bias Investigation (Mean Error ~ -0.44 Units)

> [!WARNING]
> **Underforecasting Root Cause Analysis**: All LightGBM models trained with `regression_l1` exhibit a consistent negative mean forecast error (Bias $\approx -0.44$ units).
> This is a direct mathematical consequence of L1 loss targeting the **conditional median** rather than the conditional mean on right-skewed, zero-inflated retail demand distributions.

| Segment Type | Stratum | Model | Mean Observed | Mean Forecast | Bias (Forecast - Obs) | Relative Bias % | MAE |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| Overall | All Predictions | Seasonal Naive | 1.39 | 1.41 | **+0.018** | +1.3% | 1.3411 |
| Overall | All Predictions | Model A (No Events) | 1.39 | 0.96 | **-0.433** | -31.1% | 0.9967 |
| Overall | All Predictions | Model B (Same-Day) | 1.39 | 0.96 | **-0.430** | -30.9% | 0.9880 |
| Overall | All Predictions | Model C (Event Phase) | 1.39 | 0.96 | **-0.432** | -31.0% | 0.9896 |
| Overall | All Predictions | Model D (Event Windows) | 1.39 | 0.96 | **-0.431** | -30.9% | 0.9873 |
| Volume Tier | Low Volume | Model A | 0.19 | 0.00 | **-0.187** | -98.7% | 0.1903 |
| Volume Tier | Low Volume | Model B | 0.19 | 0.00 | **-0.187** | -98.7% | 0.1904 |
| Volume Tier | Low Volume | Model C | 0.19 | 0.00 | **-0.186** | -98.6% | 0.1905 |
| Volume Tier | Low Volume | Model D | 0.19 | 0.00 | **-0.186** | -98.6% | 0.1904 |
| Volume Tier | Medium Volume | Model A | 0.60 | 0.16 | **-0.445** | -73.9% | 0.5956 |
| Volume Tier | Medium Volume | Model B | 0.60 | 0.17 | **-0.434** | -72.2% | 0.5934 |
| Volume Tier | Medium Volume | Model C | 0.60 | 0.16 | **-0.442** | -73.6% | 0.5947 |
| Volume Tier | Medium Volume | Model D | 0.60 | 0.17 | **-0.434** | -72.1% | 0.5933 |
| Volume Tier | High Volume | Model A | 3.39 | 2.72 | **-0.668** | -19.7% | 2.2042 |
| Volume Tier | High Volume | Model B | 3.39 | 2.72 | **-0.668** | -19.7% | 2.1803 |
| Volume Tier | High Volume | Model C | 3.39 | 2.72 | **-0.667** | -19.7% | 2.1834 |
| Volume Tier | High Volume | Model D | 3.39 | 2.72 | **-0.671** | -19.8% | 2.1782 |
| Event Phase | pre_1 | Model A | 1.37 | 0.83 | **-0.538** | -39.3% | 1.0164 |
| Event Phase | pre_1 | Model B | 1.37 | 0.87 | **-0.500** | -36.4% | 1.0152 |
| Event Phase | pre_1 | Model C | 1.37 | 0.84 | **-0.535** | -39.1% | 1.0209 |
| Event Phase | pre_1 | Model D | 1.37 | 0.87 | **-0.500** | -36.5% | 1.0134 |
| Event Phase | pre_7_2 | Model A | 1.42 | 0.96 | **-0.458** | -32.3% | 1.0046 |
| Event Phase | pre_7_2 | Model B | 1.42 | 0.98 | **-0.436** | -30.8% | 1.0037 |
| Event Phase | pre_7_2 | Model C | 1.42 | 0.99 | **-0.426** | -30.1% | 1.0057 |
| Event Phase | pre_7_2 | Model D | 1.42 | 0.99 | **-0.427** | -30.2% | 1.0033 |
| Event Phase | day_0 | Model A | 0.98 | 0.89 | **-0.091** | -9.3% | 0.8975 |
| Event Phase | day_0 | Model B | 0.98 | 0.73 | **-0.247** | -25.2% | 0.7771 |
| Event Phase | day_0 | Model C | 0.98 | 0.75 | **-0.232** | -23.7% | 0.7772 |
| Event Phase | day_0 | Model D | 0.98 | 0.73 | **-0.251** | -25.6% | 0.7705 |
| Event Phase | post_1_3 | Model A | 1.32 | 1.03 | **-0.286** | -21.7% | 0.9501 |
| Event Phase | post_1_3 | Model B | 1.32 | 1.05 | **-0.270** | -20.5% | 0.9517 |
| Event Phase | post_1_3 | Model C | 1.32 | 0.99 | **-0.325** | -24.7% | 0.9478 |
| Event Phase | post_1_3 | Model D | 1.32 | 1.00 | **-0.316** | -24.0% | 0.9472 |
| Event Phase | outside | Model A | 1.46 | 0.97 | **-0.483** | -33.2% | 1.0163 |
| Event Phase | outside | Model B | 1.46 | 0.98 | **-0.472** | -32.4% | 1.0150 |
| Event Phase | outside | Model C | 1.46 | 0.98 | **-0.473** | -32.5% | 1.0166 |
| Event Phase | outside | Model D | 1.46 | 0.99 | **-0.470** | -32.3% | 1.0153 |

### Operational Inventory Implications of Median Forecasting
- **Stockout Risk**: An unadjusted L1 median forecast ($\hat{y}_{50}$) will stock out on approximately 50% of replenishment cycles for intermittent items where demand exceeds the median.
- **High-Volume Items**: For high-volume items, underforecasting reaches -20% to -30% of sales volume.
- **Pre-Holiday Demand Surges**: On Day -1 (`pre_1`, Thanksgiving/Christmas Eve), demand surges while models forecast the median, creating an acute shortfall.
- **Recommendation for Inventory Simulation**: For downstream inventory simulation, median forecasts must be combined with an asymmetric quantile buffer ($\hat{y}_q$ for $q \ge 0.75$) or safety stock calculated from residual variance, rather than raw point predictions.

## 5. SKU Attention Rule: Summary of Findings

> [!NOTE]
> Detailed audit and SKU-by-SKU outcomes are saved in [sku_attention_report.md](file:///outputs/m5_models/sku_attention_report.md).
- **Calibration Scope**: Calibrated strictly on development holiday origins (`DEV_S2012`–`DEV_S2014`) by ranking 28-day cumulative forecast disagreement between Model B and Model D.
- **Operational Meaning**: The score measures forecast-implied 28-day order quantity difference / forecast disagreement, NOT actual inventory orders.
- **Review List Capacity**: Fixed at Top 50 highest-disagreement SKU-store pairs (1.6% of series).
- **Exploratory 2015 Evaluation**: Because 2015 results were previously inspected, this analysis is strictly EXPLORATORY. Model D is NOT automatically superior for flagged SKUs; disagreement flags SKUs for closer human review.
- **No Premature Inventory Claims**: No claim is made that the attention rule increases profit or reduces stockouts; those operational outcomes require a downstream inventory simulation.