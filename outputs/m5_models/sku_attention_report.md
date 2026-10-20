# SKU Attention Rule: Development Calibration & 2015 Exploratory Evaluation

> [!IMPORTANT]
> **Governance & Exploratory Status Disclosure**:
> The attention rule ranking was calibrated **strictly using 2012–2014 development holiday episodes** (`DEV_S2012`, `DEV_S2013`, `DEV_S2014`).
> However, because aggregate 2015 test metrics were inspected in prior research rounds, the evaluation of this attention rule on 2015 is designated as **exploratory**, rather than an untouched confirmatory validation.

## 1. Attention Rule Formulation
- **Operational Context**: Store inventory managers cannot review all 3,050 series in the catalog.
- **Selection Metric**: Cumulative 28-day inventory order decision discrepancy during historical development holidays:
  $$\text{Decision Diff}_s = \frac{1}{M_s} \sum_{O \in \text{DevHolidays}} \left| \sum_{h=1}^{28} \hat{y}_{D, s, O, h} - \sum_{h=1}^{28} \hat{y}_{B, s, O, h} \right|$$
- **Review List Capacity**: Limited to the **Top 50 highest-discrepancy SKU–store pairs** (1.6% of the 3,050 series).

## 2. Overall 2015 Holiday Performance (Thanksgiving 2015 & Christmas 2015)

| Catalog Segment | Series Count | Model B MAE (Same-Day) | Model D MAE (Event-Window) | Absolute Delta (D - B) | Relative Delta % |
|---|:---:|:---:|:---:|:---:|:---:|
| **Attention Review List (Top 50)** | 50 | 5.5430 | 5.6298 | +0.0868 | **+1.57%** |
| **Remaining Catalog (Unreviewed)** | 2,904 | 0.8390 | 0.8401 | +0.0011 | +0.13% |
| **All Holiday Series** | 2,954 | 0.9146 | 0.9171 | +0.0025 | +0.27% |

- **Attention List Win Rate**: **17/50 SKUs (34.0%)** showed lower MAE under Model D.
- **Regressions (Made Worse)**: **33/50 SKUs (66.0%)** exhibited higher error under Model D.
- **Neutral / Tied**: **0/50 SKUs**.

## 3. Analysis of SKUs Where Attention Made Things Worse (Regressions)

To ensure managerial transparency, any automated alerting system must expose false alarms and performance degradation:

| Series ID | Store | Category | Dev Order Delta | Test 2015 Obs Mean | MAE Model B | MAE Model D | Error Increase (D - B) | Regression Diagnosis |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `FOODS_3_318_CA_3` | CA_3 | FOODS | 22.18 units | 0.00 | 2.9353 | 3.5691 | **+0.6338** | Overforecasted pre-holiday ramp |
| `FOODS_3_318_CA_2` | CA_2 | FOODS | 8.38 units | 0.00 | 1.1402 | 1.7059 | **+0.5657** | Overforecasted pre-holiday ramp |
| `FOODS_3_318_WI_1` | WI_1 | FOODS | 10.52 units | 0.00 | 1.1553 | 1.7125 | **+0.5573** | Overforecasted pre-holiday ramp |
| `FOODS_3_318_WI_2` | WI_2 | FOODS | 22.59 units | 0.00 | 3.3775 | 3.9344 | **+0.5569** | Overforecasted pre-holiday ramp |
| `FOODS_3_318_WI_3` | WI_3 | FOODS | 27.66 units | 0.00 | 2.3968 | 2.9292 | **+0.5324** | Overforecasted pre-holiday ramp |
| `FOODS_3_288_CA_3` | CA_3 | FOODS | 101.91 units | 35.34 | 8.1122 | 8.6413 | **+0.5290** | Underforecasted post-holiday drop |
| `FOODS_3_318_CA_1` | CA_1 | FOODS | 14.19 units | 0.00 | 1.4910 | 1.9914 | **+0.5004** | Overforecasted pre-holiday ramp |
| `FOODS_3_288_CA_2` | CA_2 | FOODS | 15.57 units | 30.18 | 9.3396 | 9.5902 | **+0.2506** | Underforecasted post-holiday drop |
| `FOODS_3_007_WI_3` | WI_3 | FOODS | 11.78 units | 16.37 | 9.6928 | 9.8804 | **+0.1876** | Underforecasted post-holiday drop |
| `FOODS_3_498_WI_2` | WI_2 | FOODS | 21.78 units | 16.26 | 18.8778 | 19.0085 | **+0.1307** | Overforecasted pre-holiday ramp |

## 4. Top 15 Highest-Priority SKUs on the Attention List

| Rank | Series ID | Store | Category | Dev Decision Delta | 2015 MAE B | 2015 MAE D | 2015 $\Delta$ (D - B) | Outcome |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `FOODS_3_288_CA_3` | CA_3 | FOODS | 101.91 | 8.1122 | 8.6413 | +0.5290 | *WORSENED* |
| 2 | `FOODS_3_288_CA_1` | CA_1 | FOODS | 34.32 | 4.3010 | 4.4168 | +0.1158 | *WORSENED* |
| 3 | `FOODS_3_288_WI_2` | WI_2 | FOODS | 31.79 | 3.2853 | 3.2606 | -0.0247 | **IMPROVED** |
| 4 | `FOODS_3_007_WI_2` | WI_2 | FOODS | 29.32 | 18.9154 | 18.8949 | -0.0205 | **IMPROVED** |
| 5 | `FOODS_3_318_WI_3` | WI_3 | FOODS | 27.66 | 2.3968 | 2.9292 | +0.5324 | *WORSENED* |
| 6 | `FOODS_3_318_WI_2` | WI_2 | FOODS | 22.59 | 3.3775 | 3.9344 | +0.5569 | *WORSENED* |
| 7 | `FOODS_3_318_CA_3` | CA_3 | FOODS | 22.18 | 2.9353 | 3.5691 | +0.6338 | *WORSENED* |
| 8 | `FOODS_3_498_WI_2` | WI_2 | FOODS | 21.78 | 18.8778 | 19.0085 | +0.1307 | *WORSENED* |
| 9 | `FOODS_1_082_WI_2` | WI_2 | FOODS | 20.39 | 3.8445 | 3.8511 | +0.0066 | *WORSENED* |
| 10 | `FOODS_3_288_TX_1` | TX_1 | FOODS | 18.64 | 6.1913 | 6.2861 | +0.0949 | *WORSENED* |
| 11 | `FOODS_3_811_TX_2` | TX_2 | FOODS | 18.19 | 6.5832 | 6.6051 | +0.0219 | *WORSENED* |
| 12 | `FOODS_3_811_WI_2` | WI_2 | FOODS | 16.18 | 6.3935 | 6.4240 | +0.0305 | *WORSENED* |
| 13 | `HOUSEHOLD_1_418_CA_3` | CA_3 | HOUSEHOLD | 15.90 | 6.0798 | 6.0814 | +0.0016 | *WORSENED* |
| 14 | `FOODS_3_007_CA_3` | CA_3 | FOODS | 15.58 | 8.4087 | 8.4612 | +0.0525 | *WORSENED* |
| 15 | `FOODS_3_288_CA_2` | CA_2 | FOODS | 15.57 | 9.3396 | 9.5902 | +0.2506 | *WORSENED* |