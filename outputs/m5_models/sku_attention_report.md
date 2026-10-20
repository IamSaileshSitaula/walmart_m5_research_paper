# SKU Attention Rule: Development Calibration & 2015 Exploratory Evaluation

> [!IMPORTANT]
> **Governance & Exploratory Status Disclosure**:
> 1. **Ranking Protocol**: The SKU attention ranking was calibrated **strictly on 2012–2014 development holiday episodes** (`DEV_S2012`, `DEV_S2013`, `DEV_S2014`). Zero 2015 data entered SKU ranking or threshold selection.
> 2. **Exploratory Designation**: Because aggregate 2015 test metrics were inspected in prior research rounds, all 2015 results presented below are **EXPLORATORY**, not an untouched confirmatory test or prospective validation.
> 3. **Operational Meaning of Score**: The attention score represents a **forecast-implied 28-day order quantity difference** (forecast disagreement between Model B and Model D). It is **NOT** an actual inventory order, because the repository does not yet implement stock-on-hand, lead times, safety stocks, or an inventory reorder policy.
> 4. **Model Preference Disclaimer**: A top-50 flag identifies substantial disagreement between Model B and Model D; **it does NOT authorize switching to Model D**. Disagreement flags an SKU for closer managerial review; it does not prove Model D is more accurate.
> 5. **Unranked Policy Default**: The unranked 2015-active group's Model B fallback is a prospective policy definition for subsequent inventory simulation, NOT a claim that Model B increased profit.
> 6. **No Inventory Claims**: We do not claim that the top-50 rule increases profit or reduces stockouts; those operational outcomes require a subsequent inventory simulation.

## 1. Attention Rule Formulation & Population Accounting

- **Operational Context**: Store inventory managers cannot manually review all 3,050 series in the catalog. A disagreement rule flags the subset where event dynamics differ most between baseline Model B (same-day events) and Model D (event-window dynamics).
- **Eligibility Rule**: A forecast origin $O$ is designated as holiday-relevant if its 28-day forecast window $[O+1, O+28]$ intersects the predeclared Thanksgiving or Christmas analysis window $[-14, +7]$ days. Identified origins: 30 development origins (2012–2014) and 10 test origins (2015).
- **Full 28-Horizon Summation**: For every active series $s$ and holiday origin $O$, exactly 28 forecast horizons ($h = 1, \dots, 28$) are verified (duplicate and missing horizons strictly prohibited) and summed:
  $$Q_{B}(s, O) = \sum_{h=1}^{28} \hat{y}_{B, s, O, h}, \quad Q_{D}(s, O) = \sum_{h=1}^{28} \hat{y}_{D, s, O, h}$$
- **Forecast Disagreement Score**:
  $$\text{Disagreement}_s = \frac{1}{|\mathcal{O}_{\text{dev}, s}|} \sum_{O \in \mathcal{O}_{\text{dev}, s}} \left| Q_{D}(s, O) - Q_{B}(s, O) \right|$$
- **Review List Capacity**: Fixed at the **Top 50 highest-disagreement SKU–store pairs** (1.6% of the 3,050-series catalog).
- **Mutually Exclusive & Exhaustive Population Accounting**:
  - **Group 1: Top-50 Development-Ranked Series (Flagged for Review)**: **50 series** (14,000 forecast rows). Calibrated strictly on historical 2012–2014 development holiday episodes.
  - **Group 2: Other Development-Ranked Series (Unflagged)**: **2,904 series** (813,120 forecast rows). Active during 2012–2014 development episodes, ranked outside the top 50.
  - **Group 3: 2015-Active Series Without Dev Score (Model B Default)**: **92 series** (25,200 forecast rows). Series active in the 2015 holiday evaluation whose first positive sale occurred after the development holiday cutoff (late catalog introductions between late 2014 and 2015). Because they have zero development holiday history, they are not eligible for historical ranking. Their prospective policy assignment for subsequent inventory simulation is the baseline Model B default.
  - **Total 2015-Active Holiday Series**: **3,046 series** (852,320 forecast rows; sum of Groups 1 + 2 + 3).
  - **Sampled Series Without 2015 Holiday Evaluation (Inactive/Unobserved)**: **4 series** (0 holiday rows). Introduced in 2016 (first positive sale at day 1829 or later), so they had not yet had active sales during late-2015 holiday origins.
  - **Total Study Population**: **3,050 series** (3,046 active + 4 inactive = 3,050).

## 2. 2015 Exploratory Performance: Mutually Exclusive Population Accounting

| Catalog Segment | Active Series | Forecast Rows | Model B MAE | Model D MAE | $\Delta$ (D - B) | Relative $\Delta$ % | Model B RMSE | Model D RMSE | Model B Bias | Model D Bias |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top-50 Development-Ranked Series (Flagged for Review)** | 50 | 14,000 | 5.7461 | 5.7988 | **+0.0527** | **+0.92%** | 11.7396 | 11.8367 | -1.723 | -1.813 |
| **Other Development-Ranked Series (Unflagged)** | 2,904 | 813,120 | 0.8538 | 0.8539 | +0.0001 | +0.01% | 1.8926 | 1.8931 | -0.412 | -0.408 |
| **2015-Active Series Without Dev Score (Model B Default)** | 92 | 25,200 | 0.8205 | 0.8183 | -0.0021 | -0.26% | 1.6012 | 1.5929 | -0.481 | -0.455 |
| **All Active 2015 Holiday Series (Total Active)** | 3,046 | 852,320 | 0.9331 | 0.9340 | +0.0009 | +0.09% | 2.3993 | 2.4074 | -0.436 | -0.433 |
| *Sampled Series Without 2015 Holiday Evaluation (Inactive)* | 4 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Accounting Integrity Checks
- **Series Count Equality**: 50 (Top 50) + 2,904 (Other Ranked) + 92 (Unranked) = 3,046 active series (**Verified**).
- **Row Count Equality**: 14,000 + 813,120 + 25,200 = 852,320 forecast rows (**Verified**).
- **Weighted MAE Reproduction**: Row-weighted MAE across the three groups reproduces the all-active Model B MAE (0.933150 == 0.933150) and Model D MAE (0.934034 == 0.934034) (**Verified**).

### Top-50 Outcome Distribution & Effect Sizes
- **Improved (Model D beats Model B)**: **18/50 series (36.0%)**
- **Worsened (Model B beats Model D)**: **32/50 series (64.0%)**
- **Tied (|$\Delta$| < 1e-5)**: **0/50 series (0.0%)**
- **Mean Series-Level MAE Difference (D - B)**: **+0.0527** (Std Dev: 0.1074)
- **Standardized Effect Size (Cohen's d)**: **+0.491** (positive indicates Model D has higher error on average)
- **Paired t-test**: $t = 3.469, \; p = 0.0011$
- **Wilcoxon Signed-Rank Test**: $W = 341.0, \; p = 0.0037$

## 3. Unique Target Date Sensitivity Analysis

Because rolling forecast origins evaluate overlapping target dates with varying forecast horizons, we evaluate accuracy when predictions are aggregated by unique calendar target date $(s, t)$ across all mutually exclusive segments:

| Catalog Segment | Evaluation Level | Series Count | Forecast Rows | Model B MAE | Model D MAE | $\Delta$ (D - B) | Relative $\Delta$ % |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Top-50 Development-Ranked Series (Flagged for Review) | Origin-Horizon (Default) | 50 | 14,000 | 5.7461 | 5.7988 | +0.0527 | +0.92% |
| Top-50 Development-Ranked Series (Flagged for Review) | Unique Target Dates (Averaged) | 50 | 3,900 | 5.9360 | 5.9807 | +0.0447 | +0.75% |
| Other Development-Ranked Series (Unflagged) | Origin-Horizon (Default) | 2,904 | 813,120 | 0.8538 | 0.8539 | +0.0001 | +0.01% |
| Other Development-Ranked Series (Unflagged) | Unique Target Dates (Averaged) | 2,904 | 226,512 | 0.8552 | 0.8555 | +0.0002 | +0.03% |
| 2015-Active Series Without Dev Score (Model B Default) | Origin-Horizon (Default) | 92 | 25,200 | 0.8205 | 0.8183 | -0.0021 | -0.26% |
| 2015-Active Series Without Dev Score (Model B Default) | Unique Target Dates (Averaged) | 92 | 7,061 | 0.8376 | 0.8347 | -0.0029 | -0.35% |
| All Active 2015 Holiday Series | Origin-Horizon (Default) | 3,046 | 852,320 | 0.9331 | 0.9340 | +0.0009 | +0.09% |
| All Active 2015 Holiday Series | Unique Target Dates (Averaged) | 3,046 | 237,473 | 0.9382 | 0.9390 | +0.0009 | +0.09% |

## 4. Representative Improvements (SKUs Where Model D Reduced Error)

The attention score successfully flags high-volume seasonal items where event-window dynamics improved accuracy:

| Series ID | Store | Category | Dev Disagreement | 2015 Obs Mean | MAE Model B | MAE Model D | Error Reduction (D - B) | Relative % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `FOODS_2_326_WI_2` | WI_2 | FOODS | 42.46 units | 28.92 | 16.9903 | 16.8519 | **-0.1384** | +0.8% |
| `FOODS_3_811_CA_3` | CA_3 | FOODS | 22.95 units | 10.00 | 4.1791 | 4.0669 | **-0.1122** | +2.7% |
| `FOODS_1_113_WI_2` | WI_2 | FOODS | 25.97 units | 7.88 | 4.1155 | 4.0577 | **-0.0578** | +1.4% |
| `FOODS_3_498_WI_2` | WI_2 | FOODS | 44.97 units | 15.62 | 17.5879 | 17.5346 | **-0.0532** | +0.3% |
| `FOODS_3_288_TX_2` | TX_2 | FOODS | 22.37 units | 5.73 | 4.7654 | 4.7158 | **-0.0496** | +1.0% |

## 5. Representative Regressions (SKUs Where Model D Made Error Worse)

Crucially for managerial transparency, the attention score also flags false alarms where event-window features degraded accuracy:

| Series ID | Store | Category | Dev Disagreement | 2015 Obs Mean | MAE Model B | MAE Model D | Error Increase (D - B) | Diagnosis |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `FOODS_3_318_CA_3` | CA_3 | FOODS | 25.11 units | 0.36 | 3.1376 | 3.5133 | **+0.3757** | Overforecasted pre-holiday ramp |
| `FOODS_3_007_WI_3` | WI_3 | FOODS | 20.88 units | 20.07 | 12.0206 | 12.3588 | **+0.3383** | Underforecasted post-holiday drop |
| `FOODS_3_288_CA_3` | CA_3 | FOODS | 71.70 units | 36.21 | 8.5774 | 8.8177 | **+0.2403** | Underforecasted post-holiday drop |
| `FOODS_3_288_TX_1` | TX_1 | FOODS | 19.02 units | 10.28 | 6.1569 | 6.3942 | **+0.2373** | Underforecasted post-holiday drop |
| `FOODS_2_065_WI_2` | WI_2 | FOODS | 21.10 units | 6.90 | 3.6993 | 3.9206 | **+0.2213** | Overforecasted pre-holiday ramp |
| `FOODS_3_318_CA_4` | CA_4 | FOODS | 18.78 units | 0.24 | 1.3004 | 1.5072 | **+0.2068** | Overforecasted pre-holiday ramp |
| `FOODS_3_498_WI_3` | WI_3 | FOODS | 30.11 units | 19.81 | 10.5812 | 10.7783 | **+0.1970** | Underforecasted post-holiday drop |
| `FOODS_1_012_CA_2` | CA_2 | FOODS | 21.78 units | 15.09 | 5.9378 | 6.0889 | **+0.1511** | Underforecasted post-holiday drop |

## 6. Top 15 Highest-Priority SKUs on the Attention Review List

| Rank | Series ID | Store | Category | Dev Disagreement | 2015 MAE B | 2015 MAE D | 2015 $\Delta$ (D - B) | Outcome | Policy Default |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.0 | `FOODS_3_007_WI_2` | WI_2 | FOODS | 78.19 | 23.4631 | 23.4559 | -0.0072 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |
| 2.0 | `FOODS_3_288_CA_3` | CA_3 | FOODS | 71.70 | 8.5774 | 8.8177 | +0.2403 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 3.0 | `FOODS_3_318_WI_3` | WI_3 | FOODS | 63.09 | 2.6365 | 2.6858 | +0.0493 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 4.0 | `FOODS_3_318_WI_2` | WI_2 | FOODS | 47.50 | 3.7609 | 3.7184 | -0.0425 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |
| 5.0 | `FOODS_3_498_WI_2` | WI_2 | FOODS | 44.97 | 17.5879 | 17.5346 | -0.0532 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |
| 6.0 | `FOODS_2_326_WI_2` | WI_2 | FOODS | 42.46 | 16.9903 | 16.8519 | -0.1384 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |
| 7.0 | `FOODS_3_318_CA_1` | CA_1 | FOODS | 37.36 | 1.5447 | 1.6457 | +0.1010 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 8.0 | `FOODS_2_276_CA_3` | CA_3 | FOODS | 33.37 | 3.0683 | 3.1084 | +0.0401 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 9.0 | `FOODS_3_811_WI_3` | WI_3 | FOODS | 31.76 | 5.8120 | 5.8700 | +0.0580 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 10.0 | `FOODS_3_498_WI_3` | WI_3 | FOODS | 30.11 | 10.5812 | 10.7783 | +0.1970 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 11.0 | `FOODS_3_811_TX_1` | TX_1 | FOODS | 28.55 | 6.5132 | 6.6475 | +0.1343 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 12.0 | `FOODS_3_811_TX_2` | TX_2 | FOODS | 27.35 | 7.1558 | 7.1304 | -0.0254 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |
| 13.0 | `FOODS_3_318_WI_1` | WI_1 | FOODS | 26.62 | 1.3950 | 1.3971 | +0.0021 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 14.0 | `HOUSEHOLD_1_173_CA_3` | CA_3 | HOUSEHOLD | 26.56 | 4.9372 | 4.9974 | +0.0602 | *WORSENED* | `HUMAN_REVIEW_TRIAGE` |
| 15.0 | `FOODS_1_113_WI_2` | WI_2 | FOODS | 25.97 | 4.1155 | 4.0577 | -0.0578 | **IMPROVED** | `HUMAN_REVIEW_TRIAGE` |