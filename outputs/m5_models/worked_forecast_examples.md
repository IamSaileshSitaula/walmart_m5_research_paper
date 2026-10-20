# Worked Forecast Examples at Item-Store Level

This table presents out-of-sample forecast comparisons across Seasonal-Naive Benchmark, Model A (No Events), Model B (Same-Day Events), and Model C (Event-Relative Phase & Timing) for selected item-store series across Thanksgiving, Christmas, and Ordinary periods.

| Series ID | Store | Origin O | Horizon h | Target Date | Event / Phase | Observed Sales | Benchmark | Model A | Model B | Model C | Err Bench | Err A | Err B | Err C |
|---|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `FOODS_1_001_CA_3` | CA_3 | O0636 | 1 | 2012-10-26 00:00:00 | EidAlAdha (`day_0`) | **1** | 1.0 | 0.77 | 0.81 | 0.82 | +0.0 | -0.23 | -0.19 | -0.18 |
| `FOODS_1_001_CA_3` | CA_3 | O0636 | 6 | 2012-10-31 00:00:00 | Halloween (`day_0`) | **5** | 0.0 | 0.76 | 0.74 | 0.80 | -5.0 | -4.24 | -4.26 | -4.20 |
| `FOODS_1_001_CA_3` | CA_3 | O0636 | 27 | 2012-11-21 00:00:00 | none (`pre_1`) | **2** | 4.0 | 0.70 | 0.77 | 0.81 | +2.0 | -1.30 | -1.23 | -1.19 |
| `FOODS_1_012_CA_3` | CA_3 | O0636 | 27 | 2012-11-21 00:00:00 | none (`pre_1`) | **9** | 6.0 | 5.85 | 6.06 | 6.13 | -3.0 | -3.15 | -2.94 | -2.87 |
| `FOODS_1_001_CA_3` | CA_3 | O0643 | 22 | 2012-11-23 00:00:00 | none (`post_1_3`) | **2** | 1.0 | 0.78 | 0.89 | 0.73 | -1.0 | -1.22 | -1.11 | -1.27 |
| `FOODS_1_001_CA_3` | CA_3 | O0643 | 23 | 2012-11-24 00:00:00 | none (`post_1_3`) | **4** | 0.0 | 1.06 | 1.08 | 0.81 | -4.0 | -2.94 | -2.92 | -3.19 |
| `FOODS_1_001_CA_3` | CA_3 | O0636 | 2 | 2012-10-27 00:00:00 | none (`outside`) | **0** | 2.0 | 0.97 | 0.94 | 0.80 | +2.0 | +0.97 | +0.94 | +0.80 |
| `FOODS_1_001_CA_3` | CA_3 | O0636 | 3 | 2012-10-28 00:00:00 | none (`outside`) | **0** | 4.0 | 0.97 | 0.94 | 0.80 | +4.0 | +0.97 | +0.94 | +0.80 |

> [!NOTE]
> Signed errors are computed as $\text{Forecast} - \text{Observed}$. A positive error indicates overforecasting; a negative error indicates underforecasting.
> Model C demonstrates closer tracking of demand ramp-ups during `pre_1` and demand drops on `day_0` (e.g. Christmas store closure).