# Worked Examples: Thanksgiving and Christmas Forecast Episodes

This document details specific item-store examples around Thanksgiving and Christmas across multiple forecast horizons and planning leads.

## Episode: Thanksgiving 2014 (d_1399, 2014-11-27)
**Demonstration Series**: `FOODS_3_090_CA_3` (Department: FOODS_3, Category: FOODS, State: CA)

| Planning Lead | Origin O | Origin Date | Horizon h | Target Day t | Target Date | Signed k | Event Phase | Latest Sales Date Used | Observed Units Sold | Trailing 7d Mean at O |
|---|---|---|---|---|---|---|---|---|---|---|
| Lead 7d | d_1392 | 2014-11-20 | h=7 | d_1399 | 2014-11-27 | +0 | `day_0` | 2014-11-20 | 257 | 100.29 |
| Lead 7d (pre) | d_1392 | 2014-11-20 | h=6 | d_1398 | 2014-11-26 | -1 | `pre_1` | 2014-11-20 | 287 | 100.29 |
| Lead 14d | d_1385 | 2014-11-13 | h=14 | d_1399 | 2014-11-27 | +0 | `day_0` | 2014-11-13 | 257 | 126.00 |
| Lead 14d (pre) | d_1385 | 2014-11-13 | h=13 | d_1398 | 2014-11-26 | -1 | `pre_1` | 2014-11-13 | 287 | 126.00 |
| Lead 21d | d_1378 | 2014-11-06 | h=21 | d_1399 | 2014-11-27 | +0 | `day_0` | 2014-11-06 | 257 | 135.43 |
| Lead 21d (pre) | d_1378 | 2014-11-06 | h=20 | d_1398 | 2014-11-26 | -1 | `pre_1` | 2014-11-06 | 287 | 135.43 |
| Lead 28d | d_1371 | 2014-10-30 | h=28 | d_1399 | 2014-11-27 | +0 | `day_0` | 2014-10-30 | 257 | 120.29 |
| Lead 28d (pre) | d_1371 | 2014-10-30 | h=27 | d_1398 | 2014-11-26 | -1 | `pre_1` | 2014-10-30 | 287 | 120.29 |

## Episode: Christmas 2014 (d_1427, 2014-12-25)
**Demonstration Series**: `FOODS_3_090_CA_3` (Department: FOODS_3, Category: FOODS, State: CA)

| Planning Lead | Origin O | Origin Date | Horizon h | Target Day t | Target Date | Signed k | Event Phase | Latest Sales Date Used | Observed Units Sold | Trailing 7d Mean at O |
|---|---|---|---|---|---|---|---|---|---|---|
| Lead 7d | d_1420 | 2014-12-18 | h=7 | d_1427 | 2014-12-25 | +0 | `day_0` | 2014-12-18 | 0 | 4.71 |
| Lead 7d (pre) | d_1420 | 2014-12-18 | h=6 | d_1426 | 2014-12-24 | -1 | `pre_1` | 2014-12-18 | 0 | 4.71 |
| Lead 14d | d_1413 | 2014-12-11 | h=14 | d_1427 | 2014-12-25 | +0 | `day_0` | 2014-12-11 | 0 | 0.86 |
| Lead 14d (pre) | d_1413 | 2014-12-11 | h=13 | d_1426 | 2014-12-24 | -1 | `pre_1` | 2014-12-11 | 0 | 0.86 |
| Lead 21d | d_1406 | 2014-12-04 | h=21 | d_1427 | 2014-12-25 | +0 | `day_0` | 2014-12-04 | 0 | 57.86 |
| Lead 21d (pre) | d_1406 | 2014-12-04 | h=20 | d_1426 | 2014-12-24 | -1 | `pre_1` | 2014-12-04 | 0 | 57.86 |
| Lead 28d | d_1399 | 2014-11-27 | h=28 | d_1427 | 2014-12-25 | +0 | `day_0` | 2014-11-27 | 0 | 167.71 |
| Lead 28d (pre) | d_1399 | 2014-11-27 | h=27 | d_1426 | 2014-12-24 | -1 | `pre_1` | 2014-11-27 | 0 | 167.71 |

## Holiday Boundary Crossing: Christmas into January
For Christmas origins with planning leads (e.g. Origin d_1420 = 2014-12-18, Lead 7):
- Target horizons h = 1..7 cover pre-Christmas and Christmas Day (2014-12-19 to 2014-12-25).
- Target horizons h = 8..14 cover the post-Christmas and New Year's Eve period (2014-12-26 to 2014-12-31).
- Target horizons h = 15..28 cross the calendar-year boundary into **January 2015** (2015-01-01 to 2015-01-14).
- In Matrix C, target date 2015-01-01 (h=14) simultaneously activates:
  - `k_christmas = +7` (Phase: `post_4_7`)
  - `k_new_year = 0` (Phase: `day_0`)
- This confirms simultaneous multi-event representation across calendar-year transitions.