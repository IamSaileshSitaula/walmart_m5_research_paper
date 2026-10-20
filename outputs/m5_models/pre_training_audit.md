# Pre-Training Gates and Data Contract Audit Report

**Date**: 2026-10-20 11:45:47  
**Total Origins Audited**: 88 (Exhaustive across Matrix A, B, and C)  
**Total Rows per Matrix**: 64,386,140 rows  

## Audit Checklist

| Check ID | Scope | Contract Requirement | Checked | Failures | Status | Operational Notes |
|---|---|---|---|---|---|---|
| PRE_01 | Exhaustive (All 88 origins, 264 files) | Partition Readability & Availability | 264 | 0 | **PASS** | All 88 Hive partitions in Matrix A, B, and C are present and readable |
| PRE_02 | Exhaustive (All 64,386,140 rows) | Primary Key Uniqueness (item-store, origin, h) | 64,386,140 | 0 | **PASS** | Zero duplicate primary keys; exactly 28 rows per active series per origin |
| PRE_03 | Exhaustive (All 64,386,140 rows) | Horizon Range h in [1, 28] | 64,386,140 | 0 | **PASS** | All horizon values strictly fall within [1, 28] |
| PRE_04 | Exhaustive (All 64,386,140 rows) | Target Day Arithmetic: t == O + h | 64,386,140 | 0 | **PASS** | target_d == origin_d + h holds exactly for all rows |
| PRE_05 | Exhaustive (All 64,386,140 rows) | Nonnegative Observed Units Sold | 64,386,140 | 0 | **PASS** | target_units_sold >= 0 confirmed across all rows (no negative counts) |
| PRE_06 | Exhaustive (All 88 origins x 3 matrices) | Exact Coordinate Equality (A == B == C) | 88 | 0 | **PASS** | Identical series ordering, row counts (64,386,140), and coordinates across A, B, and C |
| PRE_07 | Exhaustive (All 5 folds) | Temporal Split Separation & Non-Contamination | 440 | 0 | **PASS** | All training labels strictly precede eval targets; final test starts exactly at d_1736 |
| PRE_08 | Exhaustive (112 Features in Manifest) | Predictor Temporal Availability (<= O vs Known Calendar t) | 112 | 0 | **PASS** | Programmatically verified: all sales and price features strictly bounded by <= O; calendar features strictly describe known target schedule |

## Data Contract Rules and Policies
1. **Label Isolation**: `target_units_sold` is strictly treated as the training label and is excluded from all predictor matrices.
2. **Excluded Administrative Keys**: `id` (provenance string), `origin_id`, `origin_d`, `origin_date`, `target_d`, and `target_date` are strictly excluded from predictors.
3. **Missingness vs Zero**: Numerical missing values (`NaN`) in price and distance features are preserved as separate distinct bins from genuine zero sales (`0.0`).
4. **Model-Update Policy**: The final model is frozen on pre-test labels ($d < 1736$). During rolling-origin test evaluation, each origin uses historical sales dated $\le O$ as updated feature inputs, but model weights are NEVER refit on test labels.