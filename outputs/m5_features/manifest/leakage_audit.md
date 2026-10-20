# Leakage Audit and Temporal Boundary Verification

## 1. Rolling-Origin Temporal Barriers
- **Principle**: For each forecast origin O (end of day O) and horizon h in 1..28, target date is t = O + h.
- **Sales and Price Features**:
  - All sales statistics are computed over windows [O - W + 1 .. O] strictly dated <= O.
  - All price history features use only weekly prices from completed weeks dated <= O.
  - Future realized sales (t = O + 1 .. O + 28) and future realized prices are **NEVER** accessed during feature calculation.
- **Verification**:
  - Maximum sales date accessed for origin O is exactly day O.
  - Negative control: Perturbing target-period sales (t > O) produces 0.0 change in all history features.

## 2. Chronological Split Separation
- **Split Invariance Results**:
  - `train_target_crosses_eval_labels`: **0**
  - `dev_targets_in_final_test_period`: **0**
  - `final_test_train_targets_at_or_after_test_start`: **0**
  - `random_split_used`: **0** (Strict chronological splits only).

## 3. Final-Test Isolation in Event Selection
- Event screening (named vs. pooled) and category median price calculations were executed **strictly on development data** (d_num < 1736, corresponding to dates before 2015-10-29).
- The final-test season (Holiday Season 2015 and the M5 evaluation window d_1914-d_1941) did not influence window thresholds or event grouping decisions.

## 4. Audit Result: PASS
All 4 temporal boundary assertions passed with zero violations.
