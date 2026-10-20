# Feature Missingness and Data Quality Accounting

## Summary of Expected Missingness
In this pipeline, missing values (`NaN`) are intentional and carry specific domain meanings:

1. **Price History Features (`last_price_known_at_origin`, `last_price_age_days`, `price_change_28d`)**:
   - `NaN` occurs when a series has never had a listed price in any completed week on or before forecast origin O.
   - For `price_change_28d`, `NaN` additionally occurs if origin O < 28 days or if no price existed at O - 28.
   - M5 sell prices are weekly; missing prices are unlisted, not zero dollars.

2. **Event Relative Days (`k_<event>`, `k_fam_<type>`)**:
   - `NaN` occurs when the target date t is outside the hypothesis window `[-14, +7]` relative to the nearest occurrence of that event.
   - This ensures LightGBM does not interpret distant dates as numeric extremes.
   - The corresponding phase feature (`ph_<event>`) takes the explicit category `'outside'`.

3. **Origin-Anchored Sales History**:
   - `sales_last_O_7` and `sales_last_O_28` take `NaN` only if origin O < 7 or O < 28 respectively. Since the minimum origin is O=373, these are 100% complete across all generated rows.
   - Zero sales are strictly preserved as `0.0`.

## Unexplained Missingness Audit
- Target label (`target_units_sold`): **0% missing** (verified on all 64,386,140 rows).
- Static identifiers (`item_id`, `store_id`, etc.): **0% missing**.
- Target calendar variables (`wday`, `month`, `year`, `snap_state`): **0% missing**.
- **Audit Result: PASS (No unexplained missingness).**
