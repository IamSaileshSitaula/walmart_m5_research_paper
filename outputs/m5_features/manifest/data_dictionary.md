# M5 Feature Pipeline Data Dictionary

This data dictionary defines all feature columns across Feature Matrices A, B, and C.

## Design Principles
- **Identical Index & Splits**: Matrices A, B, and C share identical row order, series, origins, and horizons.
- **No Leakage**: All sales- and price-derived history features are dated strictly on or before forecast origin O.
- **Explicit Known-Calendar Assumption**: Target-date weekday, month, SNAP, and calendar events are assumed known in advance.
- **Ablation Definitions**:
  - **Matrix A**: Baseline without any calendar event variables.
  - **Matrix B**: Same-day event indicators (`event_name_1/2`, `event_type_1/2`, `event_count`, `no_event`).
  - **Matrix C**: Named and pooled event-relative phase features (`k_*`, `ph_*`, multi-day spans).

## Feature Column Registry

| Column Name | Group | In A | In B | In C | Latest Info Date | Description |
|---|---|---|---|---|---|---|
| `id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: id |
| `item_id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: item_id |
| `dept_id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: dept_id |
| `cat_id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: cat_id |
| `store_id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: store_id |
| `state_id` | 1. Static hierarchy | True | True | True | Static / Time-invariant | Series hierarchy identifier: state_id |
| `origin_id` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: origin_id |
| `origin_d` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: origin_d |
| `origin_date` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: origin_date |
| `h` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: h |
| `target_d` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: target_d |
| `target_date` | 0. Index & Horizon Key | True | True | True | Forecast origin O / Target setup | Forecasting key coordinate: target_date |
| `wday` | 2. Known target-date calendar | True | True | True | Target date t (known calendar schedule) | Calendar attribute of target date t: wday |
| `month` | 2. Known target-date calendar | True | True | True | Target date t (known calendar schedule) | Calendar attribute of target date t: month |
| `year` | 2. Known target-date calendar | True | True | True | Target date t (known calendar schedule) | Calendar attribute of target date t: year |
| `snap_state` | 2. Known target-date calendar | True | True | True | Target date t (known calendar schedule) | Calendar attribute of target date t: snap_state |
| `target_units_sold` | 0. Ground Truth Label | True | True | True | Target date t (evaluation/training label only) | Observed unit sales at target date t = O + h |
| `sales_last_O` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_last_O |
| `sales_last_O_7` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_last_O_7 |
| `sales_last_O_28` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_last_O_28 |
| `sales_mean_7d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_mean_7d |
| `sales_std_7d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_std_7d |
| `sales_zero_ratio_7d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_zero_ratio_7d |
| `sales_mean_28d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_mean_28d |
| `sales_std_28d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_std_28d |
| `sales_zero_ratio_28d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_zero_ratio_28d |
| `sales_mean_56d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_mean_56d |
| `sales_std_56d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_std_56d |
| `sales_zero_ratio_56d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: sales_zero_ratio_56d |
| `store_sales_mean_7d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: store_sales_mean_7d |
| `store_sales_mean_28d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: store_sales_mean_28d |
| `cat_sales_mean_7d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: cat_sales_mean_7d |
| `cat_sales_mean_28d` | 4. Origin-anchored sales history | True | True | True | Forecast origin O (end of day O, strictly <= O) | Trailing sales statistic ending at origin O: cat_sales_mean_28d |
| `last_price_known_at_origin` | 5. Origin-anchored price history | True | True | True | Forecast origin O (completed weeks ending <= O) | Price history statistic as known at origin O: last_price_known_at_origin |
| `last_price_age_days` | 5. Origin-anchored price history | True | True | True | Forecast origin O (completed weeks ending <= O) | Price history statistic as known at origin O: last_price_age_days |
| `price_change_28d` | 5. Origin-anchored price history | True | True | True | Forecast origin O (completed weeks ending <= O) | Price history statistic as known at origin O: price_change_28d |
| `price_ratio_to_cat_median` | 5. Origin-anchored price history | True | True | True | Forecast origin O (completed weeks ending <= O) | Price history statistic as known at origin O: price_ratio_to_cat_median |
| `t_event_count` | 3. Target-date event (same-day) | False | True | True | Target date t (known calendar schedule) | Target date event indicator/name: t_event_count |
| `t_no_event` | 3. Target-date event (same-day) | False | True | True | Target date t (known calendar schedule) | Target date event indicator/name: t_no_event |
| `k_chanukah_end` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_chanukah_end |
| `ph_chanukah_end` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_chanukah_end |
| `k_christmas` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_christmas |
| `ph_christmas` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_christmas |
| `k_cinco_de_mayo` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_cinco_de_mayo |
| `ph_cinco_de_mayo` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_cinco_de_mayo |
| `k_easter` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_easter |
| `ph_easter` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_easter |
| `k_eidaladha` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_eidaladha |
| `ph_eidaladha` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_eidaladha |
| `k_father_s_day` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_father_s_day |
| `ph_father_s_day` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_father_s_day |
| `k_halloween` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_halloween |
| `ph_halloween` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_halloween |
| `k_independenceday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_independenceday |
| `ph_independenceday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_independenceday |
| `k_laborday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_laborday |
| `ph_laborday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_laborday |
| `k_lentstart` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_lentstart |
| `ph_lentstart` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_lentstart |
| `k_lentweek2` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_lentweek2 |
| `ph_lentweek2` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_lentweek2 |
| `k_martinlutherkingday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_martinlutherkingday |
| `ph_martinlutherkingday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_martinlutherkingday |
| `k_memorialday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_memorialday |
| `ph_memorialday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_memorialday |
| `k_mother_s_day` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_mother_s_day |
| `ph_mother_s_day` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_mother_s_day |
| `k_nbafinalsend` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_nbafinalsend |
| `ph_nbafinalsend` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_nbafinalsend |
| `k_nbafinalsstart` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_nbafinalsstart |
| `ph_nbafinalsstart` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_nbafinalsstart |
| `k_newyear` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_newyear |
| `ph_newyear` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_newyear |
| `k_orthodoxchristmas` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_orthodoxchristmas |
| `ph_orthodoxchristmas` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_orthodoxchristmas |
| `k_orthodoxeaster` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_orthodoxeaster |
| `ph_orthodoxeaster` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_orthodoxeaster |
| `k_presidentsday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_presidentsday |
| `ph_presidentsday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_presidentsday |
| `k_purim_end` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_purim_end |
| `ph_purim_end` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_purim_end |
| `k_ramadan_starts` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_ramadan_starts |
| `ph_ramadan_starts` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_ramadan_starts |
| `k_stpatricksday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_stpatricksday |
| `ph_stpatricksday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_stpatricksday |
| `k_superbowl` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_superbowl |
| `ph_superbowl` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_superbowl |
| `k_thanksgiving` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_thanksgiving |
| `ph_thanksgiving` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_thanksgiving |
| `k_valentinesday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_valentinesday |
| `ph_valentinesday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_valentinesday |
| `k_veteransday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_veteransday |
| `ph_veteransday` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_veteransday |
| `k_fam_cultural` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_fam_cultural |
| `ph_fam_cultural` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_fam_cultural |
| `n_fam_cultural_in_window` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: n_fam_cultural_in_window |
| `k_fam_national` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_fam_national |
| `ph_fam_national` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_fam_national |
| `n_fam_national_in_window` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: n_fam_national_in_window |
| `k_fam_religious` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_fam_religious |
| `ph_fam_religious` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_fam_religious |
| `n_fam_religious_in_window` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: n_fam_religious_in_window |
| `k_fam_sporting` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: k_fam_sporting |
| `ph_fam_sporting` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: ph_fam_sporting |
| `n_fam_sporting_in_window` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: n_fam_sporting_in_window |
| `span_nba_finals` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: span_nba_finals |
| `span_ramadan` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: span_ramadan |
| `span_lent` | 3. Event timing & relative phase | False | False | True | Target date t (known calendar schedule) | Event timing relative day or phase hypothesis: span_lent |
| `t_event_name_1` | 3. Target-date event (same-day) | False | True | False | Target date t (known calendar schedule) | Target date event indicator/name: t_event_name_1 |
| `t_event_type_1` | 3. Target-date event (same-day) | False | True | False | Target date t (known calendar schedule) | Target date event indicator/name: t_event_type_1 |
| `t_event_name_2` | 3. Target-date event (same-day) | False | True | False | Target date t (known calendar schedule) | Target date event indicator/name: t_event_name_2 |
| `t_event_type_2` | 3. Target-date event (same-day) | False | True | False | Target date t (known calendar schedule) | Target date event indicator/name: t_event_type_2 |

## Missingness & Encoding Semantics
- **Numerical NaN vs Genuine Zero**:
  - Numerical features (such as `last_price_age_days`, `price_change_28d`, `k_*`) preserve `NaN` when unobserved or outside the active window.
  - Genuine zero sales (`sales_last_O == 0`) are encoded as `0.0`, distinct from `NaN`.
- **Categorical Variables**:
  - High-cardinality non-ordinal features (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`) use native dictionary/categorical encoding.
  - Missing event names are explicitly represented by the string category `'none'`.
  - Event phases outside the hypothesis window `[-14, +7]` are explicitly represented by `'outside'`.