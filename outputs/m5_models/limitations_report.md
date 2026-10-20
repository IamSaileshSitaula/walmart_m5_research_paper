# Research Limitations and Causal Disclaimers

## 1. Association vs. Causal Attribution
- The event-relative features (`k_*`, `ph_*`) capture **historical descriptive associations** between target-date calendar placement and observed point-of-sale retail units.
- LightGBM is a non-causal statistical regression model. Observed coefficient splits on `ph_thanksgiving` or `ph_christmas` reflect conditional correlations, not exogenous causal treatment effects.
- We do **NOT** claim that the calendar event caused the observed sales trajectory in isolation, because marketing promotions, local staffing, store opening hours, weather, and supplier delivery schedules covary with major holidays.

## 2. Limited Independent Holiday Episodes
- The M5 dataset spans 5.4 years (2011 to 2016), containing exactly **4 completed Christmas closures** and **4 Thanksgiving episodes** in historical training/validation data, plus 1 final test season (2015).
- Although the evaluation dataset contains hundreds of thousands of item–store forecast rows around Christmas, these rows represent **cross-sectional SKU realizations within the same few calendar episodes**, not thousands of statistically independent holiday occurrences.
- Consequently, findings from the final 2015 holiday test represent a **limited chronological validation**, not definitive proof of universal generalization across arbitrary future macroeconomic shifts.

## 3. Observed Sales vs Unconstrained Demand
- The target variable `target_units_sold` measures **recorded transactions**, which conflate true consumer demand with supply-side stock availability.
- A recorded zero sale indicates zero units were scanned at checkout. It does **not** prove zero customer demand (unrecorded stockouts) nor does it prove complete assortment stockout.
- Any downstream inventory planning must account for censoring and unobserved stockouts through dedicated simulation.