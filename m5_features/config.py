"""Configuration for the origin-based feature pipeline (all choices prespecified).

Every number here was fixed BEFORE looking at final-test data. The only
data-driven choice (which candidate events get their own named columns vs.
pooling by family) is made by `events.select_events` on development dates only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

FEATURE_PIPELINE_VERSION = "1.0.0"

# Expected concept -> actual cleaned-panel column (verified at run time).
FIELD_MAP = {
    "item identifier": "item_id",
    "store identifier": "store_id",
    "state identifier": "state_id",
    "category identifier": "cat_id",
    "department identifier": "dept_id",
    "date": "date",
    "day index (M5 d)": "d / d_num",
    "observed unit sales": "units_sold",          # == units_sold_raw unless a logged correction exists
    "event slot 1 name/type": "event_name_1 / event_type_1",
    "event slot 2 name/type": "event_name_2 / event_type_2",
    "state SNAP indicator": "snap_state",          # state-matched per row
    "weekly price": "sell_price_raw",              # raw weekly listed price, null = not listed
    "price listing status": "price_status",
}
REQUIRED_PANEL_COLUMNS = ["item_id", "store_id", "state_id", "cat_id", "dept_id", "date", "d", "d_num",
                          "units_sold", "units_sold_raw", "units_corrected_flag", "event_name_1",
                          "event_type_1", "event_name_2", "event_type_2", "snap_state",
                          "sell_price_raw", "price_status"]

# Prespecified event families = M5 event_type (no subjective re-grouping).
# Multi-day events appear only as start/end markers; spans are formed ONLY
# where both markers are listed in the calendar (never invented).
SPAN_PAIRS = (  # (span name, start marker, end marker, end_inclusive)
    ("nba_finals", "NBAFinalsStart", "NBAFinalsEnd", True),
    ("ramadan", "Ramadan starts", "Eid al-Fitr", False),   # Eid al-Fitr = day after Ramadan
    ("lent", "LentStart", "Easter", False),
)


@dataclass
class FeatureConfig:
    panel_path: Path
    calendar_features_path: Path
    cleaning_manifest_path: Path
    out_dir: Path

    horizon: int = 28
    # --- origin sampling rule ---
    base_step: int = 28                 # non-overlapping 28-day origins
    base_anchor_offset: int = 28        # last base origin = last observed day - 28 (M5 public window)
    min_origin_d: int = 365             # >= 1 year of history before any origin
    pre_event_holidays: tuple = ("Thanksgiving", "Christmas")
    pre_event_leads: tuple = (7, 14, 21, 28)   # event lands at horizon h = lead
    # --- splits ---
    dev_seasons: tuple = (2012, 2013, 2014)
    test_season: int = 2015
    # --- eligibility ---
    min_active_days: int = 28           # first positive sale must be <= O - 27
    # --- event timing (hypotheses, not known spike durations) ---
    k_min: int = -14
    k_max: int = 7
    phase_windows: tuple = (("pre_14_8", -14, -8), ("pre_7_2", -7, -2), ("pre_1", -1, -1),
                            ("day_0", 0, 0), ("post_1_3", 1, 3), ("post_4_7", 4, 7))
    forced_named_events: tuple = ("Thanksgiving", "Christmas")   # thesis claims, prespecified
    min_dev_occurrences: int = 3
    min_abs_effect: float = 0.05        # |median log-ratio| in any phase, dev data only
    baseline_weeks: tuple = (4, 5, 6, 7)
    # --- optional calendar indices (off: they encode fixed-date holidays in matrix A) ---
    include_seasonal_index: bool = False
    # --- audit ---
    perturbation_origins: int = 8
    worked_example_series: tuple = (("FOODS_3_090", "CA_3"), ("HOUSEHOLD_1_118", "TX_2"))
    seed: int = 20261019

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
                for k, v in d.items()}
