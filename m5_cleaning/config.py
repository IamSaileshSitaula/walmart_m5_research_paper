"""Central configuration for the M5 cleaning pipeline.

Every threshold used anywhere in the pipeline lives here so that the
decision log, the report, and the code cannot silently drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

PIPELINE_VERSION = "1.0.0"

# ----------------------------------------------------------------------------
# Expected raw schemas (Kaggle M5 Forecasting - Accuracy release)
# ----------------------------------------------------------------------------
EXPECTED_CALENDAR_COLUMNS = [
    "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]
EXPECTED_PRICE_COLUMNS = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Preference order: evaluation (d_1..d_1941) first, validation (d_1..d_1913) second.
# The two files overlap and must NEVER be appended.
SALES_FILES = [
    ("evaluation", "sales_train_evaluation.csv"),
    ("validation", "sales_train_validation.csv"),
]
CALENDAR_FILE = "calendar.csv"
PRICES_FILE = "sell_prices.csv"

KNOWN_EVENT_TYPES = {"Sporting", "Cultural", "National", "Religious"}
WEEKDAY_TO_WDAY = {  # M5 convention: Walmart week starts Saturday
    "Saturday": 1, "Sunday": 2, "Monday": 3, "Tuesday": 4,
    "Wednesday": 5, "Thursday": 6, "Friday": 7,
}


@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    corrections_file: Path | None = None

    # --- Anomaly REVIEW flags (descriptive audit only; never alter values) ---
    anomaly_min_units: int = 10            # ignore tiny counts
    anomaly_series_quantile: float = 0.995 # vs. the series' own active history
    anomaly_local_ratio: float = 5.0       # vs. mean of +/- halfwidth days
    anomaly_local_halfwidth: int = 14
    isolated_spike_ratio: float = 3.0      # max neighbour < units / ratio
    store_day_low_ratio: float = 0.10      # store total < 10% of local median
    context_event_window: int = 7
    price_reduction_ratio: float = 0.90    # price < 90% of series median

    # --- Calendar / event features (feature engineering stage) ---
    event_window_main: int = 7
    event_window_sensitivity: int = 28
    focus_holidays: tuple = (
        "SuperBowl", "ValentinesDay", "Easter", "Mother's day", "MemorialDay",
        "Father's day", "IndependenceDay", "LaborDay", "Halloween",
        "Thanksgiving", "Christmas", "NewYear",
    )

    # --- Chronological backtest folds ---
    horizon: int = 28
    holiday_position_in_window: int = 21   # holiday lands at horizon h=21
    min_train_days: int = 730              # >= 2 years history before origin
    fold_holidays: tuple = (
        "SuperBowl", "ValentinesDay", "Easter", "Mother's day", "MemorialDay",
        "IndependenceDay", "LaborDay", "Halloween", "Thanksgiving",
        "Christmas", "NewYear",
    )
    # Target-relative lags proposed for the model; each must be >= horizon.
    proposed_target_lags: tuple = (28, 35, 42, 49, 56, 364)
    illustrative_invalid_lags: tuple = (1, 7, 14)

    # --- Output ---
    row_group_size: int = 1_000_000
    max_cell_exceptions: int = 50_000  # cap on per-cell raw-value exception rows

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d
