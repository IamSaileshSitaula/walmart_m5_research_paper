"""Chronological 28-day holiday backtest fold manifest (no random splits)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import _event_occurrences, slug


def build_fold_manifest(cal: pd.DataFrame, last_sales_d: int, cfg: Config) -> pd.DataFrame:
    """Expanding-window folds. For each occurrence of a target holiday H the
    origin is T = H - holiday_position_in_window, so H falls at horizon
    h = holiday_position_in_window and the 28-day window contains the
    pre-holiday run-up and the post-holiday days. Folds whose forecast window
    exceeds observed sales, or with < min_train_days history, are kept in the
    manifest with included=False and a reason (transparent, not silently dropped)."""
    occ = _event_occurrences(cal)
    date_of = dict(zip(cal["d_num"].astype(int), cal["date"]))
    H = cfg.horizon
    pos = cfg.holiday_position_in_window
    rows = []

    def window_events(fs: int, fe: int) -> str:
        w = occ[(occ["event_d_num"] >= fs) & (occ["event_d_num"] <= fe)]
        return "; ".join(f"{r.event_name}(h={int(r.event_d_num) - fs + 1})" for r in w.itertuples())

    def make(fold_type, target, ev_d, T):
        fs, fe = T + 1, T + H
        reasons = []
        if T < cfg.min_train_days:
            reasons.append(f"history {T}d < min_train_days {cfg.min_train_days}")
        if fe > last_sales_d:
            reasons.append(f"forecast_end d_{fe} beyond last observed sales d_{last_sales_d}")
        return {
            "fold_type": fold_type, "target_event": target,
            "event_date": date_of.get(ev_d) if ev_d else pd.NaT,
            "event_d": f"d_{ev_d}" if ev_d else None,
            "event_horizon": (ev_d - T) if ev_d else None,
            "train_start_d": "d_1", "train_start_date": date_of[1],
            "origin_d": f"d_{T}", "origin_date": date_of.get(T),
            "train_end_d": f"d_{T}", "train_end_date": date_of.get(T),
            "n_train_days": T,
            "early_stop_valid_start_d": f"d_{T - H + 1}", "early_stop_valid_end_d": f"d_{T}",
            "forecast_start_d": f"d_{fs}", "forecast_start_date": date_of.get(fs),
            "forecast_end_d": f"d_{fe}", "forecast_end_date": date_of.get(fe),
            "horizon_days": H,
            "events_in_forecast_window": window_events(fs, fe) if fe <= len(cal) else "",
            "fit_rule": f"All imputation/scaling/encoding/anomaly thresholds/early stopping fit on d <= {T} only",
            "included": not reasons, "exclusion_reason": "; ".join(reasons),
            "_T": T,
        }

    for name in cfg.fold_holidays:
        for ev_d in occ.loc[occ["event_name"] == name, "event_d_num"].astype(int):
            rows.append(make("holiday", name, ev_d, ev_d - pos))
    rows.append(make("final_holdout_m5_evaluation", "M5 public evaluation window", None, last_sales_d - H))

    m = pd.DataFrame(rows).sort_values(["_T", "target_event"]).reset_index(drop=True)
    m.insert(0, "fold_id", [
        f"F{i:02d}_{slug(r.target_event)[:20]}_{pd.Timestamp(r.event_date).year if pd.notna(r.event_date) else 'final'}"
        for i, r in enumerate(m.itertuples(), start=1)])
    return m


def check_fold_manifest(m: pd.DataFrame, last_sales_d: int) -> dict:
    inc = m[m["included"]]
    T = inc["_T"].to_numpy()
    fs = inc["forecast_start_d"].str.slice(2).astype(int).to_numpy()
    fe = inc["forecast_end_d"].str.slice(2).astype(int).to_numpy()
    es = inc["early_stop_valid_start_d"].str.slice(2).astype(int).to_numpy()
    return {
        "n_folds_total": int(len(m)), "n_folds_included": int(len(inc)),
        "train_end_before_forecast_start_failures": int((T >= fs).sum()),
        "forecast_window_length_failures": int(((fe - fs + 1) != 28).sum()),
        "forecast_beyond_observed_failures": int((fe > last_sales_d).sum()),
        "early_stop_outside_train_failures": int(((es < 1) | (es > T)).sum()),
        "random_split_used": False,
    }
