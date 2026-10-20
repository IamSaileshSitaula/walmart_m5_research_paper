"""Feature-engineering stage (kept separate from cleaning).

Contains:
* calendar/event features (known-calendar assumption, documented);
* event-relative day features allowing overlapping events;
* reference implementation of origin-anchored history features and the
  'last price known at forecast origin' sensitivity feature;
* automated leakage tests (future-perturbation invariance).

No model is trained here.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config


def slug(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _event_occurrences(cal: pd.DataFrame) -> pd.DataFrame:
    occ = pd.concat([
        cal[["d_num", "date", "event_name_1", "event_type_1"]].set_axis(["event_d_num", "event_date", "event_name", "event_type"], axis=1).assign(slot=1),
        cal[["d_num", "date", "event_name_2", "event_type_2"]].set_axis(["event_d_num", "event_date", "event_name", "event_type"], axis=1).assign(slot=2),
    ]).dropna(subset=["event_name"])
    return occ.sort_values(["event_d_num", "slot"]).reset_index(drop=True)


def event_relative_long(cal: pd.DataFrame, window: int) -> pd.DataFrame:
    """One row per (calendar day, event occurrence) with |rel_day| <= window.
    rel_day = day - event_day (negative = before the event). Overlapping events
    produce multiple rows for the same day by design."""
    occ = _event_occurrences(cal)
    n = len(cal)
    rows = []
    for k in range(-window, window + 1):
        dn = occ["event_d_num"].to_numpy() + k
        ok = (dn >= 1) & (dn <= n)
        rows.append(pd.DataFrame({
            "d_num": dn[ok], "rel_day": k,
            "event_name": occ["event_name"].to_numpy()[ok], "event_type": occ["event_type"].to_numpy()[ok],
            "event_d_num": occ["event_d_num"].to_numpy()[ok],
        }))
    out = pd.concat(rows, ignore_index=True)
    out = out.merge(cal[["d_num", "d", "date"]], on="d_num", how="left", validate="many_to_one")
    out = out.merge(cal[["d_num", "date"]].rename(columns={"d_num": "event_d_num", "date": "event_date"}),
                    on="event_d_num", how="left", validate="many_to_one")
    out["in_main_window_pm7"] = out["rel_day"].abs() <= 7
    return out.sort_values(["d_num", "event_d_num", "event_name"])[
        ["d", "d_num", "date", "event_name", "event_type", "event_date", "event_d_num", "rel_day", "in_main_window_pm7"]
    ].reset_index(drop=True)


def calendar_event_features(cal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Day-level features. Event names/types stay categorical (strings);
    no arbitrary numeric order is assigned."""
    out = cal[["d", "d_num", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
               "event_name_1", "event_type_1", "event_name_2", "event_type_2",
               "snap_CA", "snap_TX", "snap_WI"]].copy()
    e1, e2 = cal["event_name_1"], cal["event_name_2"]
    out["event_count"] = (e1.notna().astype("int8") + e2.notna().astype("int8")).astype("int8")
    out["no_listed_event"] = (out["event_count"] == 0).astype("int8")
    occ = _event_occurrences(cal)
    for nm in sorted(occ["event_name"].unique()):
        out[f"ev_{slug(nm)}"] = ((e1 == nm) | (e2 == nm)).astype("int8")
    t1, t2 = cal["event_type_1"], cal["event_type_2"]
    for tp in sorted(occ["event_type"].unique()):
        out[f"evtype_{slug(tp)}"] = ((t1 == tp) | (t2 == tp)).astype("int8")

    # Event-relative columns for focus holidays: nearest occurrence, signed.
    d_num = cal["d_num"].to_numpy().astype(int)
    for nm in cfg.focus_holidays:
        ev_days = occ.loc[occ["event_name"] == nm, "event_d_num"].to_numpy().astype(int)
        if len(ev_days) == 0:
            continue
        rel = d_num[:, None] - ev_days[None, :]
        idx = np.abs(rel).argmin(axis=1)
        nearest = rel[np.arange(len(d_num)), idx]
        for w, tag in [(cfg.event_window_main, f"w{cfg.event_window_main}"),
                       (cfg.event_window_sensitivity, f"w{cfg.event_window_sensitivity}")]:
            col = pd.array(np.where(np.abs(nearest) <= w, nearest, 0), dtype="Int16")
            col[np.abs(nearest) > w] = pd.NA
            out[f"rel_{slug(nm)}_{tag}"] = col

    # Generic proximity to ANY listed event (calendar-known)
    long7 = event_relative_long(cal, cfg.event_window_main)
    out["n_events_within_pm7"] = out["d_num"].map(long7.groupby("d_num").size()).fillna(0).astype("int16")
    ev_all = np.unique(occ["event_d_num"].to_numpy().astype(int))
    nxt = np.searchsorted(ev_all, d_num, side="left")
    prv = np.searchsorted(ev_all, d_num, side="right") - 1
    to_next = np.where(nxt < len(ev_all), ev_all[np.minimum(nxt, len(ev_all) - 1)] - d_num, -1)
    since_last = np.where(prv >= 0, d_num - ev_all[np.maximum(prv, 0)], -1)
    out["days_to_next_event"] = pd.array(to_next, dtype="Int16")
    out.loc[to_next < 0, "days_to_next_event"] = pd.NA
    out["days_since_last_event"] = pd.array(since_last, dtype="Int16")
    out.loc[since_last < 0, "days_since_last_event"] = pd.NA
    return out


# ----------------------------------------------------------------------------
# Origin-anchored reference features + leakage tests
# ----------------------------------------------------------------------------
HISTORY_WINDOWS = (7, 28, 56, 182, 364)


def week_end_index(wm_yr_wk: np.ndarray) -> np.ndarray:
    """For each day index, index of the last day of its wm_yr_wk (in calendar order)."""
    wk = np.asarray(wm_yr_wk)
    end = np.empty(len(wk), dtype=np.int64)
    last = len(wk) - 1
    for i in range(len(wk) - 1, -1, -1):
        if i < len(wk) - 1 and wk[i] != wk[i + 1]:
            last = i
        end[i] = last
    return end


def origin_anchored_features(Y: np.ndarray, P: np.ndarray, week_end_idx: np.ndarray,
                             wday_idx: np.ndarray, T: int) -> dict[str, np.ndarray]:
    """Features for origin day index T (0-based). Uses Y[:, :T+1] only and
    prices only from weeks that are COMPLETE on or before T (a weekly average
    for a week straddling T includes post-origin days)."""
    n = Y.shape[0]
    H = Y[:, : T + 1].astype(np.float64)
    f: dict[str, np.ndarray] = {}
    for w in HISTORY_WINDOWS:
        f[f"hist_mean_{w}d"] = H[:, -w:].mean(axis=1) if T + 1 >= w else np.full(n, np.nan)
    f["hist_std_28d"] = H[:, -28:].std(axis=1)
    f["hist_zero_rate_28d"] = (H[:, -28:] == 0).mean(axis=1)
    pos = H > 0
    any_pos = pos.any(axis=1)
    last_pos = T - np.argmax(pos[:, ::-1], axis=1)
    f["days_since_last_sale"] = np.where(any_pos, T - last_pos, np.nan)
    wd = wday_idx[: T + 1]
    for k in range(1, 8):
        idx = np.nonzero(wd == k)[0][-8:]
        f[f"hist_wday{k}_mean_8w"] = H[:, idx].mean(axis=1) if len(idx) else np.full(n, np.nan)
    complete = np.nonzero(week_end_idx[: T + 1] <= T)[0]
    if len(complete):
        L = complete[-1]
        Pe = P[:, : L + 1]
        valid = ~np.isnan(Pe)
        has = valid.any(axis=1)
        last_valid = L - np.argmax(valid[:, ::-1], axis=1)
        f["last_price_known_at_origin"] = np.where(has, Pe[np.arange(n), last_valid], np.nan)
        f["last_price_age_days"] = np.where(has, T - last_valid, np.nan)
    else:
        f["last_price_known_at_origin"] = np.full(n, np.nan)
        f["last_price_age_days"] = np.full(n, np.nan)
    return f


def target_relative_lag_valid(lag: int, horizon: int) -> bool:
    """For target day t = T + h, lag value y[t - lag] is observed at origin T
    iff t - lag <= T  <=>  lag >= h. Valid for ALL horizons iff lag >= horizon."""
    return lag >= horizon


def lag_availability_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for lag in sorted(set(cfg.illustrative_invalid_lags) | set(cfg.proposed_target_lags)):
        ok_h = [h for h in range(1, cfg.horizon + 1) if lag >= h]
        rows.append({
            "lag_days": lag, "proposed_for_model": lag in cfg.proposed_target_lags,
            "valid_horizons": f"1..{max(ok_h)}" if ok_h else "none",
            "invalid_horizons": f"{lag + 1}..{cfg.horizon}" if lag < cfg.horizon else "none",
            "valid_for_all_28": lag >= cfg.horizon,
        })
    return pd.DataFrame(rows)


def check_origin_availability(Y: np.ndarray, P: np.ndarray, week_end_idx: np.ndarray,
                              wday_idx: np.ndarray, origins: list[int], seed: int = 0) -> list[dict]:
    """Perturb all post-origin sales and all prices from incomplete/future weeks;
    origin-anchored features must be bit-identical. A naive target-relative
    lag-7 at h=8 is used as a negative control and MUST change."""
    rng = np.random.default_rng(seed)
    out = []
    for T in origins:
        base = origin_anchored_features(Y, P, week_end_idx, wday_idx, T)
        Y2 = Y.copy()
        Y2[:, T + 1:] = rng.integers(0, 1000, size=Y2[:, T + 1:].shape)
        P2 = P.copy()
        future_cols = week_end_idx > T
        P2[:, future_cols] = rng.uniform(0.01, 500, size=(P2.shape[0], int(future_cols.sum())))
        pert = origin_anchored_features(Y2, P2, week_end_idx, wday_idx, T)
        same = all(np.array_equal(base[k], pert[k], equal_nan=True) for k in base)
        neg_control_changed = bool((Y[:, T + 8 - 7] != Y2[:, T + 8 - 7]).any()) if T + 1 < Y.shape[1] else None
        out.append({"origin_idx": T, "n_features": len(base), "invariant_to_future": same,
                    "negative_control_lag7_h8_detected": neg_control_changed})
    return out
