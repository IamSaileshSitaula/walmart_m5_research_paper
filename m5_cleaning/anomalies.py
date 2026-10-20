"""Descriptive anomaly REVIEW flags.

These flags never modify any value. They are computed on the full observed
history purely to help a human reviewer; because they use full-history and
centred-window statistics they are NOT valid model features (they would leak
future information). Fold-specific thresholds must be refit on d <= origin.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config


def series_extreme_flags(Y: np.ndarray, P: np.ndarray, wday: np.ndarray, cfg: Config):
    """Return (flag_mask, candidate_frame).

    A cell (series i, day t) is flagged when ALL hold:
      * the series is active (on/after its first positive sale),
      * units >= anomaly_min_units,
      * units > series' own active-history quantile (anomaly_series_quantile),
      * units >= anomaly_local_ratio * max(1, mean of active neighbours within
        +/- anomaly_local_halfwidth days, excluding day t).
    """
    n, D = Y.shape
    Yf = Y.astype(np.float64)
    pos = Y > 0
    has = pos.any(axis=1)
    first = np.argmax(pos, axis=1)
    active = (np.arange(D)[None, :] >= first[:, None]) & has[:, None]
    Ya = np.where(active, Yf, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanquantile(Ya, cfg.anomaly_series_quantile, axis=1)
        w = cfg.anomaly_local_halfwidth
        Z = np.where(active, Yf, 0.0)
        A = active.astype(np.float64)
        cz = np.concatenate([np.zeros((n, 1)), Z.cumsum(axis=1)], axis=1)
        ca = np.concatenate([np.zeros((n, 1)), A.cumsum(axis=1)], axis=1)
        t = np.arange(D)
        lo, hi = np.clip(t - w, 0, D), np.clip(t + w + 1, 0, D)
        wsum = cz[:, hi] - cz[:, lo] - Z
        wcnt = ca[:, hi] - ca[:, lo] - A
        local_mean = np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)
        base = np.maximum(np.nan_to_num(local_mean, nan=0.0), 1.0)
        mask = active & (Y >= cfg.anomaly_min_units) & (Yf > q[:, None]) & (Yf >= cfg.anomaly_local_ratio * base)

        r, c = np.nonzero(mask)
        if len(r) == 0:
            return mask, pd.DataFrame()
        # neighbour max (excluding self) only for candidates
        nmax = np.full(len(r), -np.inf)
        for off in range(-w, w + 1):
            if off == 0:
                continue
            cc = c + off
            ok = (cc >= 0) & (cc < D)
            v = np.full(len(r), -np.inf)
            v[ok] = np.nan_to_num(Ya[r[ok], cc[ok]], nan=-np.inf)
            nmax = np.maximum(nmax, v)
        # same-weekday mean over active history
        wmean = np.full((n, 8), np.nan)
        for k in range(1, 8):
            cols = wday == k
            wmean[:, k] = np.nanmean(Ya[:, cols], axis=1)
        med_price = np.nanmedian(P, axis=1)
    price = P[r, c]
    prev_price = np.where(c >= 7, P[r, np.maximum(c - 7, 0)], np.nan)
    units = Yf[r, c]
    isolated = nmax < units / cfg.isolated_spike_ratio
    reasons = np.where(isolated,
                       f"gt_series_q{cfg.anomaly_series_quantile};ge_{cfg.anomaly_local_ratio:g}x_local_mean;isolated_spike",
                       f"gt_series_q{cfg.anomaly_series_quantile};ge_{cfg.anomaly_local_ratio:g}x_local_mean")
    cand = pd.DataFrame({
        "row_item": r, "day_idx": c, "units_sold_raw": Y[r, c],
        "series_quantile_value": q[r], "local_mean_pm14_excl_day": local_mean[r, c],
        "local_max_pm14_excl_day": np.where(np.isfinite(nmax), nmax, np.nan),
        "ratio_to_local_mean": units / base[r, c],
        "same_weekday_mean": wmean[r, wday[c]],
        "ratio_to_same_weekday_mean": units / np.maximum(wmean[r, wday[c]], 1e-9),
        "sell_price_raw": price, "series_median_price": med_price[r],
        "price_ratio_to_series_median": price / med_price[r],
        "price_change_vs_7d_earlier": price - prev_price,
        "isolated_spike": isolated, "reasons": reasons,
    })
    return mask, cand


def store_day_low_flags(Y: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag store-days whose total units < store_day_low_ratio x median total
    of the +/- halfwidth surrounding days (excluding the day). Typical cause in
    M5: store closure (e.g. Christmas). Values are retained."""
    tot = Y.sum(axis=0).astype(np.float64)
    D = len(tot)
    w = cfg.anomaly_local_halfwidth
    med = np.empty(D)
    for t in range(D):
        nb = np.r_[tot[max(0, t - w):t], tot[t + 1:min(D, t + w + 1)]]
        med[t] = np.median(nb)
    flag = tot < cfg.store_day_low_ratio * med
    return flag, tot, med


def context_label(row_events: str, near_name: str | None, near_rel, snap: int,
                  price_ratio: float, cfg: Config) -> str:
    parts = []
    if isinstance(row_events, str) and row_events:
        parts.append(f"listed_event_day[{row_events}]")
    elif isinstance(near_name, str) and near_name and pd.notna(near_rel):
        parts.append(f"event_window[{near_name},{int(near_rel):+d}d]")
    if snap == 1:
        parts.append("snap_day")
    if np.isfinite(price_ratio) and price_ratio < cfg.price_reduction_ratio:
        parts.append(f"price_below_{int(cfg.price_reduction_ratio * 100)}pct_series_median")
    return "; ".join(parts) if parts else "no_calendar_or_price_context"
