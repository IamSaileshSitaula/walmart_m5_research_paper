"""Computation of origin-anchored sales history and price history features.

All features are strictly anchored at forecast origin O (d_num <= O).
No target-period sales or target-week prices are ever accessed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_origin_sales_features(Y: np.ndarray, O_idx: int) -> dict[str, np.ndarray]:
    """Compute series-level trailing sales statistics ending at day O_idx (0-indexed).

    Parameters
    ----------
    Y : np.ndarray, shape (S, D)
        Dense sales matrix (int16/int32).
    O_idx : int
        0-indexed day index of forecast origin O.

    Returns
    -------
    dict of str -> np.ndarray of shape (S,)
    """
    S = Y.shape[0]
    out = {}

    # Exact day lags: O, O-7, O-28
    out["sales_last_O"] = Y[:, O_idx].astype(np.float32)
    out["sales_last_O_7"] = Y[:, O_idx - 7].astype(np.float32) if O_idx >= 7 else np.full(S, np.nan, dtype=np.float32)
    out["sales_last_O_28"] = Y[:, O_idx - 28].astype(np.float32) if O_idx >= 28 else np.full(S, np.nan, dtype=np.float32)

    # Trailing 7-day window [O-6 .. O]
    w7 = Y[:, max(0, O_idx - 6) : O_idx + 1].astype(np.float32)
    out["sales_mean_7d"] = w7.mean(axis=1)
    out["sales_std_7d"] = w7.std(axis=1)
    out["sales_zero_ratio_7d"] = (w7 == 0).mean(axis=1).astype(np.float32)

    # Trailing 28-day window [O-27 .. O]
    w28 = Y[:, max(0, O_idx - 27) : O_idx + 1].astype(np.float32)
    out["sales_mean_28d"] = w28.mean(axis=1)
    out["sales_std_28d"] = w28.std(axis=1)
    out["sales_zero_ratio_28d"] = (w28 == 0).mean(axis=1).astype(np.float32)

    # Trailing 56-day window [O-55 .. O]
    w56 = Y[:, max(0, O_idx - 55) : O_idx + 1].astype(np.float32)
    out["sales_mean_56d"] = w56.mean(axis=1)
    out["sales_std_56d"] = w56.std(axis=1)
    out["sales_zero_ratio_56d"] = (w56 == 0).mean(axis=1).astype(np.float32)

    return out


def compute_aggregate_history(Y: np.ndarray, static: pd.DataFrame, O_idx: int) -> dict[str, np.ndarray]:
    """Compute store-level and category-level trailing mean sales ending at O_idx."""
    # Pre-aggregated store totals
    store_codes = static["store_id"].cat.codes.to_numpy()
    cat_codes = static["cat_id"].cat.codes.to_numpy()
    n_stores = len(static["store_id"].cat.categories)
    n_cats = len(static["cat_id"].cat.categories)

    # Trailing 7-day and 28-day windows
    w7 = Y[:, max(0, O_idx - 6) : O_idx + 1].astype(np.float32)
    w28 = Y[:, max(0, O_idx - 27) : O_idx + 1].astype(np.float32)

    s_mean7 = np.zeros(n_stores, dtype=np.float32)
    s_mean28 = np.zeros(n_stores, dtype=np.float32)
    for s in range(n_stores):
        mask = store_codes == s
        s_mean7[s] = w7[mask].mean()
        s_mean28[s] = w28[mask].mean()

    c_mean7 = np.zeros(n_cats, dtype=np.float32)
    c_mean28 = np.zeros(n_cats, dtype=np.float32)
    for c in range(n_cats):
        mask = cat_codes == c
        c_mean7[c] = w7[mask].mean()
        c_mean28[c] = w28[mask].mean()

    return {
        "store_sales_mean_7d": s_mean7[store_codes],
        "store_sales_mean_28d": s_mean28[store_codes],
        "cat_sales_mean_7d": c_mean7[cat_codes],
        "cat_sales_mean_28d": c_mean28[cat_codes],
    }


def compute_origin_price_features(P: np.ndarray, O_idx: int, cat_median_price: np.ndarray) -> dict[str, np.ndarray]:
    """Compute price features as known at origin O_idx (0-indexed).

    Uses only prices on or before O_idx.
    """
    S = P.shape[0]
    # Restrict price history to days <= O_idx
    P_hist = P[:, : O_idx + 1]

    # Find last valid (non-NaN) price for each series on or before O_idx
    valid = ~np.isnan(P_hist)
    has_price = valid.any(axis=1)

    # Reverse along axis 1 to find the index of the latest valid price
    # argmax on reversed boolean array gives distance from O_idx
    rev_dist = np.argmax(valid[:, ::-1], axis=1)
    last_idx = O_idx - rev_dist

    last_price = np.where(has_price, P_hist[np.arange(S), last_idx], np.nan).astype(np.float32)
    price_age = np.where(has_price, (O_idx - last_idx).astype(np.float32), np.nan).astype(np.float32)

    # Price 28 days prior to O_idx (if available)
    if O_idx >= 28:
        # Find latest valid price up to O_idx - 28
        P_prev = P[:, : O_idx - 27]
        val_prev = ~np.isnan(P_prev)
        has_prev = val_prev.any(axis=1)
        prev_idx = (O_idx - 28) - np.argmax(val_prev[:, ::-1], axis=1)
        prev_price = np.where(has_prev, P_prev[np.arange(S), prev_idx], np.nan).astype(np.float32)
        price_change_28d = last_price - prev_price
    else:
        price_change_28d = np.full(S, np.nan, dtype=np.float32)

    # Ratio to series category median
    price_ratio_cat = (last_price / cat_median_price).astype(np.float32)

    return {
        "last_price_known_at_origin": last_price,
        "last_price_age_days": price_age,
        "price_change_28d": price_change_28d,
        "price_ratio_to_cat_median": price_ratio_cat,
    }
