"""Builder module to generate feature matrices A, B, and C with identical rows and splits."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import FeatureConfig
from .events import (
    NONE,
    OUTSIDE,
    day_level_event_features,
    event_date_table,
    marker_inspection,
    select_events,
    slug,
)
from .features import (
    compute_aggregate_history,
    compute_origin_price_features,
    compute_origin_sales_features,
)
from .inputs import load_calendar, load_wide, verify_panel_calendar_consistency
from .splits import assign_folds, build_origins, check_splits, repeated_target_dates, season_blocks

PHASE_CATEGORIES = [OUTSIDE, "pre_14_8", "pre_7_2", "pre_1", "day_0", "post_1_3", "post_4_7"]


def build_feature_datasets(
    cfg: FeatureConfig,
    log: Callable[[str], None],
    skip_existing: bool = True,
) -> dict:
    """Build feature matrices A, B, and C, ensuring identical row order and splits."""
    t0 = time.time()
    out_dir = Path(cfg.out_dir)
    manifest_dir = out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Calendar and Wide Data
    log("Loading calendar features...")
    cal = load_calendar(Path(cfg.calendar_features_path))

    log("Verifying cleaned panel schema and consistency with calendar...")
    cal_chk = verify_panel_calendar_consistency(Path(cfg.panel_path), cal)
    log(f"Calendar check: {cal_chk['event_slot_mismatches']} event mismatches, {cal_chk['snap_mismatches']} SNAP mismatches")

    log("Loading wide matrices from cleaned panel...")
    wide = load_wide(Path(cfg.panel_path), log)
    Y = wide["Y"]
    P = wide["P"]
    static = wide["static"]
    S, D = wide["S"], wide["D"]

    # Convert static string columns to Categorical
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        static[col] = static[col].astype("category")

    # Series identifier
    static["id"] = static["item_id"].astype(str) + "_" + static["store_id"].astype(str)

    # First sale date index per series (for active status determination)
    pos_sales = Y > 0
    first_sale_idx = np.where(pos_sales.any(axis=1), np.argmax(pos_sales, axis=1), 9999)

    # 2. Splits and Seasons
    log("Building season blocks and origin sampling schedule...")
    seasons = season_blocks(cal, cfg)
    test_start = int(seasons.set_index("season").loc[cfg.test_season, "block_start_d"])
    log(f"Final test start: d_{test_start} ({cal.loc[cal.d_num == test_start, 'date'].iloc[0].date()})")

    origins = build_origins(cal, D, cfg)
    folds, assign = assign_folds(origins, seasons, cfg)
    split_chk = check_splits(origins, folds, assign, test_start)
    log(f"Split check results: {split_chk}")

    # 3. Dev-Only Event Screening & Marker Inspection
    log("Deriving event date table and inspecting multi-day markers...")
    ev = event_date_table(cal, test_start)
    markers = marker_inspection(ev, cal)

    log("Screening events on development data only (d_num < test_start)...")
    # Dev store totals for screening: shape (n_stores, test_start - 1)
    store_codes = static["store_id"].cat.codes.to_numpy()
    n_stores = len(static["store_id"].cat.categories)
    T_dev = np.zeros((n_stores, test_start - 1), dtype=np.float64)
    for s in range(n_stores):
        T_dev[s] = Y[store_codes == s, : test_start - 1].sum(axis=0)

    sel_events, effect_profiles = select_events(T_dev, ev, cfg)
    log(f"Event selection completed: {int((sel_events.decision == 'named').sum())} named events, {int((sel_events.decision == 'pooled').sum())} pooled events.")

    # Compute dev-only category median price for normalization
    log("Computing category median price on development data only...")
    cat_codes = static["cat_id"].cat.codes.to_numpy()
    n_cats = len(static["cat_id"].cat.categories)
    cat_medians = np.zeros(n_cats, dtype=np.float32)
    P_dev = P[:, : test_start - 1]
    for c in range(n_cats):
        vals = P_dev[cat_codes == c]
        valid = vals[~np.isnan(vals)]
        cat_medians[c] = float(np.median(valid)) if len(valid) > 0 else 1.0
    cat_median_price_per_series = cat_medians[cat_codes]

    # Day-level event lookup tables for B and C
    day_B, day_C, named_event_list, family_list, rep_chk = day_level_event_features(cal, ev, sel_events, cfg)
    log(f"Calendar event representation check: {rep_chk}")

    # 4. Prepare Output Directories
    mat_A_dir = out_dir / "matrix_A"
    mat_B_dir = out_dir / "matrix_B"
    mat_C_dir = out_dir / "matrix_C"
    for d in [mat_A_dir, mat_B_dir, mat_C_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 5. Process Origin by Origin
    log(f"Building feature matrices A, B, and C across {len(origins)} origins...")
    H = cfg.horizon
    total_active_rows = 0
    origin_stats = []

    # Pre-compute date and SNAP lookups for fast retrieval
    cal_indexed = cal.set_index("d_num")
    wday_lookup = cal_indexed["wday"].to_numpy(dtype=np.int8)
    month_lookup = cal_indexed["month"].to_numpy(dtype=np.int8)
    year_lookup = cal_indexed["year"].to_numpy(dtype=np.int16)
    date_lookup = cal_indexed["date"].to_numpy()
    snap_CA = cal_indexed["snap_CA"].to_numpy(dtype=np.int8)
    snap_TX = cal_indexed["snap_TX"].to_numpy(dtype=np.int8)
    snap_WI = cal_indexed["snap_WI"].to_numpy(dtype=np.int8)

    # State code mapping for series
    state_arr = static["state_id"].to_numpy()

    # Define categorical types for Matrix B event columns
    event_names_cat = sorted(list(set(cal["event_name_1"].dropna()) | set(cal["event_name_2"].dropna()) | {NONE}))
    event_types_cat = sorted(list(set(cal["event_type_1"].dropna()) | set(cal["event_type_2"].dropna()) | {NONE}))

    for o_idx, o_row in origins.iterrows():
        O_d = int(o_row["origin_d"])
        O_idx = O_d - 1  # 0-indexed day
        origin_id = o_row["origin_id"]
        origin_date_val = str(o_row["origin_date"].date())

        # Active series filter: first positive sale <= O - 27 (at least 28 days of active history)
        active_mask = first_sale_idx <= (O_d - 28)
        n_act = int(active_mask.sum())
        n_rows_origin = n_act * H
        total_active_rows += n_rows_origin

        origin_stats.append({
            "origin_id": origin_id,
            "origin_d": O_d,
            "origin_date": origin_date_val,
            "active_series": n_act,
            "inactive_series": S - n_act,
            "total_rows": n_rows_origin,
        })

        p_C_file = mat_C_dir / f"origin_id={origin_id}" / "data.parquet"
        if skip_existing and p_C_file.exists():
            continue

        # Static hierarchy for active series
        act_static = static[active_mask].reset_index(drop=True)
        act_states = state_arr[active_mask]

        # Origin-anchored sales history
        sales_feats = compute_origin_sales_features(Y[active_mask], O_idx)
        agg_feats = compute_aggregate_history(Y, static, O_idx)
        # Filter agg_feats for active series
        for k in agg_feats:
            agg_feats[k] = agg_feats[k][active_mask]

        # Origin-anchored price history
        price_feats = compute_origin_price_features(
            P[active_mask], O_idx, cat_median_price_per_series[active_mask]
        )

        # Build table for horizons h = 1..28
        # Repetition vectors: series repeated H times, h tiled across series
        rep_indices = np.repeat(np.arange(n_act), H)
        h_vec = np.tile(np.arange(1, H + 1, dtype=np.int16), n_act)
        target_d_vec = O_d + h_vec.astype(np.int32)
        target_idx_vec = target_d_vec - 1

        # Target actual sales label
        # Shape of Y: (S, D). Extract Y[active_series, target_d - 1]
        act_indices = np.nonzero(active_mask)[0]
        y_label = Y[act_indices[:, None], target_idx_vec.reshape(n_act, H)].ravel()

        # Target date calendar features
        target_wday = wday_lookup[target_idx_vec]
        target_month = month_lookup[target_idx_vec]
        target_year = year_lookup[target_idx_vec]
        target_dates = date_lookup[target_idx_vec]

        # SNAP for target date matched to each series' state
        snap_all = np.zeros(n_rows_origin, dtype=np.int8)
        for st_name, snap_lookup in [("CA", snap_CA), ("TX", snap_TX), ("WI", snap_WI)]:
            st_mask = np.repeat(act_states == st_name, H)
            snap_all[st_mask] = snap_lookup[target_idx_vec[st_mask]]

        # Base DataFrame common to A, B, and C
        base_dict = {
            "id": np.repeat(act_static["id"].to_numpy(), H),
            "item_id": pd.Categorical(np.repeat(act_static["item_id"].to_numpy(), H), categories=static["item_id"].cat.categories),
            "dept_id": pd.Categorical(np.repeat(act_static["dept_id"].to_numpy(), H), categories=static["dept_id"].cat.categories),
            "cat_id": pd.Categorical(np.repeat(act_static["cat_id"].to_numpy(), H), categories=static["cat_id"].cat.categories),
            "store_id": pd.Categorical(np.repeat(act_static["store_id"].to_numpy(), H), categories=static["store_id"].cat.categories),
            "state_id": pd.Categorical(np.repeat(act_static["state_id"].to_numpy(), H), categories=static["state_id"].cat.categories),
            "origin_id": origin_id,
            "origin_d": O_d,
            "origin_date": origin_date_val,
            "h": h_vec,
            "target_d": target_d_vec,
            "target_date": target_dates,
            "wday": target_wday,
            "month": target_month,
            "year": target_year,
            "snap_state": snap_all,
            "target_units_sold": y_label,
        }

        # Add origin-anchored sales features (repeated across H)
        for k, v in sales_feats.items():
            base_dict[k] = np.repeat(v, H)
        for k, v in agg_feats.items():
            base_dict[k] = np.repeat(v, H)

        # Add origin-anchored price features (repeated across H)
        for k, v in price_feats.items():
            base_dict[k] = np.repeat(v, H)

        # ---------------------------------------------------------------------
        # Matrix A (No Event Variables)
        # ---------------------------------------------------------------------
        df_A = pd.DataFrame(base_dict)
        table_A = pa.Table.from_pandas(df_A, preserve_index=False)
        p_A_dir = mat_A_dir / f"origin_id={origin_id}"
        p_A_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table_A, p_A_dir / "data.parquet", compression="zstd")

        # ---------------------------------------------------------------------
        # Matrix B (Same-Day Event Name/Type Variables)
        # ---------------------------------------------------------------------
        # Tile target date event features across active series
        day_B_subset = day_B.set_index("d_num").loc[target_d_vec[:H]]
        dict_B = dict(base_dict)
        for col in ["t_event_name_1", "t_event_type_1", "t_event_name_2", "t_event_type_2"]:
            cats = event_names_cat if "name" in col else event_types_cat
            dict_B[col] = pd.Categorical(np.tile(day_B_subset[col].to_numpy(), n_act), categories=cats)
        dict_B["t_event_count"] = np.tile(day_B_subset["t_event_count"].to_numpy(), n_act)
        dict_B["t_no_event"] = np.tile(day_B_subset["t_no_event"].to_numpy(), n_act)

        df_B = pd.DataFrame(dict_B)
        table_B = pa.Table.from_pandas(df_B, preserve_index=False)
        p_B_dir = mat_B_dir / f"origin_id={origin_id}"
        p_B_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table_B, p_B_dir / "data.parquet", compression="zstd")

        # ---------------------------------------------------------------------
        # Matrix C (Named/Grouped Event-Relative Phase Variables)
        # ---------------------------------------------------------------------
        day_C_subset = day_C.set_index("d_num").loc[target_d_vec[:H]]
        dict_C = dict(base_dict)
        dict_C["t_event_count"] = np.tile(day_B_subset["t_event_count"].to_numpy(), n_act)
        dict_C["t_no_event"] = np.tile(day_B_subset["t_no_event"].to_numpy(), n_act)

        for col in day_C.columns:
            if col == "d_num":
                continue
            vals = day_C_subset[col].to_numpy()
            tiled = np.tile(vals, n_act)
            if col.startswith("ph_"):
                dict_C[col] = pd.Categorical(tiled, categories=PHASE_CATEGORIES)
            elif col.startswith("k_"):
                dict_C[col] = tiled.astype(np.float32)
            else:
                dict_C[col] = tiled

        df_C = pd.DataFrame(dict_C)
        table_C = pa.Table.from_pandas(df_C, preserve_index=False)
        p_C_dir = mat_C_dir / f"origin_id={origin_id}"
        p_C_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table_C, p_C_dir / "data.parquet", compression="zstd")

        if (o_idx + 1) % 10 == 0 or (o_idx + 1) == len(origins):
            log(f"  Processed {o_idx + 1}/{len(origins)} origins ({origin_id}: {n_act:,} active series, {n_rows_origin:,} rows)")

    elapsed = time.time() - t0
    log(f"All feature matrices built in {elapsed:.1f}s. Total rows per matrix: {total_active_rows:,}")

    return {
        "origins": origins,
        "folds": folds,
        "assign": assign,
        "seasons": seasons,
        "sel_events": sel_events,
        "effect_profiles": effect_profiles,
        "markers": markers,
        "ev": ev,
        "rep_chk": rep_chk,
        "origin_stats": pd.DataFrame(origin_stats),
        "total_active_rows": total_active_rows,
        "cols_A": list(df_A.columns) if "df_A" in locals() else list(pq.ParquetFile(mat_A_dir / f"origin_id={origins['origin_id'].iloc[0]}" / "data.parquet").schema.names),
        "cols_B": list(df_B.columns) if "df_B" in locals() else list(pq.ParquetFile(mat_B_dir / f"origin_id={origins['origin_id'].iloc[0]}" / "data.parquet").schema.names),
        "cols_C": list(df_C.columns) if "df_C" in locals() else list(pq.ParquetFile(mat_C_dir / f"origin_id={origins['origin_id'].iloc[0]}" / "data.parquet").schema.names),
    }
