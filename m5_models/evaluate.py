"""Out-of-sample evaluation, metrics calculation, and stratified reporting module."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from m5_features.events import PHASE_CATEGORIES
from .benchmark import SeasonalNaiveBenchmark
from .config import ModelConfig
from .data import load_partition_subset


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard forecasting accuracy measures."""
    if mask is not None:
        mask = mask.astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "rmse": np.nan, "wape": np.nan, "mean_y": np.nan}
        
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2
    
    sum_y = float(np.sum(y_true))
    mean_y = float(np.mean(y_true))
    mae = float(np.mean(abs_err))
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    wape = float(np.sum(abs_err) / sum_y) if sum_y > 0 else np.nan
    
    return {
        "n": n,
        "mean_y": mean_y,
        "mae": mae,
        "bias": bias,
        "rmse": rmse,
        "wape": wape,
    }


def predict_evaluation_origins(
    models: Dict[str, Dict],
    benchmark: SeasonalNaiveBenchmark,
    eval_origs: List[str],
    fold_id: str,
    sampled_ids: Set[str],
    cfg: ModelConfig,
    fold_cat_medians: Optional[Dict[str, float]] = None,
    fold_day_C: Optional[pd.DataFrame] = None,
    log: callable = print,
) -> pd.DataFrame:
    """Generate predictions for benchmark, Model A, Model B, Model C, and Model D on eval origins."""
    log(f"Evaluating {len(eval_origs)} origins for fold {fold_id} across Benchmark, A, B, C, and D...")

    
    # We will load origin by origin to keep memory light (< 100 MB per origin)
    all_preds_list = []
    
    # Event mapping to tag events
    cal_p = Path(cfg.features_dir) / "manifest" / "event_date_mapping.csv"
    ev_df = pd.read_csv(cal_p) if cal_p.exists() else None
    
    for o_idx, orig_id in enumerate(eval_origs):
        # 1. Load coordinates and true label from Matrix A
        p_A = cfg.matrix_A_dir / f"origin_id={orig_id}" / "data.parquet"
        tbl_A = pq.ParquetFile(p_A).read()
        df_A = tbl_A.to_pandas()
        df_A = df_A[df_A["id"].isin(sampled_ids)].reset_index(drop=True)
        
        if len(df_A) == 0:
            continue
            
        # Update price_ratio_to_cat_median with fold training medians (zero look-ahead)
        if fold_cat_medians is not None and "last_price_known_at_origin" in df_A.columns and "cat_id" in df_A.columns:
            denom = df_A["cat_id"].map(fold_cat_medians).astype(float)
            df_A["price_ratio_to_cat_median"] = np.where(
                df_A["last_price_known_at_origin"].notna(),
                df_A["last_price_known_at_origin"] / denom,
                np.nan
            ).astype(np.float32)
            
        coords = df_A[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
                       "origin_id", "origin_d", "h", "target_d", "target_date", "target_units_sold",
                       "sales_last_O", "sales_zero_ratio_28d", "sales_mean_28d"]].copy()
        coords["fold_id"] = fold_id
        
        # 2. Benchmark prediction
        bench_preds, bench_avail = benchmark.predict(coords)
        coords["pred_bench"] = bench_preds
        coords["bench_available"] = bench_avail
        
        # 3. Model A prediction
        feat_A = models["model_A"]["feature_cols"]
        for c in models["model_A"]["cat_cols"]:
            df_A[c] = df_A[c].astype("category")
        pred_A = models["model_A"]["booster"].predict(df_A[feat_A])
        if cfg.clip_nonnegative:
            pred_A = np.clip(pred_A, 0.0, None)
        coords["pred_A"] = pred_A
        del df_A
        
        # 4. Model B prediction
        p_B = cfg.matrix_B_dir / f"origin_id={orig_id}" / "data.parquet"
        tbl_B = pq.ParquetFile(p_B).read()
        df_B = tbl_B.to_pandas()
        df_B = df_B[df_B["id"].isin(sampled_ids)].reset_index(drop=True)
        if fold_cat_medians is not None and "last_price_known_at_origin" in df_B.columns and "cat_id" in df_B.columns:
            denom = df_B["cat_id"].map(fold_cat_medians).astype(float)
            df_B["price_ratio_to_cat_median"] = np.where(
                df_B["last_price_known_at_origin"].notna(),
                df_B["last_price_known_at_origin"] / denom,
                np.nan
            ).astype(np.float32)
            
        feat_B = models["model_B"]["feature_cols"]
        for c in models["model_B"]["cat_cols"]:
            df_B[c] = df_B[c].astype("category")
        pred_B = models["model_B"]["booster"].predict(df_B[feat_B])
        if cfg.clip_nonnegative:
            pred_B = np.clip(pred_B, 0.0, None)
        coords["pred_B"] = pred_B
        
        # Extract same-day event info
        coords["t_event_name_1"] = df_B["t_event_name_1"].astype(str)
        coords["t_no_event"] = df_B["t_no_event"].to_numpy()
        
        # 5. Model C prediction
        if fold_day_C is not None:
            day_C_indexed = fold_day_C.set_index("d_num")
            c_cols = [c for c in fold_day_C.columns if c != "d_num"]
            df_C_feats = day_C_indexed[c_cols].loc[df_B["target_d"]].reset_index(drop=True)
            b_event_cols = {"t_event_name_1", "t_event_type_1", "t_event_name_2", "t_event_type_2"}
            base_cols_C = [c for c in df_B.columns if c not in b_event_cols]
            df_C = pd.concat([df_B[base_cols_C], df_C_feats], axis=1)
        else:
            p_C = cfg.matrix_C_dir / f"origin_id={orig_id}" / "data.parquet"
            tbl_C = pq.ParquetFile(p_C).read()
            df_C = tbl_C.to_pandas()
            df_C = df_C[df_C["id"].isin(sampled_ids)].reset_index(drop=True)
            if fold_cat_medians is not None and "last_price_known_at_origin" in df_C.columns and "cat_id" in df_C.columns:
                denom = df_C["cat_id"].map(fold_cat_medians).astype(float)
                df_C["price_ratio_to_cat_median"] = np.where(
                    df_C["last_price_known_at_origin"].notna(),
                    df_C["last_price_known_at_origin"] / denom,
                    np.nan
                ).astype(np.float32)
            
        for c in df_C.columns:
            if c.startswith("ph_"):
                df_C[c] = pd.Categorical(df_C[c], categories=PHASE_CATEGORIES)
                
        feat_C = models["model_C"]["feature_cols"]
        for c in models["model_C"]["cat_cols"]:
            df_C[c] = df_C[c].astype("category")
        pred_C = models["model_C"]["booster"].predict(df_C[feat_C])
        if cfg.clip_nonnegative:
            pred_C = np.clip(pred_C, 0.0, None)
        coords["pred_C"] = pred_C
        
        # Extract phase info for key events
        if "ph_thanksgiving" in df_C.columns:
            coords["ph_thanksgiving"] = df_C["ph_thanksgiving"].astype(str)
        if "ph_christmas" in df_C.columns:
            coords["ph_christmas"] = df_C["ph_christmas"].astype(str)
            
        # 6. Model D prediction (Matrix B + C event-window features)
        if "model_D" in models:
            if fold_day_C is not None:
                day_C_indexed = fold_day_C.set_index("d_num")
                c_extra = [c for c in fold_day_C.columns if c.startswith("ph_") or c.startswith("k_") or c.startswith("n_fam_") or c.startswith("span_")]
                df_C_extra = day_C_indexed[c_extra].loc[df_B["target_d"]].reset_index(drop=True)
            else:
                c_extra = [c for c in df_C.columns if c.startswith("ph_") or c.startswith("k_")]
                df_C_extra = df_C[c_extra].reset_index(drop=True)
            
            assert len(df_B) == len(df_C_extra), (
                f"Row mismatch between B ({len(df_B)}) and C extra ({len(df_C_extra)}) in origin {orig_id}"
            )
            df_D = pd.concat([df_B, df_C_extra], axis=1)
            for c in df_D.columns:
                if c.startswith("ph_"):
                    df_D[c] = pd.Categorical(df_D[c], categories=PHASE_CATEGORIES)
                    
            feat_D = models["model_D"]["feature_cols"]
            for c in models["model_D"]["cat_cols"]:
                df_D[c] = df_D[c].astype("category")
            pred_D = models["model_D"]["booster"].predict(df_D[feat_D])
            if cfg.clip_nonnegative:
                pred_D = np.clip(pred_D, 0.0, None)
            coords["pred_D"] = pred_D
            del df_D
            
        del df_B, df_C
        all_preds_list.append(coords)
        
    full_preds = pd.concat(all_preds_list, ignore_index=True)
    
    # Calculate signed errors (forecast minus observed)
    y = full_preds["target_units_sold"].to_numpy()
    full_preds["err_bench"] = full_preds["pred_bench"] - y
    full_preds["err_A"] = full_preds["pred_A"] - y
    full_preds["err_B"] = full_preds["pred_B"] - y
    full_preds["err_C"] = full_preds["pred_C"] - y
    if "pred_D" in full_preds.columns:
        full_preds["err_D"] = full_preds["pred_D"] - y
    
    # Absolute errors
    full_preds["abs_err_bench"] = np.abs(full_preds["err_bench"])
    full_preds["abs_err_A"] = np.abs(full_preds["err_A"])
    full_preds["abs_err_B"] = np.abs(full_preds["err_B"])
    full_preds["abs_err_C"] = np.abs(full_preds["err_C"])
    if "pred_D" in full_preds.columns:
        full_preds["abs_err_D"] = np.abs(full_preds["err_D"])
    
    # Assign general event phase category
    # Check if either Thanksgiving or Christmas is active, else check other events
    def assign_primary_phase(row):
        th_ph = row.get("ph_thanksgiving", "outside")
        xm_ph = row.get("ph_christmas", "outside")
        if th_ph != "outside" and th_ph != "nan":
            return th_ph
        if xm_ph != "outside" and xm_ph != "nan":
            return xm_ph
        if row.get("t_no_event", 1) == 0:
            return "day_0"
        return "outside"
        
    full_preds["primary_phase"] = full_preds.apply(assign_primary_phase, axis=1)
    
    # Define primary event name
    def assign_event_name(row):
        ev1 = str(row.get("t_event_name_1", "None"))
        if ev1 not in ["None", "nan", "<NA>"]:
            return ev1
        th_ph = row.get("ph_thanksgiving", "outside")
        xm_ph = row.get("ph_christmas", "outside")
        if th_ph != "outside" and th_ph != "nan":
            return "Thanksgiving_Window"
        if xm_ph != "outside" and xm_ph != "nan":
            return "Christmas_Window"
        return "None"
        
    full_preds["event_name"] = full_preds.apply(assign_event_name, axis=1)
    
    # Define horizon bands
    full_preds["horizon_band"] = pd.cut(
        full_preds["h"],
        bins=[0, 7, 14, 21, 28],
        labels=["h_01_07", "h_08_14", "h_15_21", "h_22_28"]
    )
    
    # Define volume tiers
    full_preds["volume_tier"] = pd.qcut(
        full_preds["sales_mean_28d"].rank(method="first"),
        q=3,
        labels=["Low Volume", "Medium Volume", "High Volume"]
    )
    
    # Define intermittency tiers
    full_preds["intermittency_tier"] = pd.cut(
        full_preds["sales_zero_ratio_28d"],
        bins=[-0.01, 0.20, 0.50, 0.80, 1.01],
        labels=["Continuous (<20% zero)", "Moderate (20-50% zero)", "Intermittent (50-80% zero)", "Highly Intermittent (>80% zero)"]
    )
    
    return full_preds


def aggregate_strata_metrics(
    df: pd.DataFrame,
    group_col: str,
    benchmark_common_only: bool = False,
) -> pd.DataFrame:
    """Compute MAE, Bias, RMSE, and WAPE across groups for Bench, A, B, and C."""
    records = []
    
    for g_name, grp in df.groupby(group_col, observed=False):
        if benchmark_common_only:
            sub = grp[grp["bench_available"]].copy()
        else:
            sub = grp.copy()
            
        y = sub["target_units_sold"].to_numpy()
        
        m_bench = compute_metrics(y, sub["pred_bench"].to_numpy(), mask=sub["bench_available"].to_numpy())
        m_A = compute_metrics(y, sub["pred_A"].to_numpy())
        m_B = compute_metrics(y, sub["pred_B"].to_numpy())
        m_C = compute_metrics(y, sub["pred_C"].to_numpy())
        
        has_D = "pred_D" in sub.columns
        if has_D:
            m_D = compute_metrics(y, sub["pred_D"].to_numpy())
            d_DB_mae = m_D["mae"] - m_B["mae"]
            d_DB_pct = (d_DB_mae / m_B["mae"] * 100.0) if m_B["mae"] > 0 else np.nan
            d_DA_mae = m_D["mae"] - m_A["mae"]
            d_DA_pct = (d_DA_mae / m_A["mae"] * 100.0) if m_A["mae"] > 0 else np.nan
        else:
            m_D = {"mae": np.nan, "bias": np.nan, "rmse": np.nan, "wape": np.nan}
            d_DB_mae, d_DB_pct, d_DA_mae, d_DA_pct = np.nan, np.nan, np.nan, np.nan
        
        # Paired differences
        d_BA_mae = m_B["mae"] - m_A["mae"]
        d_CB_mae = m_C["mae"] - m_B["mae"]
        
        d_BA_pct = (d_BA_mae / m_A["mae"] * 100.0) if m_A["mae"] > 0 else np.nan
        d_CB_pct = (d_CB_mae / m_B["mae"] * 100.0) if m_B["mae"] > 0 else np.nan
        
        records.append({
            "stratum": str(g_name),
            "n_rows": len(sub),
            "bench_coverage_pct": (sub["bench_available"].mean() * 100.0) if len(sub) > 0 else 0.0,
            "mean_observed": m_A["mean_y"],
            # MAE
            "mae_bench": m_bench["mae"],
            "mae_A": m_A["mae"],
            "mae_B": m_B["mae"],
            "mae_C": m_C["mae"],
            "mae_D": m_D["mae"],
            # Bias (forecast - observed)
            "bias_bench": m_bench["bias"],
            "bias_A": m_A["bias"],
            "bias_B": m_B["bias"],
            "bias_C": m_C["bias"],
            "bias_D": m_D["bias"],
            # RMSE
            "rmse_bench": m_bench["rmse"],
            "rmse_A": m_A["rmse"],
            "rmse_B": m_B["rmse"],
            "rmse_C": m_C["rmse"],
            "rmse_D": m_D["rmse"],
            # WAPE
            "wape_bench": m_bench["wape"],
            "wape_A": m_A["wape"],
            "wape_B": m_B["wape"],
            "wape_C": m_C["wape"],
            "wape_D": m_D["wape"],
            # Differences
            "delta_BA_mae": d_BA_mae,
            "delta_BA_pct": d_BA_pct,
            "delta_CB_mae": d_CB_mae,
            "delta_CB_pct": d_CB_pct,
            "delta_DB_mae": d_DB_mae,
            "delta_DB_pct": d_DB_pct,
            "delta_DA_mae": d_DA_mae,
            "delta_DA_pct": d_DA_pct,
        })
        
    return pd.DataFrame(records)
