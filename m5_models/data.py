"""Data loading and feature preparation module for LightGBM models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import ModelConfig


def get_study_population_series(
    cfg: ModelConfig,
) -> Tuple[Set[str], pd.DataFrame]:
    """Extract reproducible 10% stratified sample of series (3,050 series)."""
    # Sample from the latest origin partition to ensure full catalog coverage
    sample_pf = pq.ParquetFile(cfg.matrix_A_dir / "origin_id=O1913" / "data.parquet")
    df_static = sample_pf.read(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]).to_pandas().drop_duplicates("id")
    
    # Sort items systematically and take every stride-th item
    unique_items = sorted(df_static["item_id"].unique())
    sampled_items = set(unique_items[::cfg.sample_item_stride])  # 305 items
    
    df_study = df_static[df_static["item_id"].isin(sampled_items)].sort_values(["store_id", "item_id"]).reset_index(drop=True)
    sampled_ids = set(df_study["id"].tolist())
    return sampled_ids, df_study


def compute_fold_category_median_prices(
    origin_ids: List[str],
    sampled_ids: Set[str],
    cfg: ModelConfig,
) -> Dict[str, float]:
    """Compute category median price strictly using the fold's training origins.
    
    Eliminates price look-ahead for earlier development folds.
    """
    prices_by_cat = {}
    for orig_id in origin_ids:
        p = cfg.matrix_A_dir / f"origin_id={orig_id}" / "data.parquet"
        if not p.exists():
            continue
        tbl = pq.ParquetFile(p).read(columns=["id", "cat_id", "last_price_known_at_origin"])
        df = tbl.to_pandas()
        df = df[df["id"].isin(sampled_ids)].drop_duplicates("id")
        for cat, g in df.groupby("cat_id", observed=True):
            valid = g["last_price_known_at_origin"].dropna().to_numpy()
            if cat not in prices_by_cat:
                prices_by_cat[cat] = []
            prices_by_cat[cat].extend(valid)
            
    cat_medians = {}
    for cat, vals in prices_by_cat.items():
        cat_medians[cat] = float(np.median(vals)) if len(vals) > 0 else 1.0
    return cat_medians


from m5_features.events import PHASE_CATEGORIES


def load_partition_subset(
    matrix_dir: Path,
    origin_ids: List[str],
    sampled_ids: Set[str],
    cfg: ModelConfig,
    matrix_name: Optional[str] = None,
    fold_cat_medians: Optional[Dict[str, float]] = None,
    fold_day_C: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load and concatenate data for specified origins and study population.
    
    Retains forecast horizon 'h' as an explicit numeric predictor.
    Supports Matrix C and D with fold-specific chronological event mapping (fold_day_C).
    Applies fold-specific category median price normalization if provided.
    Asserts exact coordinate and row alignment across matrices.
    """
    dfs = []
    coords_dfs = []
    
    # Feature columns to exclude from training (keeps 'h' in X!)
    drop_from_X = set(cfg.feature_excluded_cols + [cfg.label_col])
    
    is_matrix_D = (matrix_name is not None and ("model_D" in matrix_name or "matrix_D" in matrix_name)) or (matrix_dir.name in ["matrix_D", "model_D"])
    is_matrix_C = (matrix_name is not None and ("model_C" in matrix_name or "matrix_C" in matrix_name)) or (matrix_dir.name in ["matrix_C", "model_C"])
    
    feature_cols = None
    cat_cols = None

    for orig_id in origin_ids:
        if is_matrix_D:
            p_B = cfg.matrix_B_dir / f"origin_id={orig_id}" / "data.parquet"
            if not p_B.exists():
                continue
            df_B = pq.ParquetFile(p_B).read().to_pandas()
            b_mask = df_B["id"].isin(sampled_ids).to_numpy()
            df_B_sub = df_B[b_mask].reset_index(drop=True)
            
            if fold_day_C is not None:
                day_C_indexed = fold_day_C.set_index("d_num")
                c_extra = [c for c in fold_day_C.columns if c.startswith("ph_") or c.startswith("k_") or c.startswith("n_fam_") or c.startswith("span_")]
                df_C_sub = day_C_indexed[c_extra].loc[df_B_sub["target_d"]].reset_index(drop=True)
            else:
                p_C = cfg.matrix_C_dir / f"origin_id={orig_id}" / "data.parquet"
                if not p_C.exists():
                    continue
                pf_C = pq.ParquetFile(p_C)
                coord_cols = [c for c in ["id", "h", "target_d"] if c in pf_C.schema.names]
                c_extra = [c for c in pf_C.schema.names if c.startswith("ph_") or c.startswith("k_")]
                df_C_raw = pf_C.read(columns=coord_cols + c_extra).to_pandas()
                df_C_sub = df_C_raw[b_mask].reset_index(drop=True)
                
                assert len(df_B_sub) == len(df_C_sub), (
                    f"Row count mismatch between B ({len(df_B_sub)}) and C ({len(df_C_sub)}) for origin {orig_id}"
                )
                for coord in ["id", "h", "target_d"]:
                    if coord in df_C_sub.columns and coord in df_B_sub.columns:
                        assert (df_B_sub[coord].to_numpy() == df_C_sub[coord].to_numpy()).all(), (
                            f"Coordinate mismatch on '{coord}' between Matrix B and Matrix C for origin {orig_id}"
                        )
                df_C_sub = df_C_sub[c_extra]
            
            assert len(df_B_sub) == len(df_C_sub), (
                f"Row count mismatch between B ({len(df_B_sub)}) and C ({len(df_C_sub)}) for origin {orig_id}"
            )
            df_sub = pd.concat([df_B_sub, df_C_sub], axis=1)
        elif is_matrix_C and fold_day_C is not None:
            p_A = cfg.matrix_A_dir / f"origin_id={orig_id}" / "data.parquet"
            p_B = cfg.matrix_B_dir / f"origin_id={orig_id}" / "data.parquet"
            if not (p_A.exists() and p_B.exists()):
                continue
            df_A = pq.ParquetFile(p_A).read().to_pandas()
            a_mask = df_A["id"].isin(sampled_ids).to_numpy()
            df_sub_base = df_A[a_mask].reset_index(drop=True)
            
            pf_B = pq.ParquetFile(p_B)
            coord_cols_B = [c for c in ["id", "h", "target_d"] if c in pf_B.schema.names]
            df_B_events = pf_B.read(columns=coord_cols_B + ["t_event_count", "t_no_event"]).to_pandas()
            df_B_sub_ev = df_B_events[a_mask].reset_index(drop=True)
            for coord in ["id", "h", "target_d"]:
                if coord in df_sub_base.columns and coord in df_B_sub_ev.columns:
                    assert (df_sub_base[coord].to_numpy() == df_B_sub_ev[coord].to_numpy()).all(), (
                        f"Coordinate mismatch on '{coord}' between Matrix A and Matrix B for origin {orig_id}"
                    )
            df_sub_base["t_event_count"] = df_B_sub_ev["t_event_count"].to_numpy()
            df_sub_base["t_no_event"] = df_B_sub_ev["t_no_event"].to_numpy()
            
            day_C_indexed = fold_day_C.set_index("d_num")
            c_cols = [c for c in fold_day_C.columns if c != "d_num"]
            df_C_feats = day_C_indexed[c_cols].loc[df_sub_base["target_d"]].reset_index(drop=True)
            assert len(df_sub_base) == len(df_C_feats)
            df_sub = pd.concat([df_sub_base, df_C_feats], axis=1)
        else:
            p = matrix_dir / f"origin_id={orig_id}" / "data.parquet"
            if not p.exists():
                continue
            
            pf = pq.ParquetFile(p)
            tbl = pf.read()
            df = tbl.to_pandas()
            df_sub = df[df["id"].isin(sampled_ids)].copy().reset_index(drop=True)
            
        if len(df_sub) == 0:
            continue
            
        # Recompute price_ratio_to_cat_median using fold-specific training medians (zero look-ahead)
        if fold_cat_medians is not None and "last_price_known_at_origin" in df_sub.columns and "cat_id" in df_sub.columns:
            denom = df_sub["cat_id"].map(fold_cat_medians).astype(float)
            df_sub["price_ratio_to_cat_median"] = np.where(
                df_sub["last_price_known_at_origin"].notna(),
                df_sub["last_price_known_at_origin"] / denom,
                np.nan
            ).astype(np.float32)
            
        # Ensure all phase columns have unified categorical type
        for c in df_sub.columns:
            if c.startswith("ph_"):
                df_sub[c] = pd.Categorical(df_sub[c], categories=PHASE_CATEGORIES)
            
        if feature_cols is None:
            feature_cols = [c for c in df_sub.columns if c not in drop_from_X]
            # Ensure 'h' is present
            if "h" not in feature_cols and "h" in df_sub.columns:
                feature_cols.insert(0, "h")
            # Identify categoricals (ensure all ph_ event phases are treated as native categoricals)
            cat_cols = [c for c in feature_cols if str(df_sub[c].dtype) == "category" or df_sub[c].dtype == object or c.startswith("ph_")]
        
        # Keep coordinate columns separately
        coord_df = df_sub[["id", "origin_id", "origin_d", "h", "target_d", "target_date", "target_units_sold"]].copy()
        coords_dfs.append(coord_df)
        
        # Extract features and ensure categorical types
        for c in cat_cols:
            df_sub[c] = df_sub[c].astype("category")
            
        dfs.append(df_sub[feature_cols + [cfg.label_col]])

        
    if not dfs:
        raise ValueError(f"No data loaded for origins: {origin_ids}")
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_coords = pd.concat(coords_dfs, ignore_index=True)
    
    return full_df, full_coords, feature_cols, cat_cols


def prepare_lgb_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cat_cols: List[str],
    reference: Optional[lgb.Dataset] = None,
) -> lgb.Dataset:
    """Build LightGBM Dataset with native categoricals and memory efficiency."""
    X = df[feature_cols]
    y = df[label_col].to_numpy(dtype=np.float32)
    
    ds = lgb.Dataset(
        data=X,
        label=y,
        categorical_feature=cat_cols,
        reference=reference,
        free_raw_data=False,  # Retain for out-of-sample prediction and validation scoring
    )
    return ds
