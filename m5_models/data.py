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


def load_partition_subset(
    matrix_dir: Path,
    origin_ids: List[str],
    sampled_ids: Set[str],
    cfg: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Load and concatenate data for specified origins and study population."""
    dfs = []
    coords_dfs = []
    
    # Feature columns to exclude from training
    drop_from_X = set(cfg.key_cols + [cfg.label_col])
    
    feature_cols = None
    cat_cols = None

    for orig_id in origin_ids:
        p = matrix_dir / f"origin_id={orig_id}" / "data.parquet"
        if not p.exists():
            continue
        
        pf = pq.ParquetFile(p)
        tbl = pf.read()
        df = tbl.to_pandas()
        
        # Filter to study population series
        df_sub = df[df["id"].isin(sampled_ids)].copy()
        if len(df_sub) == 0:
            continue
            
        if feature_cols is None:
            feature_cols = [c for c in df_sub.columns if c not in drop_from_X]
            # Identify categoricals
            cat_cols = [c for c in feature_cols if str(df_sub[c].dtype) == "category" or df_sub[c].dtype == object]
        
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
