"""LightGBM training, hyperparameter tuning, and model persistence module."""
from __future__ import annotations

import gc
import json
import os
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import load_partition_subset, prepare_lgb_dataset


def select_best_objective(
    cfg: ModelConfig,
    train_origs: List[str],
    val_origs: List[str],
    sampled_ids: Set[str],
    log: callable = print,
) -> Tuple[str, pd.DataFrame]:
    """Compare regression_l1 vs poisson on DEV_S2012 using Matrix A."""
    log("Comparing candidate objectives on development fold (DEV_S2012)...")
    
    # Load Matrix A data
    df_train, _, feat_cols, cat_cols = load_partition_subset(
        cfg.matrix_A_dir, train_origs, sampled_ids, cfg
    )
    df_val, _, _, _ = load_partition_subset(
        cfg.matrix_A_dir, val_origs, sampled_ids, cfg
    )
    
    trn_ds = prepare_lgb_dataset(df_train, feat_cols, cfg.label_col, cat_cols)
    val_ds = prepare_lgb_dataset(df_val, feat_cols, cfg.label_col, cat_cols, reference=trn_ds)
    
    y_val = df_val[cfg.label_col].to_numpy()
    
    results = []
    best_obj = None
    best_mae = float("inf")
    
    base_params = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "metric": "mae",
        "verbose": -1,
        "n_jobs": 8,
        "seed": cfg.seed,
    }
    
    for obj in cfg.candidate_objectives:
        t0 = time.time()
        params = dict(base_params)
        params["objective"] = obj
        
        evals_result = {}
        bst = lgb.train(
            params,
            trn_ds,
            num_boost_round=100,
            valid_sets=[trn_ds, val_ds],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=15, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )
        elapsed = time.time() - t0
        
        val_preds = bst.predict(df_val[feat_cols])
        if cfg.clip_nonnegative:
            val_preds = np.clip(val_preds, 0.0, None)
            
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        val_rmse = float(np.sqrt(np.mean((val_preds - y_val) ** 2)))
        
        log(f"  Objective: {obj:15s} | Best Iter: {bst.best_iteration:3d} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Time: {elapsed:.1f}s")
        
        results.append({
            "objective": obj,
            "best_iteration": bst.best_iteration,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "training_time_s": elapsed,
        })
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_obj = obj
            
    del df_train, df_val, trn_ds, val_ds
    gc.collect()
    
    log(f"Selected objective: '{best_obj}' with lowest validation MAE: {best_mae:.4f}")
    return best_obj, pd.DataFrame(results)


def tune_hyperparameters(
    cfg: ModelConfig,
    chosen_objective: str,
    train_origs: List[str],
    val_origs: List[str],
    sampled_ids: Set[str],
    log: callable = print,
) -> Tuple[Dict, pd.DataFrame]:
    """Run bounded hyperparameter search on development fold DEV_S2012."""
    log(f"Running hyperparameter grid search ({len(cfg.hyperparam_grid)} configurations)...")
    
    df_train, _, feat_cols, cat_cols = load_partition_subset(
        cfg.matrix_A_dir, train_origs, sampled_ids, cfg
    )
    df_val, _, _, _ = load_partition_subset(
        cfg.matrix_A_dir, val_origs, sampled_ids, cfg
    )
    
    trn_ds = prepare_lgb_dataset(df_train, feat_cols, cfg.label_col, cat_cols)
    val_ds = prepare_lgb_dataset(df_val, feat_cols, cfg.label_col, cat_cols, reference=trn_ds)
    y_val = df_val[cfg.label_col].to_numpy()
    
    records = []
    best_params = None
    best_mae = float("inf")
    
    for idx, grid_item in enumerate(cfg.hyperparam_grid):
        t0 = time.time()
        params = {
            "objective": chosen_objective,
            "metric": "mae",
            "verbose": -1,
            "n_jobs": 8,
            "seed": cfg.seed,
            "bagging_freq": 1 if grid_item.get("bagging_fraction", 1.0) < 1.0 else 0,
            **grid_item,
        }
        
        evals_result = {}
        bst = lgb.train(
            params,
            trn_ds,
            num_boost_round=cfg.max_boost_rounds,
            valid_sets=[trn_ds, val_ds],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )
        elapsed = time.time() - t0
        
        val_preds = bst.predict(df_val[feat_cols])
        if cfg.clip_nonnegative:
            val_preds = np.clip(val_preds, 0.0, None)
            
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        
        rec = {
            "trial_id": idx + 1,
            **grid_item,
            "best_iteration": bst.best_iteration,
            "val_mae": val_mae,
            "time_s": elapsed,
        }
        records.append(rec)
        log(f"  Trial {idx + 1}/{len(cfg.hyperparam_grid)}: lr={grid_item['learning_rate']}, leaves={grid_item['num_leaves']}, min_leaf={grid_item['min_data_in_leaf']} -> Val MAE: {val_mae:.4f} (best iter {bst.best_iteration})")
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            
    del df_train, df_val, trn_ds, val_ds
    gc.collect()
    
    log(f"Best hyperparameters selected (Val MAE: {best_mae:.4f}): {best_params}")
    return best_params, pd.DataFrame(records)


def train_single_model(
    matrix_dir: Path,
    matrix_name: str,
    train_origs: List[str],
    val_origs: List[str],
    sampled_ids: Set[str],
    params: Dict,
    cfg: ModelConfig,
    fold_cat_medians: Optional[Dict[str, float]] = None,
    log: callable = print,
) -> Dict:
    """Train one LightGBM model for specified matrix and origin partitions."""
    process = psutil.Process()
    m0 = process.memory_info().rss / (1024**2)
    t0 = time.time()
    
    log(f"Training Model {matrix_name} ({len(train_origs)} train origins, {len(val_origs)} val origins)...")
    
    df_train, _, feat_cols, cat_cols = load_partition_subset(
        matrix_dir, train_origs, sampled_ids, cfg, matrix_name=matrix_name, fold_cat_medians=fold_cat_medians
    )
    df_val, _, _, _ = load_partition_subset(
        matrix_dir, val_origs, sampled_ids, cfg, matrix_name=matrix_name, fold_cat_medians=fold_cat_medians
    )
    
    trn_ds = prepare_lgb_dataset(df_train, feat_cols, cfg.label_col, cat_cols)
    val_ds = prepare_lgb_dataset(df_val, feat_cols, cfg.label_col, cat_cols, reference=trn_ds)
    
    evals_result = {}
    bst = lgb.train(
        params,
        trn_ds,
        num_boost_round=cfg.max_boost_rounds,
        valid_sets=[trn_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )
    
    elapsed = time.time() - t0
    m1 = process.memory_info().rss / (1024**2)
    
    # Feature importances
    imp_gain = bst.feature_importance(importance_type="gain")
    imp_split = bst.feature_importance(importance_type="split")
    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance_gain": imp_gain,
        "importance_split": imp_split,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    
    del df_train, df_val, trn_ds, val_ds
    gc.collect()
    
    log(f"  Model {matrix_name} trained in {elapsed:.1f}s | Best Iteration: {bst.best_iteration} | Peak RSS: {m1:.1f} MB")
    
    return {
        "matrix_name": matrix_name,
        "booster": bst,
        "feature_cols": feat_cols,
        "cat_cols": cat_cols,
        "best_iteration": bst.best_iteration,
        "evals_result": evals_result,
        "feature_importance": df_imp,
        "training_time_s": elapsed,
        "peak_rss_mb": m1,
    }
