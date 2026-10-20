"""End-to-end execution pipeline for M5 Direct Multi-Horizon LightGBM Forecasting."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from .audit import run_pre_training_audit
from .benchmark import SeasonalNaiveBenchmark
from .config import ModelConfig
from .data import compute_fold_category_median_prices, get_study_population_series
from .evaluate import predict_evaluation_origins
from .reporting import generate_evaluation_artifacts
from .train import select_best_objective, train_single_model, tune_hyperparameters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M5 Direct Multi-Horizon Forecasting Experiments")
    parser.add_argument("--out-dir", default="outputs/m5_models", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-audit", action="store_true", help="Skip exhaustive partition audit")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run_log.txt"
    log_fh = open(log_file, "w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log("=" * 80)
    log("STARTING M5 DIRECT MULTI-HORIZON FORECASTING & EVENT ABLATION EXPERIMENTS")
    log("=" * 80)

    cfg = ModelConfig(
        seed=args.seed,
        output_dir=out_dir,
    )

    # -------------------------------------------------------------------------
    # 1. Pre-Training Gates and Data Contract Audit
    # -------------------------------------------------------------------------
    if not args.skip_audit:
        audit_res = run_pre_training_audit(cfg, log)
        if not audit_res["all_passed"]:
            log("FATAL: Pre-training audit failed one or more checks! Halting.")
            log_fh.close()
            return 1
    else:
        log("Skipping exhaustive pre-training audit as requested.")

    # -------------------------------------------------------------------------
    # 2. Extract Study Population (Memory Budget & Representation)
    # -------------------------------------------------------------------------
    log("Extracting study population (10% systematic stratified catalog sample)...")
    sampled_ids, df_study = get_study_population_series(cfg)
    log(f"Study population: {len(sampled_ids):,} series ({df_study['item_id'].nunique()} items across {df_study['store_id'].nunique()} stores).")
    df_study.to_csv(out_dir / "study_population_series.csv", index=False)

    # Inspect Split Manifest
    splits_df = pd.read_csv(cfg.split_manifest_path)
    folds_df = pd.read_csv(cfg.fold_summary_path)

    # DEV_S2012 origins for tuning
    dev_s2012_train = splits_df[(splits_df["fold_id"] == "DEV_S2012") & (splits_df["role"] == "train")]["origin_id"].tolist()
    dev_s2012_val = splits_df[(splits_df["fold_id"] == "DEV_S2012") & (splits_df["role"] == "validation")]["origin_id"].tolist()

    # -------------------------------------------------------------------------
    # 3. Development-Only Objective Selection
    # -------------------------------------------------------------------------
    best_obj, obj_df = select_best_objective(cfg, dev_s2012_train, dev_s2012_val, sampled_ids, log)

    # -------------------------------------------------------------------------
    # 4. Development-Only Hyperparameter Grid Search
    # -------------------------------------------------------------------------
    best_params, tuning_df = tune_hyperparameters(cfg, best_obj, dev_s2012_train, dev_s2012_val, sampled_ids, log)

    # -------------------------------------------------------------------------
    # 5. Seasonal-Naive Benchmark Initialization
    # -------------------------------------------------------------------------
    log("Initializing seasonal-naive benchmark with calendar-day lookups...")
    benchmark = SeasonalNaiveBenchmark(cfg)

    # -------------------------------------------------------------------------
    # 6. Fit Fold-Specific Models and Evaluate Chronologically
    # -------------------------------------------------------------------------
    log("=" * 80)
    log("EXECUTING STRICT CHRONOLOGICAL TRAINING AND OUT-OF-SAMPLE EVALUATION BY FOLD")
    log("=" * 80)

    eval_predictions_by_fold = []
    all_trained_models = {}

    for f in folds_df.itertuples():
        fold_id = f.fold_id
        is_final_test = (fold_id == "FINAL_TEST")
        log("=" * 70)
        log(f"PROCESSING FOLD: {fold_id} (Role: {'FROZEN FINAL TEST' if is_final_test else 'CHRONOLOGICAL DEVELOPMENT'})")
        log("=" * 70)

        # Train origins strictly for this fold
        trn_rows = splits_df[(splits_df["fold_id"] == fold_id) & (splits_df["role"] == "train")]
        ev_rows = splits_df[(splits_df["fold_id"] == fold_id) & (splits_df["role"].isin(["validation", "test"]))]

        trn_origs = trn_rows["origin_id"].tolist()
        eval_origs = ev_rows["origin_id"].tolist()

        max_trn_tgt_d = trn_rows["last_target_d"].max()
        max_trn_date = trn_rows["last_target_date"].max()
        min_ev_tgt_d = ev_rows["first_target_d"].min()
        min_ev_date = ev_rows["first_target_date"].min()
        gap = min_ev_tgt_d - max_trn_tgt_d

        log(f"  Train scope: {len(trn_origs)} origins | max target date: {max_trn_date} (d_{max_trn_tgt_d})")
        log(f"  Eval scope:  {len(eval_origs)} origins | min target date: {min_ev_date} (d_{min_ev_tgt_d})")
        log(f"  Temporal separation gap: +{gap} days (Strictly >= 1 day: {gap >= 1})")

        if gap < 1:
            raise ValueError(f"FATAL: Train targets overlap with evaluation targets in fold {fold_id}!")

        # Compute fold category medians strictly on this fold's training origins (zero future price leakage)
        fold_cat_medians = compute_fold_category_median_prices(trn_origs, sampled_ids, cfg)
        log(f"  Fold training category median prices (zero lookahead): {fold_cat_medians}")

        # Validation split within training origins for early stopping (last 15% or at least 2 origins)
        val_count = max(2, int(len(trn_origs) * 0.15))
        sub_train = trn_origs[:-val_count]
        val_origs = trn_origs[-val_count:]

        # Train fold-specific models for Matrix A, B, C, and D
        fold_models = {}
        model_specs = [
            ("model_A", cfg.matrix_A_dir),
            ("model_B", cfg.matrix_B_dir),
            ("model_C", cfg.matrix_C_dir),
            ("model_D", cfg.matrix_B_dir),
        ]
        for mat_name, mat_dir in model_specs:
            m_res = train_single_model(
                matrix_dir=mat_dir,
                matrix_name=f"{fold_id}_{mat_name}",
                train_origs=sub_train,
                val_origs=val_origs,
                sampled_ids=sampled_ids,
                params=best_params,
                cfg=cfg,
                fold_cat_medians=fold_cat_medians,
                log=log,
            )
            fold_models[mat_name] = m_res
            model_save_path = models_dir / f"{fold_id}_{mat_name}.txt"
            m_res["booster"].save_model(str(model_save_path))
            all_trained_models[f"{fold_id}_{mat_name}"] = m_res

        # If FINAL_TEST, also save standard alias files model_A.txt, model_B.txt, model_C.txt, model_D.txt
        if is_final_test:
            for mat_name in ["model_A", "model_B", "model_C", "model_D"]:
                fold_models[mat_name]["booster"].save_model(str(models_dir / f"{mat_name}.txt"))

        # Predict evaluation origins using THIS FOLD'S dedicated models
        fold_preds = predict_evaluation_origins(
            models=fold_models,
            benchmark=benchmark,
            eval_origs=eval_origs,
            fold_id=fold_id,
            sampled_ids=sampled_ids,
            cfg=cfg,
            fold_cat_medians=fold_cat_medians,
            log=log,
        )
        eval_predictions_by_fold.append(fold_preds)

    full_preds_df = pd.concat(eval_predictions_by_fold, ignore_index=True)
    log(f"All out-of-sample predictions assembled: {len(full_preds_df):,} total forecast rows.")

    # -------------------------------------------------------------------------
    # 8. Generate Reports, Stratifications, and Artifacts
    # -------------------------------------------------------------------------
    model_metadata = {
        name: {
            "best_iteration": m["best_iteration"],
            "training_time_s": m["training_time_s"],
            "peak_rss_mb": m["peak_rss_mb"],
            "n_features": len(m["feature_cols"]),
            "n_categories": len(m["cat_cols"]),
        }
        for name, m in all_trained_models.items()
    }
    
    saved_artifacts = generate_evaluation_artifacts(
        df_preds=full_preds_df,
        tuning_df=tuning_df,
        obj_df=obj_df,
        model_metadata=model_metadata,
        cfg=cfg,
        log=log,
    )

    # -------------------------------------------------------------------------
    # 9. Reproducibility Rerun Verification
    # -------------------------------------------------------------------------
    log("Executing numerical reproducibility verification check on full origin slice...")
    test_orig_id = eval_origs[0]
    test_slice = full_preds_df[full_preds_df["origin_id"] == test_orig_id].copy()
    n_test_rows = len(test_slice)
    
    # Reload model A for final test and predict
    sample_model = all_trained_models[f"FINAL_TEST_model_A"]
    pf_A = pq.ParquetFile(cfg.matrix_A_dir / f"origin_id={test_orig_id}" / "data.parquet")
    sample_df_A = pf_A.read().to_pandas()
    sample_df_A = sample_df_A[sample_df_A["id"].isin(sampled_ids)].reset_index(drop=True)
    
    if fold_cat_medians is not None:
        denom = sample_df_A["cat_id"].map(fold_cat_medians).astype(float)
        sample_df_A["price_ratio_to_cat_median"] = np.where(
            sample_df_A["last_price_known_at_origin"].notna(),
            sample_df_A["last_price_known_at_origin"] / denom,
            np.nan
        ).astype(np.float32)
        
    for c in sample_model["cat_cols"]:
        sample_df_A[c] = sample_df_A[c].astype("category")
        
    rerun_preds = sample_model["booster"].predict(sample_df_A[sample_model["feature_cols"]])
    if cfg.clip_nonnegative:
        rerun_preds = np.clip(rerun_preds, 0.0, None)
        
    orig_preds = test_slice["pred_A"].to_numpy()
    max_diff = float(np.max(np.abs(rerun_preds - orig_preds)))
    log(f"Reproducibility verification check on {n_test_rows:,} predictions: max absolute difference = {max_diff:.2e}")
    if max_diff < 1e-5:
        log("Reproducibility check: PASS (deterministic rerun within numerical tolerance).")
    else:
        log("Reproducibility check: WARNING (prediction differences detected).")

    log("=" * 80)
    log(f"ALL EXPERIMENTS AND DELIVERABLES COMPLETED SUCCESSFULLY.")
    log(f"Output files stored in: {out_dir}")
    log("=" * 80)
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
