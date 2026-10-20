"""Configuration module for M5 Direct Multi-Horizon Forecasting and Evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    # Random seed for full reproducibility
    seed: int = 42

    # Root paths
    base_dir: Path = Path(".").resolve()
    features_dir: Path = Path("outputs/m5_features").resolve()
    manifest_dir: Path = Path("outputs/m5_features/manifest").resolve()
    cleaned_panel_path: Path = Path("outputs/m5_cleaning/panel/m5_daily_item_store_clean.parquet").resolve()
    output_dir: Path = Path("outputs/m5_models").resolve()

    # Matrix paths
    matrix_A_dir: Path = field(default_factory=lambda: Path("outputs/m5_features/matrix_A").resolve())
    matrix_B_dir: Path = field(default_factory=lambda: Path("outputs/m5_features/matrix_B").resolve())
    matrix_C_dir: Path = field(default_factory=lambda: Path("outputs/m5_features/matrix_C").resolve())

    # Manifest paths
    split_manifest_path: Path = field(default_factory=lambda: Path("outputs/m5_features/manifest/split_manifest.csv").resolve())
    fold_summary_path: Path = field(default_factory=lambda: Path("outputs/m5_features/manifest/fold_summary.csv").resolve())
    feature_manifest_path: Path = field(default_factory=lambda: Path("outputs/m5_features/manifest/feature_manifest.csv").resolve())
    event_mapping_path: Path = field(default_factory=lambda: Path("outputs/m5_features/manifest/event_date_mapping.csv").resolve())
    leakage_audit_path: Path = field(default_factory=lambda: Path("outputs/m5_features/manifest/leakage_audit.md").resolve())

    # Study Population (Predeclared Reduction for Memory Budget)
    # Systematic 10% stratified sample of items (every 10th sorted item across all 10 stores)
    # Total items: 305 items x 10 stores = 3,050 series.
    # Preserves: all 10 stores, all 7 departments, all 3 categories, all 88 origins, all 28 horizons.
    sample_item_stride: int = 10
    total_catalog_series: int = 30490

    # Forecasting parameters
    horizon: int = 28
    label_col: str = "target_units_sold"
    key_cols: List[str] = field(default_factory=lambda: [
        "id", "origin_id", "origin_d", "origin_date", "h", "target_d", "target_date"
    ])
    static_cats: List[str] = field(default_factory=lambda: [
        "item_id", "dept_id", "cat_id", "store_id", "state_id"
    ])

    # Objective selection: compare regression_l1 vs poisson on DEV_S2012
    candidate_objectives: List[str] = field(default_factory=lambda: ["regression_l1", "poisson"])
    primary_metric: str = "mae"

    # Hyperparameter Grid for DEV tuning
    hyperparam_grid: List[Dict] = field(default_factory=lambda: [
        {"learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        {"learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "reg_alpha": 0.5, "reg_lambda": 1.0},
        {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "reg_alpha": 0.5, "reg_lambda": 1.0},
        {"learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 1.0, "bagging_fraction": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0},
        {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 1.0, "bagging_fraction": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0},
    ])

    # Training limits
    max_boost_rounds: int = 150
    early_stopping_rounds: int = 15

    # Post-processing
    clip_nonnegative: bool = True
