"""End-to-end runner for the M5 feature engineering pipeline.

Usage:
    python -m m5_features.run_pipeline \
        --panel-path outputs/m5_cleaning/panel/m5_daily_item_store_clean.parquet \
        --calendar-path outputs/m5_cleaning/features/calendar_event_features.parquet \
        --manifest-path outputs/m5_cleaning/run_manifest.json \
        --out-dir outputs/m5_features
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import run_unit_checks
from .builder import build_feature_datasets
from .config import FEATURE_PIPELINE_VERSION, FeatureConfig
from .inputs import inspect_panel, load_calendar, load_wide, pin_inputs
from .manifests import (
    generate_data_dictionary,
    generate_feature_manifest,
    generate_leakage_audit_report,
    generate_missing_feature_report,
    generate_worked_examples,
)
from .splits import repeated_target_dates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M5 Feature Engineering Pipeline for LightGBM")
    parser.add_argument(
        "--panel-path",
        default="outputs/m5_cleaning/panel/m5_daily_item_store_clean.parquet",
        help="Path to cleaned daily item-store panel",
    )
    parser.add_argument(
        "--calendar-path",
        default="outputs/m5_cleaning/features/calendar_event_features.parquet",
        help="Path to calendar event features",
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/m5_cleaning/run_manifest.json",
        help="Path to cleaning run manifest",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/m5_features",
        help="Output directory for feature matrices and manifests",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "run_log.txt"
    log_fh = open(log_file, "w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"Starting M5 Feature Engineering Pipeline v{FEATURE_PIPELINE_VERSION}")
    log(f"Panel: {args.panel_path}")
    log(f"Output: {out_dir}")

    cfg = FeatureConfig(
        panel_path=Path(args.panel_path).resolve(),
        calendar_features_path=Path(args.calendar_path).resolve(),
        cleaning_manifest_path=Path(args.manifest_path).resolve(),
        out_dir=out_dir,
    )

    # 1. Inspect Panel Schema & Pin Inputs
    log("Inspecting cleaned panel schema and pinning input hashes...")
    panel_info = inspect_panel(cfg.panel_path)
    log(f"Panel rows: {panel_info['num_rows']:,}; date range: {panel_info['date_min']} to {panel_info['date_max']}")

    pin_res = pin_inputs(
        {
            "cleaned_panel": cfg.panel_path,
            "calendar_features": cfg.calendar_features_path,
            "cleaning_manifest": cfg.cleaning_manifest_path,
        },
        manifest_dir / "input_pins.json",
        repin=False,
        log=log,
    )

    # 2. Build Datasets
    res = build_feature_datasets(cfg, log)

    # 3. Generate Manifests & Metadata
    log("Generating manifests and documentation...")
    feat_manifest = generate_feature_manifest(res["cols_A"], res["cols_B"], res["cols_C"])
    feat_manifest.to_csv(manifest_dir / "feature_manifest.csv", index=False)

    # Split Manifest: combine origins and fold assignments
    origins = res["origins"]
    assign = res["assign"]
    split_manifest = assign.merge(origins, on="origin_id")
    split_manifest.to_csv(manifest_dir / "split_manifest.csv", index=False)

    # Fold Summary
    folds = res["folds"]
    folds.to_csv(manifest_dir / "fold_summary.csv", index=False)

    # Repeated Target Dates
    rep_dates = repeated_target_dates(origins, cfg.horizon)
    rep_dates.to_csv(manifest_dir / "repeated_target_dates.csv", index=False)

    # Event Date Mapping
    ev = res["ev"]
    ev.to_csv(manifest_dir / "event_date_mapping.csv", index=False)

    # Marker Inspection
    markers = res["markers"]
    markers.to_csv(manifest_dir / "marker_inspection.csv", index=False)

    # Event Selection Summary
    sel_events = res["sel_events"]
    sel_events.to_csv(manifest_dir / "event_selection_summary.csv", index=False)

    # Event Counts by Fold
    # Cross-tabulate event occurrences in each fold's evaluation period
    event_fold_counts = []
    for f in folds.itertuples():
        ev_sub = ev[(ev.d_num >= f.eval_target_start_d) & (ev.d_num <= f.eval_target_end_d)]
        for ev_name, count in ev_sub["event_name"].value_counts().items():
            event_fold_counts.append({
                "fold_id": f.fold_id,
                "event_name": ev_name,
                "occurrences_in_eval_period": count,
            })
    pd.DataFrame(event_fold_counts).to_csv(manifest_dir / "event_counts_by_fold.csv", index=False)

    # Documentation files
    generate_data_dictionary(feat_manifest, manifest_dir / "data_dictionary.md")
    generate_missing_feature_report(feat_manifest, manifest_dir / "missing_feature_report.md")

    from .splits import check_splits
    test_start = int(res["seasons"].set_index("season").loc[cfg.test_season, "block_start_d"])
    split_chk = check_splits(origins, folds, assign, test_start)
    generate_leakage_audit_report(split_chk, manifest_dir / "leakage_audit.md")

    # Worked Examples around Thanksgiving and Christmas
    wide = load_wide(cfg.panel_path, log=lambda _: None)
    cal = load_calendar(cfg.calendar_features_path)
    generate_worked_examples(origins, cal, wide["Y"], wide["static"], manifest_dir / "worked_examples.md")

    # 4. Unit Checks (i) through (vii)
    checks_df = run_unit_checks(out_dir, res, log)
    checks_df.to_csv(manifest_dir / "unit_checks.csv", index=False)

    # Write summary markdown of checks
    checks_md = [
        "# Unit Checks and Validation Checklist",
        "",
        "| ID | Requirement | Description | Checked | Failures | Status | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in checks_df.itertuples():
        checks_md.append(f"| {r.check_id} | {r.requirement} | {r.description} | {r.n_checked:,} | {r.n_failures} | **{r.status}** | {r.notes} |")
    (manifest_dir / "unit_checks.md").write_text("\n".join(checks_md), encoding="utf-8")

    n_failures = int((checks_df["status"] == "FAIL").sum())
    log(f"Pipeline completed with {n_failures} failures. Outputs saved to {out_dir}")
    log_fh.close()
    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
