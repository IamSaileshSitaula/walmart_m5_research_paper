"""SKU Attention Rule: Development Ranking, Decision Difference, and 2015 Test Evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def evaluate_sku_attention_rule(
    full_preds_df: pd.DataFrame,
    out_dir: Path,
    top_k: int = 50,
    log: callable = print,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calibrate SKU attention ranking on development holiday episodes and evaluate on 2015 test."""
    log("Building and evaluating SKU attention rule (B vs D discrepancy)...")
    
    # 1. Isolate Development Holiday Predictions (DEV_S2012, DEV_S2013, DEV_S2014)
    ep_col = "holiday_episode" if "holiday_episode" in full_preds_df.columns else "event_name"
    dev_mask = (
        full_preds_df["fold_id"].isin(["DEV_S2012", "DEV_S2013", "DEV_S2014"])
        & full_preds_df[ep_col].astype(str).str.contains("Thanksgiving|Christmas", na=False)
    )
    dev_hol_df = full_preds_df[dev_mask].copy()
    
    if len(dev_hol_df) == 0:
        raise ValueError("No development holiday episodes found for SKU attention rule calibration.")
        
    # Calculate row-level absolute forecast difference between Model B (same-day) and Model D (event-window)
    dev_hol_df["abs_diff_fcst"] = np.abs(dev_hol_df["pred_D"] - dev_hol_df["pred_B"])
    
    # Calculate origin-level 28-day cumulative order decision difference: Delta Q = |sum(pred_D) - sum(pred_B)|
    orig_orders = dev_hol_df.groupby(["id", "origin_id"])[["pred_B", "pred_D"]].sum().reset_index()
    orig_orders["decision_diff"] = np.abs(orig_orders["pred_D"] - orig_orders["pred_B"])
    mean_dec_diff = orig_orders.groupby("id")["decision_diff"].mean().rename("mean_decision_diff")
    
    # Aggregate development metrics per series
    dev_sku_stats = dev_hol_df.groupby("id").agg(
        item_id=("item_id", "first"),
        dept_id=("dept_id", "first"),
        cat_id=("cat_id", "first"),
        store_id=("store_id", "first"),
        mean_observed_sales=("target_units_sold", "mean"),
        mean_abs_fcst_diff=("abs_diff_fcst", "mean"),
        max_abs_fcst_diff=("abs_diff_fcst", "max"),
        dev_mae_B=("abs_err_B", "mean"),
        dev_mae_D=("abs_err_D", "mean"),
    ).join(mean_dec_diff)
    
    # Attention Score: 28-day decision discrepancy (order delta) between B and D
    dev_sku_stats["attention_score"] = dev_sku_stats["mean_decision_diff"]
    
    # Rank strictly on historical development evidence
    dev_sku_stats = dev_sku_stats.sort_values("attention_score", ascending=False).reset_index()
    dev_sku_stats["rank"] = np.arange(1, len(dev_sku_stats) + 1)
    dev_sku_stats["in_top_50"] = dev_sku_stats["rank"] <= top_k
    dev_sku_stats["in_top_100"] = dev_sku_stats["rank"] <= 100
    
    # 2. Evaluate on 2015 Final Test Holiday Episodes (Thanksgiving 2015 & Christmas 2015)
    test_mask = (
        (full_preds_df["fold_id"] == "FINAL_TEST")
        & full_preds_df[ep_col].astype(str).str.contains("Thanksgiving|Christmas", na=False)
    )
    test_hol_df = full_preds_df[test_mask].copy()
    
    # Calculate 2015 test accuracy by series
    test_sku_stats = test_hol_df.groupby("id").agg(
        test_obs_sales=("target_units_sold", "mean"),
        test_mae_B=("abs_err_B", "mean"),
        test_mae_D=("abs_err_D", "mean"),
        test_bias_B=("err_B", "mean"),
        test_bias_D=("err_D", "mean"),
    ).reset_index()
    
    test_sku_stats["test_delta_mae"] = test_sku_stats["test_mae_D"] - test_sku_stats["test_mae_B"]
    test_sku_stats["test_pct_improvement"] = np.where(
        test_sku_stats["test_mae_B"] > 0,
        (test_sku_stats["test_mae_B"] - test_sku_stats["test_mae_D"]) / test_sku_stats["test_mae_B"] * 100.0,
        0.0
    )
    test_sku_stats["outcome"] = np.where(
        test_sku_stats["test_delta_mae"] < -1e-5, "IMPROVED",
        np.where(test_sku_stats["test_delta_mae"] > 1e-5, "WORSENED", "TIED")
    )
    
    # Join development ranking with test evaluation
    full_sku_ranked = dev_sku_stats.merge(test_sku_stats, on="id", how="left")
    
    # Save full ranked list (3,050 series)
    ranked_csv = out_dir / "sku_attention_ranked_list.csv"
    full_sku_ranked.to_csv(ranked_csv, index=False)
    
    # Top 50 audit list
    top_50_df = full_sku_ranked[full_sku_ranked["in_top_50"]].copy()
    top_50_csv = out_dir / "sku_attention_2015_performance.csv"
    top_50_df.to_csv(top_50_csv, index=False)
    
    # 3. Aggregate Performance Comparisons on 2015 Test
    top50_mae_B = float(np.mean(test_hol_df[test_hol_df["id"].isin(top_50_df["id"])]["abs_err_B"]))
    top50_mae_D = float(np.mean(test_hol_df[test_hol_df["id"].isin(top_50_df["id"])]["abs_err_D"]))
    top50_delta_pct = (top50_mae_D - top50_mae_B) / top50_mae_B * 100.0
    
    non_top50_ids = set(dev_sku_stats[~dev_sku_stats["in_top_50"]]["id"])
    rest_mae_B = float(np.mean(test_hol_df[test_hol_df["id"].isin(non_top50_ids)]["abs_err_B"]))
    rest_mae_D = float(np.mean(test_hol_df[test_hol_df["id"].isin(non_top50_ids)]["abs_err_D"]))
    rest_delta_pct = (rest_mae_D - rest_mae_B) / rest_mae_B * 100.0
    
    all_mae_B = float(np.mean(test_hol_df["abs_err_B"]))
    all_mae_D = float(np.mean(test_hol_df["abs_err_D"]))
    all_delta_pct = (all_mae_D - all_mae_B) / all_mae_B * 100.0
    
    n_improved = int((top_50_df["outcome"] == "IMPROVED").sum())
    n_worsened = int((top_50_df["outcome"] == "WORSENED").sum())
    n_tied = int((top_50_df["outcome"] == "TIED").sum())
    pct_improved = (n_improved / top_k) * 100.0
    
    # Regressions audit: SKUs where attention made things worse
    worsened_skus = top_50_df[top_50_df["outcome"] == "WORSENED"].sort_values("test_delta_mae", ascending=False)
    
    # Markdown Report Generation
    md_lines = [
        "# SKU Attention Rule: Development Calibration & 2015 Exploratory Evaluation",
        "",
        "> [!IMPORTANT]",
        "> **Governance & Exploratory Status Disclosure**:",
        "> The attention rule ranking was calibrated **strictly using 2012–2014 development holiday episodes** (`DEV_S2012`, `DEV_S2013`, `DEV_S2014`).",
        "> However, because aggregate 2015 test metrics were inspected in prior research rounds, the evaluation of this attention rule on 2015 is designated as **exploratory**, rather than an untouched confirmatory validation.",
        "",
        "## 1. Attention Rule Formulation",
        "- **Operational Context**: Store inventory managers cannot review all 3,050 series in the catalog.",
        "- **Selection Metric**: Cumulative 28-day inventory order decision discrepancy during historical development holidays:",
        "  $$\\text{Decision Diff}_s = \\frac{1}{M_s} \\sum_{O \\in \\text{DevHolidays}} \\left| \\sum_{h=1}^{28} \\hat{y}_{D, s, O, h} - \\sum_{h=1}^{28} \\hat{y}_{B, s, O, h} \\right|$$",
        f"- **Review List Capacity**: Limited to the **Top {top_k} highest-discrepancy SKU–store pairs** (1.6% of the 3,050 series).",
        "",
        "## 2. Overall 2015 Holiday Performance (Thanksgiving 2015 & Christmas 2015)",
        "",
        "| Catalog Segment | Series Count | Model B MAE (Same-Day) | Model D MAE (Event-Window) | Absolute Delta (D - B) | Relative Delta % |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
        f"| **Attention Review List (Top {top_k})** | {top_k} | {top50_mae_B:.4f} | {top50_mae_D:.4f} | {top50_mae_D - top50_mae_B:+.4f} | **{top50_delta_pct:+.2f}%** |",
        f"| **Remaining Catalog (Unreviewed)** | {len(dev_sku_stats) - top_k:,} | {rest_mae_B:.4f} | {rest_mae_D:.4f} | {rest_mae_D - rest_mae_B:+.4f} | {rest_delta_pct:+.2f}% |",
        f"| **All Holiday Series** | {len(dev_sku_stats):,} | {all_mae_B:.4f} | {all_mae_D:.4f} | {all_mae_D - all_mae_B:+.4f} | {all_delta_pct:+.2f}% |",
        "",
        f"- **Attention List Win Rate**: **{n_improved}/{top_k} SKUs ({pct_improved:.1f}%)** showed lower MAE under Model D.",
        f"- **Regressions (Made Worse)**: **{n_worsened}/{top_k} SKUs ({(n_worsened/top_k)*100:.1f}%)** exhibited higher error under Model D.",
        f"- **Neutral / Tied**: **{n_tied}/{top_k} SKUs**.",
        "",
        "## 3. Analysis of SKUs Where Attention Made Things Worse (Regressions)",
        "",
        "To ensure managerial transparency, any automated alerting system must expose false alarms and performance degradation:",
        "",
        "| Series ID | Store | Category | Dev Order Delta | Test 2015 Obs Mean | MAE Model B | MAE Model D | Error Increase (D - B) | Regression Diagnosis |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|",
    ]
    
    for r in worsened_skus.head(10).itertuples():
        diag = "Overforecasted pre-holiday ramp" if r.test_bias_D > r.test_bias_B else "Underforecasted post-holiday drop"
        md_lines.append(
            f"| `{r.id}` | {r.store_id} | {r.cat_id} | {r.mean_decision_diff:.2f} units | {r.test_obs_sales:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | **{r.test_delta_mae:+.4f}** | {diag} |"
        )
        
    md_lines.extend([
        "",
        "## 4. Top 15 Highest-Priority SKUs on the Attention List",
        "",
        "| Rank | Series ID | Store | Category | Dev Decision Delta | 2015 MAE B | 2015 MAE D | 2015 $\\Delta$ (D - B) | Outcome |",
        "|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    
    for r in top_50_df.head(15).itertuples():
        outcome_badge = f"**{r.outcome}**" if r.outcome == "IMPROVED" else f"*{r.outcome}*"
        md_lines.append(
            f"| {r.rank} | `{r.id}` | {r.store_id} | {r.cat_id} | {r.mean_decision_diff:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | {r.test_delta_mae:+.4f} | {outcome_badge} |"
        )
        
    report_text = "\n".join(md_lines)
    report_file = out_dir / "sku_attention_report.md"
    report_file.write_text(report_text, encoding="utf-8")
    
    log(f"SKU attention evaluation complete: Top {top_k} win rate = {pct_improved:.1f}% ({n_improved} improved, {n_worsened} worsened).")
    return full_sku_ranked, top_50_df, report_text
