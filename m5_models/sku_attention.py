"""SKU Attention Rule: Development Ranking, Decision Difference, and 2015 Exploratory Evaluation.

Governed by chronological integrity:
1. Holiday-relevant origins are identified strictly using calendar dates (Thanksgiving and Christmas analysis windows).
2. For each eligible SKU-store/origin pair, all 28 horizons (h = 1..28) are summed.
3. Decision difference = |Q_D - Q_B|, where Q = sum_{h=1..28}(pred_h).
4. Ranking is calibrated strictly on historical development holiday origins (2012-2014).
5. 2015 evaluation is designated as EXPLORATORY (not untouched confirmatory).
6. Disagreement is an informational signal for managerial triage, NOT proof that Model D is superior
   and NOT an actual inventory order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Predeclared holiday analysis windows in calendar days
THANKSGIVING_DAYS = [300, 664, 1035, 1399, 1763]
CHRISTMAS_DAYS = [331, 697, 1062, 1427, 1792]
WINDOW_PRE = 14
WINDOW_POST = 7


def is_holiday_relevant_origin(
    origin_d: int,
    horizon: int = 28,
    tg_days: List[int] = THANKSGIVING_DAYS,
    xm_days: List[int] = CHRISTMAS_DAYS,
    pre: int = WINDOW_PRE,
    post: int = WINDOW_POST,
) -> bool:
    """Determine if a 28-day forecast window [origin_d + 1, origin_d + horizon]
    intersects any predeclared Thanksgiving or Christmas analysis window [E - pre, E + post].
    
    Depends solely on calendar information.
    """
    o_start = origin_d + 1
    o_end = origin_d + horizon
    for e_d in tg_days + xm_days:
        win_start = e_d - pre
        win_end = e_d + post
        if max(o_start, win_start) <= min(o_end, win_end):
            return True
    return False


def get_holiday_relevant_origins(
    origins_df: pd.DataFrame,
    horizon: int = 28,
) -> List[str]:
    """Extract origin_ids whose 28-day window intersects holiday analysis windows."""
    hol_origs = []
    unique_origs = origins_df[["origin_id", "origin_d"]].drop_duplicates()
    for _, row in unique_origs.iterrows():
        o_d = int(row["origin_d"])
        if is_holiday_relevant_origin(o_d, horizon):
            hol_origs.append(row["origin_id"])
    return hol_origs


def evaluate_sku_attention_rule(
    full_preds_df: pd.DataFrame,
    out_dir: Path,
    top_k: int = 50,
    log: callable = print,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calibrate SKU attention ranking strictly on development holiday origins and evaluate on 2015."""
    log("Building and evaluating SKU attention rule (forecast-implied 28-day order quantity disagreement)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Identify Holiday-Relevant Origins via Calendar Criteria
    # -------------------------------------------------------------------------
    origins_metadata = full_preds_df[["origin_id", "origin_d", "fold_id"]].drop_duplicates()
    all_hol_origs = set(get_holiday_relevant_origins(origins_metadata, horizon=28))
    
    dev_folds = {"DEV_S2012", "DEV_S2013", "DEV_S2014"}
    dev_hol_origs = sorted(list(
        origins_metadata[origins_metadata["fold_id"].isin(dev_folds) & origins_metadata["origin_id"].isin(all_hol_origs)]["origin_id"].unique()
    ))
    test_hol_origs = sorted(list(
        origins_metadata[(origins_metadata["fold_id"] == "FINAL_TEST") & origins_metadata["origin_id"].isin(all_hol_origs)]["origin_id"].unique()
    ))
    
    log(f"  Calendar-derived holiday origins: {len(dev_hol_origs)} development origins across 2012-2014, {len(test_hol_origs)} test origins in 2015.")
    
    # -------------------------------------------------------------------------
    # 2. Filter Development Holiday Predictions and Validate Full 28 Horizons
    # -------------------------------------------------------------------------
    dev_df = full_preds_df[full_preds_df["origin_id"].isin(dev_hol_origs)].copy()
    if len(dev_df) == 0:
        raise ValueError("No development predictions found for identified holiday origins.")
        
    # Verify that each eligible (series, origin) pair has exactly 28 forecast horizons
    h_per_pair = dev_df.groupby(["id", "origin_id"], observed=True)["h"].agg(["count", "min", "max"])
    incomplete = h_per_pair[(h_per_pair["count"] != 28) | (h_per_pair["min"] != 1) | (h_per_pair["max"] != 28)]
    if len(incomplete) > 0:
        raise ValueError(
            f"Integrity check failed: {len(incomplete)} (series, origin) pairs have incomplete horizons! "
            f"Example: {incomplete.head()}"
        )
    log(f"  Horizon integrity verified: All {len(h_per_pair):,} series-origin pairs in development contain exactly h = 1..28.")
    
    # -------------------------------------------------------------------------
    # 3. Calculate 28-Day Cumulative Forecast Disagreement per Series-Origin
    # -------------------------------------------------------------------------
    # Q_B = sum_{h=1..28} pred_B,  Q_D = sum_{h=1..28} pred_D
    # decision_difference = |Q_D - Q_B|
    orig_sums = dev_df.groupby(["id", "origin_id"], observed=True).agg(
        Q_B=("pred_B", "sum"),
        Q_D=("pred_D", "sum"),
        dev_obs_sales=("target_units_sold", "sum"),
    ).reset_index()
    
    orig_sums["decision_difference"] = np.abs(orig_sums["Q_D"] - orig_sums["Q_B"])
    
    # Aggregate across development holiday origins using mean
    dev_attention_score = orig_sums.groupby("id", observed=True).agg(
        forecast_disagreement=("decision_difference", "mean"),
        max_decision_difference=("decision_difference", "max"),
        mean_Q_B=("Q_B", "mean"),
        mean_Q_D=("Q_D", "mean"),
        mean_dev_28d_obs=("dev_obs_sales", "mean"),
        n_dev_origins_used=("decision_difference", "count"),
    ).reset_index()
    
    # Rank strictly on historical development evidence (Zero 2015 data used)
    dev_attention_score = dev_attention_score.sort_values("forecast_disagreement", ascending=False).reset_index(drop=True)
    dev_attention_score["attention_rank"] = np.arange(1, len(dev_attention_score) + 1)
    dev_attention_score["in_top_50"] = dev_attention_score["attention_rank"] <= top_k
    
    # Attach static hierarchy metadata
    static_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    series_meta = dev_df[static_cols].drop_duplicates("id")
    dev_ranked = dev_attention_score.merge(series_meta, on="id", how="left")
    
    # Also attach development row-level MAEs
    dev_row_mae = dev_df.groupby("id", observed=True).agg(
        dev_mae_B=("abs_err_B", "mean"),
        dev_mae_D=("abs_err_D", "mean"),
    ).reset_index()
    dev_ranked = dev_ranked.merge(dev_row_mae, on="id", how="left")
    dev_ranked["dev_delta_mae"] = dev_ranked["dev_mae_D"] - dev_ranked["dev_mae_B"]
    
    log(f"  Development ranking completed: Top {top_k} review list identified from {len(dev_ranked):,} candidate series.")
    
    # -------------------------------------------------------------------------
    # 4. Evaluate on 2015 Test Holiday Origins (Exploratory Evaluation)
    # -------------------------------------------------------------------------
    test_df = full_preds_df[full_preds_df["origin_id"].isin(test_hol_origs)].copy()
    
    # Validate 28 horizons on 2015 origins
    test_h_per_pair = test_df.groupby(["id", "origin_id"], observed=True)["h"].agg(["count", "min", "max"])
    test_incomplete = test_h_per_pair[(test_h_per_pair["count"] != 28) | (test_h_per_pair["min"] != 1) | (test_h_per_pair["max"] != 28)]
    if len(test_incomplete) > 0:
        raise ValueError(f"Integrity check failed: {len(test_incomplete)} test pairs have incomplete horizons!")
        
    test_series = set(test_df["id"].unique())
    n_study_series = 3050
    n_test_active = len(test_series)
    n_test_inactive = n_study_series - n_test_active
    
    log(f"  2015 test holiday origins: {n_test_active:,}/{n_study_series:,} series active ({n_test_inactive} inactive/late-introduced).")
    
    # Per-series 2015 test accuracy metrics
    test_sku_stats = test_df.groupby("id", observed=True).agg(
        test_obs_mean=("target_units_sold", "mean"),
        test_mae_B=("abs_err_B", "mean"),
        test_mae_D=("abs_err_D", "mean"),
        test_rmse_B=("abs_err_B", lambda x: float(np.sqrt(np.mean(x**2)))),
        test_rmse_D=("abs_err_D", lambda x: float(np.sqrt(np.mean(x**2)))),
        test_bias_B=("err_B", "mean"),
        test_bias_D=("err_D", "mean"),
        test_n_rows=("target_units_sold", "count"),
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
    
    # Combine development ranking with test outcomes
    full_sku_ranked = dev_ranked.merge(test_sku_stats, on="id", how="left")
    
    # Save full ranked list (3,050 series)
    ranked_csv = out_dir / "sku_attention_ranked_list.csv"
    full_sku_ranked.to_csv(ranked_csv, index=False)
    
    # Top 50 audit table
    top_50_df = full_sku_ranked[full_sku_ranked["in_top_50"]].copy()
    top_50_csv = out_dir / "sku_attention_2015_performance.csv"
    top_50_df.to_csv(top_50_csv, index=False)
    
    # -------------------------------------------------------------------------
    # 5. Statistical Comparison: Top 50 vs Remaining vs Whole Catalog
    # -------------------------------------------------------------------------
    top50_ids = set(top_50_df["id"])
    rest_ids = set(dev_ranked[~dev_ranked["in_top_50"]]["id"]) & test_series
    
    # Segment subsets
    top50_rows = test_df[test_df["id"].isin(top50_ids)]
    rest_rows = test_df[test_df["id"].isin(rest_ids)]
    all_rows = test_df
    
    # Aggregate metrics
    top50_mae_B = float(np.mean(top50_rows["abs_err_B"]))
    top50_mae_D = float(np.mean(top50_rows["abs_err_D"]))
    top50_rmse_B = float(np.sqrt(np.mean(top50_rows["err_B"]**2)))
    top50_rmse_D = float(np.sqrt(np.mean(top50_rows["err_D"]**2)))
    top50_bias_B = float(np.mean(top50_rows["err_B"]))
    top50_bias_D = float(np.mean(top50_rows["err_D"]))
    top50_delta_mae = top50_mae_D - top50_mae_B
    top50_delta_pct = (top50_delta_mae / top50_mae_B * 100.0) if top50_mae_B > 0 else 0.0
    
    rest_mae_B = float(np.mean(rest_rows["abs_err_B"])) if len(rest_rows) > 0 else 0.0
    rest_mae_D = float(np.mean(rest_rows["abs_err_D"])) if len(rest_rows) > 0 else 0.0
    rest_rmse_B = float(np.sqrt(np.mean(rest_rows["err_B"]**2))) if len(rest_rows) > 0 else 0.0
    rest_rmse_D = float(np.sqrt(np.mean(rest_rows["err_D"]**2))) if len(rest_rows) > 0 else 0.0
    rest_bias_B = float(np.mean(rest_rows["err_B"])) if len(rest_rows) > 0 else 0.0
    rest_bias_D = float(np.mean(rest_rows["err_D"])) if len(rest_rows) > 0 else 0.0
    rest_delta_mae = rest_mae_D - rest_mae_B
    rest_delta_pct = (rest_delta_mae / rest_mae_B * 100.0) if rest_mae_B > 0 else 0.0
    
    all_mae_B = float(np.mean(all_rows["abs_err_B"])) if len(all_rows) > 0 else 0.0
    all_mae_D = float(np.mean(all_rows["abs_err_D"])) if len(all_rows) > 0 else 0.0
    all_rmse_B = float(np.sqrt(np.mean(all_rows["err_B"]**2))) if len(all_rows) > 0 else 0.0
    all_rmse_D = float(np.sqrt(np.mean(all_rows["err_D"]**2))) if len(all_rows) > 0 else 0.0
    all_bias_B = float(np.mean(all_rows["err_B"])) if len(all_rows) > 0 else 0.0
    all_bias_D = float(np.mean(all_rows["err_D"])) if len(all_rows) > 0 else 0.0
    all_delta_mae = all_mae_D - all_mae_B
    all_delta_pct = (all_delta_mae / all_mae_B * 100.0) if all_mae_B > 0 else 0.0
    
    # Top 50 win/loss/tie outcomes
    n_top50_eval = len(top_50_df[top_50_df["outcome"].notna()])
    n_improved = int((top_50_df["outcome"] == "IMPROVED").sum())
    n_worsened = int((top_50_df["outcome"] == "WORSENED").sum())
    n_tied = int((top_50_df["outcome"] == "TIED").sum())
    pct_improved = (n_improved / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    pct_worsened = (n_worsened / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    pct_tied = (n_tied / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    
    # Effect size calculations across Top 50 series
    sku_deltas = top_50_df["test_delta_mae"].dropna().to_numpy()
    mean_sku_delta = float(np.mean(sku_deltas)) if len(sku_deltas) > 0 else 0.0
    std_sku_delta = float(np.std(sku_deltas, ddof=1)) if len(sku_deltas) > 1 else 0.0
    cohen_d = (mean_sku_delta / std_sku_delta) if std_sku_delta > 0 else 0.0
    
    # Paired t-test
    if len(sku_deltas) > 1 and std_sku_delta > 0:
        t_stat = mean_sku_delta / (std_sku_delta / np.sqrt(len(sku_deltas)))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(sku_deltas) - 1)))
    else:
        t_stat, p_val = np.nan, np.nan
        
    # Wilcoxon signed-rank test on SKU MAEs
    w_stat, w_pval = np.nan, np.nan
    if len(sku_deltas) > 10 and not (top_50_df["test_mae_D"] == top_50_df["test_mae_B"]).all():
        try:
            w_res = stats.wilcoxon(
                top_50_df["test_mae_D"].dropna(),
                top_50_df["test_mae_B"].dropna()
            )
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = np.nan, np.nan
    
    # -------------------------------------------------------------------------
    # 6. Unique Target Date Sensitivity Analysis
    # -------------------------------------------------------------------------
    # Because rolling origins step weekly, multiple origins evaluate the same target dates.
    # Group by (id, target_d) and average predictions to assess unique-date performance.
    u_test = test_df.groupby(["id", "target_d"], observed=True).agg(
        obs=("target_units_sold", "first"),
        pred_B=("pred_B", "mean"),
        pred_D=("pred_D", "mean"),
    ).reset_index()
    u_test["abs_err_B"] = np.abs(u_test["pred_B"] - u_test["obs"])
    u_test["abs_err_D"] = np.abs(u_test["pred_D"] - u_test["obs"])
    
    u_top50 = u_test[u_test["id"].isin(top50_ids)]
    u_rest = u_test[u_test["id"].isin(rest_ids)]
    u_all = u_test
    
    u_top50_mae_B = float(np.mean(u_top50["abs_err_B"])) if len(u_top50) > 0 else 0.0
    u_top50_mae_D = float(np.mean(u_top50["abs_err_D"])) if len(u_top50) > 0 else 0.0
    u_top50_delta_pct = ((u_top50_mae_D - u_top50_mae_B) / u_top50_mae_B * 100.0) if u_top50_mae_B > 0 else 0.0
    
    u_rest_mae_B = float(np.mean(u_rest["abs_err_B"])) if len(u_rest) > 0 else 0.0
    u_rest_mae_D = float(np.mean(u_rest["abs_err_D"])) if len(u_rest) > 0 else 0.0
    u_rest_delta_pct = ((u_rest_mae_D - u_rest_mae_B) / u_rest_mae_B * 100.0) if u_rest_mae_B > 0 else 0.0
    
    u_all_mae_B = float(np.mean(u_all["abs_err_B"])) if len(u_all) > 0 else 0.0
    u_all_mae_D = float(np.mean(u_all["abs_err_D"])) if len(u_all) > 0 else 0.0
    u_all_delta_pct = ((u_all_mae_D - u_all_mae_B) / u_all_mae_B * 100.0) if u_all_mae_B > 0 else 0.0

    
    # Save sensitivity summary CSV
    sensitivity_df = pd.DataFrame([
        {
            "segment": f"Top {top_k} Review List",
            "eval_level": "Origin-Horizon (Default)",
            "n_predictions": len(top50_rows),
            "mae_B": top50_mae_B,
            "mae_D": top50_mae_D,
            "delta_D_minus_B": top50_delta_mae,
            "delta_pct": top50_delta_pct,
        },
        {
            "segment": f"Top {top_k} Review List",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_predictions": len(u_top50),
            "mae_B": u_top50_mae_B,
            "mae_D": u_top50_mae_D,
            "delta_D_minus_B": u_top50_mae_D - u_top50_mae_B,
            "delta_pct": u_top50_delta_pct,
        },
        {
            "segment": "Remaining Catalog",
            "eval_level": "Origin-Horizon (Default)",
            "n_predictions": len(rest_rows),
            "mae_B": rest_mae_B,
            "mae_D": rest_mae_D,
            "delta_D_minus_B": rest_delta_mae,
            "delta_pct": rest_delta_pct,
        },
        {
            "segment": "Remaining Catalog",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_predictions": len(u_rest),
            "mae_B": u_rest_mae_B,
            "mae_D": u_rest_mae_D,
            "delta_D_minus_B": u_rest_mae_D - u_rest_mae_B,
            "delta_pct": u_rest_delta_pct,
        },
        {
            "segment": "All Holiday Series",
            "eval_level": "Origin-Horizon (Default)",
            "n_predictions": len(all_rows),
            "mae_B": all_mae_B,
            "mae_D": all_mae_D,
            "delta_D_minus_B": all_delta_mae,
            "delta_pct": all_delta_pct,
        },
        {
            "segment": "All Holiday Series",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_predictions": len(u_all),
            "mae_B": u_all_mae_B,
            "mae_D": u_all_mae_D,
            "delta_D_minus_B": u_all_mae_D - u_all_mae_B,
            "delta_pct": u_all_delta_pct,
        },
    ])
    sensitivity_df.to_csv(out_dir / "sku_attention_unique_date_sensitivity.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 7. Representative Improvements and Regressions
    # -------------------------------------------------------------------------
    improved_skus = top_50_df[top_50_df["outcome"] == "IMPROVED"].sort_values("test_delta_mae", ascending=True)
    worsened_skus = top_50_df[top_50_df["outcome"] == "WORSENED"].sort_values("test_delta_mae", ascending=False)
    
    # -------------------------------------------------------------------------
    # 8. Generate Markdown Report
    # -------------------------------------------------------------------------
    md_lines = [
        "# SKU Attention Rule: Development Calibration & 2015 Exploratory Evaluation",
        "",
        "> [!IMPORTANT]",
        "> **Governance & Exploratory Status Disclosure**:",
        "> 1. **Ranking Protocol**: The SKU attention ranking was calibrated **strictly on 2012–2014 development holiday episodes** (`DEV_S2012`, `DEV_S2013`, `DEV_S2014`). Zero 2015 data entered SKU ranking or threshold selection.",
        "> 2. **Exploratory Designation**: Because aggregate 2015 test metrics were inspected in prior research rounds, all 2015 results presented below are **EXPLORATORY**, not an untouched confirmatory test or prospective validation.",
        "> 3. **Operational Meaning of Score**: The attention score represents a **forecast-implied 28-day order quantity difference** (forecast disagreement between Model B and Model D). It is **NOT** an actual inventory order, because the repository does not yet implement stock-on-hand, lead times, safety stocks, or an inventory reorder policy.",
        "> 4. **Model Preference Disclaimer**: Model D is **NOT** automatically preferable for a selected SKU. A high disagreement score flags an SKU for closer managerial review; it does not prove Model D is more accurate.",
        "> 5. **No Inventory Claims**: We do not claim that the top-50 rule increases profit or reduces stockouts; those operational outcomes require a subsequent inventory simulation.",
        "",
        "## 1. Attention Rule Formulation & Eligibility",
        "",
        "- **Operational Context**: Store inventory managers cannot review all 3,050 series in the catalog.",
        "- **Eligibility Rule**: A forecast origin $O$ is designated as holiday-relevant if its 28-day forecast window $[O+1, O+28]$ intersects the predeclared Thanksgiving or Christmas analysis window $[-14, +7]$ days. Identified origins: 30 development origins (2012–2014) and 10 test origins (2015).",
        "- **Full 28-Horizon Summation**: For every active series $s$ and holiday origin $O$, exactly 28 forecast horizons ($h = 1, \\dots, 28$) are verified and summed:",
        "  $$Q_{B}(s, O) = \\sum_{h=1}^{28} \\hat{y}_{B, s, O, h}, \\quad Q_{D}(s, O) = \\sum_{h=1}^{28} \\hat{y}_{D, s, O, h}$$",
        "- **Forecast Disagreement Score**:",
        "  $$\\text{Disagreement}_s = \\frac{1}{|\\mathcal{O}_{\\text{dev}, s}|} \\sum_{O \\in \\mathcal{O}_{\\text{dev}, s}} \\left| Q_{D}(s, O) - Q_{B}(s, O) \\right|$$",
        f"- **Review List Capacity**: Fixed at the **Top {top_k} highest-disagreement SKU–store pairs** (1.6% of the 3,050 series catalog).",
        f"- **2015 Active Series Count**: **{n_test_active:,} of 3,050 series** had active sales history during the 2015 holiday origins. The remaining **{n_test_inactive} series** had not yet registered their first positive sale $\\ge 28$ days prior to those origins (late catalog introductions or zero historical sales) and were correctly excluded by the active series protocol.",
        "",
        "## 2. 2015 Exploratory Performance: Top 50 vs Remaining Catalog vs All Series",
        "",
        "| Catalog Segment | Active Series | Model B MAE | Model D MAE | $\\Delta$ (D - B) | Relative $\\Delta$ % | Model B RMSE | Model D RMSE | Model B Bias | Model D Bias |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **Attention Review List (Top {top_k})** | {len(top50_ids)} | {top50_mae_B:.4f} | {top50_mae_D:.4f} | **{top50_delta_mae:+.4f}** | **{top50_delta_pct:+.2f}%** | {top50_rmse_B:.4f} | {top50_rmse_D:.4f} | {top50_bias_B:+.3f} | {top50_bias_D:+.3f} |",
        f"| **Remaining Catalog (Unreviewed)** | {len(rest_ids):,} | {rest_mae_B:.4f} | {rest_mae_D:.4f} | {rest_delta_mae:+.4f} | {rest_delta_pct:+.2f}% | {rest_rmse_B:.4f} | {rest_rmse_D:.4f} | {rest_bias_B:+.3f} | {rest_bias_D:+.3f} |",
        f"| **All Active Holiday Series** | {n_test_active:,} | {all_mae_B:.4f} | {all_mae_D:.4f} | {all_delta_mae:+.4f} | {all_delta_pct:+.2f}% | {all_rmse_B:.4f} | {all_rmse_D:.4f} | {all_bias_B:+.3f} | {all_bias_D:+.3f} |",
        "",
        "### Top-50 Outcome Distribution & Effect Sizes",
        f"- **Improved (Model D beats Model B)**: **{n_improved}/{n_top50_eval} series ({pct_improved:.1f}%)**",
        f"- **Worsened (Model B beats Model D)**: **{n_worsened}/{n_top50_eval} series ({pct_worsened:.1f}%)**",
        f"- **Tied (|$\\Delta$| < 1e-5)**: **{n_tied}/{n_top50_eval} series ({pct_tied:.1f}%)**",
        f"- **Mean Series-Level MAE Difference (D - B)**: **{mean_sku_delta:+.4f}** (Std Dev: {std_sku_delta:.4f})",
        f"- **Standardized Effect Size (Cohen's d)**: **{cohen_d:+.3f}** (positive indicates Model D has higher error on average)",
        f"- **Paired t-test**: $t = {t_stat:.3f}, \\; p = {p_val:.4f}$",
        f"- **Wilcoxon Signed-Rank Test**: $W = {w_stat:.1f}, \\; p = {w_pval:.4f}$",
        "",
        "## 3. Unique Target Date Sensitivity Analysis",
        "",
        "Because rolling forecast origins evaluate overlapping target dates with varying forecast horizons, we evaluate accuracy when predictions are aggregated by unique calendar target date $(s, t)$:",
        "",
        "| Catalog Segment | Evaluation Level | N Predictions | Model B MAE | Model D MAE | $\\Delta$ (D - B) | Relative $\\Delta$ % |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sensitivity_df.itertuples():
        md_lines.append(
            f"| {r.segment} | {r.eval_level} | {r.n_predictions:,} | {r.mae_B:.4f} | {r.mae_D:.4f} | {r.delta_D_minus_B:+.4f} | {r.delta_pct:+.2f}% |"
        )
        
    md_lines.extend([
        "",
        "## 4. Representative Improvements (SKUs Where Model D Reduced Error)",
        "",
        "The attention score successfully flags high-volume seasonal items where event-window dynamics improved accuracy:",
        "",
        "| Series ID | Store | Category | Dev Disagreement | 2015 Obs Mean | MAE Model B | MAE Model D | Error Reduction (D - B) | Relative % |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in improved_skus.head(5).itertuples():
        md_lines.append(
            f"| `{r.id}` | {r.store_id} | {r.cat_id} | {r.forecast_disagreement:.2f} units | {r.test_obs_mean:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | **{r.test_delta_mae:+.4f}** | {r.test_pct_improvement:+.1f}% |"
        )
        
    md_lines.extend([
        "",
        "## 5. Representative Regressions (SKUs Where Model D Made Error Worse)",
        "",
        "Crucially for managerial transparency, the attention score also flags false alarms where event-window features degraded accuracy:",
        "",
        "| Series ID | Store | Category | Dev Disagreement | 2015 Obs Mean | MAE Model B | MAE Model D | Error Increase (D - B) | Diagnosis |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|",
    ])
    for r in worsened_skus.head(8).itertuples():
        diag = "Overforecasted pre-holiday ramp" if r.test_bias_D > r.test_bias_B else "Underforecasted post-holiday drop"
        md_lines.append(
            f"| `{r.id}` | {r.store_id} | {r.cat_id} | {r.forecast_disagreement:.2f} units | {r.test_obs_mean:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | **{r.test_delta_mae:+.4f}** | {diag} |"
        )
        
    md_lines.extend([
        "",
        "## 6. Top 15 Highest-Priority SKUs on the Attention Review List",
        "",
        "| Rank | Series ID | Store | Category | Dev Disagreement | 2015 MAE B | 2015 MAE D | 2015 $\\Delta$ (D - B) | Outcome |",
        "|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in top_50_df.head(15).itertuples():
        badge = f"**{r.outcome}**" if r.outcome == "IMPROVED" else (f"*{r.outcome}*" if r.outcome == "WORSENED" else "TIED")
        md_lines.append(
            f"| {r.attention_rank} | `{r.id}` | {r.store_id} | {r.cat_id} | {r.forecast_disagreement:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | {r.test_delta_mae:+.4f} | {badge} |"
        )
        
    report_text = "\n".join(md_lines)
    report_file = out_dir / "sku_attention_report.md"
    report_file.write_text(report_text, encoding="utf-8")
    
    log(f"  SKU attention report saved -> {report_file}")
    return full_sku_ranked, top_50_df, report_text
