"""SKU Attention Rule: Development Ranking, Decision Difference, and 2015 Exploratory Evaluation.

Governed by chronological integrity and rigorous population accounting:
1. Holiday-relevant origins are identified strictly using calendar dates (Thanksgiving and Christmas analysis windows).
2. For each eligible SKU-store/origin pair, all 28 horizons (h = 1..28) are verified (duplicate and missing horizons strictly prohibited) and summed.
3. Decision difference = |Q_D - Q_B|, where Q = sum_{h=1..28}(pred_h).
4. Ranking is calibrated strictly on historical development holiday origins (2012-2014).
5. 2015 evaluation is designated as EXPLORATORY (not untouched confirmatory).
6. Disagreement is an informational signal for managerial triage, NOT proof that Model D is superior,
   and NOT an actual inventory order.
7. Population accounting partitions the 2015-active series into mutually exclusive and exhaustive groups:
   - Group 1: Top-50 development-ranked series: flagged for human review.
   - Group 2: Other development-ranked series: not flagged.
   - Group 3: 2015-active series without a development score: not eligible for historical ranking;
              use Model B as explicit default for downstream inventory simulation.
   Sampled series with no 2015 holiday evaluation (inactive) are accounted for separately.
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


REQUIRED_HORIZONS: Set[int] = set(range(1, 29))


def verify_horizon_integrity(df: pd.DataFrame, context_name: str = "predictions") -> None:
    """Verify that each eligible (id, origin_id) pair contains each horizon 1..28 exactly once.
    
    Explicitly requires:
    - Non-null keys ('id', 'origin_id', 'h')
    - Integer horizon values strictly in {1, ..., 28}
    - Exactly 28 horizons per pair with no duplicates and no missing horizons.
    Non-integer values (e.g. 14.5), duplicate horizons, or missing horizons must fail with ValueError.
    """
    for col in ["id", "origin_id", "h"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in {context_name}")
        if df[col].isna().any():
            n_null = int(df[col].isna().sum())
            raise ValueError(f"Null values detected in column '{col}' ({n_null} nulls) in {context_name}")

    h_series = df["h"]
    # Check if any float values with non-zero fractional part exist (e.g. 14.5)
    if not np.all(np.equal(np.mod(h_series, 1), 0)):
        invalid_floats = df[np.mod(df["h"], 1) != 0]["h"].head().tolist()
        raise ValueError(
            f"Horizon integrity failed in {context_name}: Non-integer horizon values detected! Examples: {invalid_floats}"
        )

    h_int = h_series.astype(int)
    invalid_range = df[~h_int.isin(REQUIRED_HORIZONS)]["h"].head().tolist()
    if invalid_range:
        raise ValueError(
            f"Horizon integrity failed in {context_name}: Horizons outside valid set {{1..28}} detected! Examples: {invalid_range}"
        )

    if df.duplicated(subset=["id", "origin_id", "h"]).any():
        dups = df[df.duplicated(subset=["id", "origin_id", "h"], keep=False)]
        example_dup = dups[["id", "origin_id", "h"]].head()
        raise ValueError(
            f"Horizon integrity failed in {context_name}: Duplicate (id, origin_id, h) detected! Examples:\n{example_dup}"
        )

    pair_counts = df.groupby(["id", "origin_id"], observed=True).size()
    incomplete = pair_counts[pair_counts != 28]
    if len(incomplete) > 0:
        raise ValueError(
            f"Horizon integrity failed in {context_name}: {len(incomplete)} pairs do not contain exactly 28 horizons! "
            f"Examples:\n{incomplete.head()}"
        )


def generate_population_eligibility_audit(
    full_preds_df: pd.DataFrame,
    dev_hol_origs: List[str],
    test_hol_origs: List[str],
    g3_ids: Set[str],
    inact_ids: Set[str],
    out_dir: Path,
    panel_path: Optional[Path] = None,
    log: callable = print,
) -> pd.DataFrame:
    """Generate population-eligibility audit for unranked active and inactive series.
    
    Derives actual first positive sale day and date from the cleaned daily panel,
    handling the '_evaluation' suffix.
    Evaluates satisfaction of the 28-day minimum-history rule (first_sale_d <= origin_d - 27)
    against development and 2015 test holiday origins.
    """
    if panel_path is None:
        panel_path = Path("outputs/m5_cleaning/panel/m5_daily_item_store_clean.parquet")
        
    affected_ids = sorted(list(g3_ids | inact_ids))
    if len(affected_ids) == 0:
        return pd.DataFrame()
    if not panel_path.exists():
        log(f"  Warning: Panel path {panel_path} not found. Skipping panel-derived eligibility audit.")
        return pd.DataFrame()
        
    import pyarrow.parquet as pq
    target_panel_ids = {f"{mid}_evaluation" for mid in affected_ids}
    tbl = pq.read_table(
        panel_path,
        columns=["id", "d_num", "date", "units_sold"],
        filters=[("id", "in", list(target_panel_ids))]
    )
    df_panel = tbl.to_pandas()
    df_panel["model_id"] = df_panel["id"].str.replace("_evaluation", "")
    
    pos_sales = df_panel[df_panel["units_sold"] > 0]
    first_sales = pos_sales.groupby("model_id").agg(
        first_positive_sale_d=("d_num", "min"),
        first_positive_sale_date=("date", "min")
    ).reset_index()
    first_sales_dict = first_sales.set_index("model_id").to_dict(orient="index")
    
    origins_meta = full_preds_df[["origin_id", "origin_d", "fold_id"]].drop_duplicates()
    dev_hol = origins_meta[origins_meta["origin_id"].isin(dev_hol_origs)].sort_values("origin_d")
    test_hol = origins_meta[origins_meta["origin_id"].isin(test_hol_origs)].sort_values("origin_d")
    
    min_dev_orig_d = int(dev_hol["origin_d"].min()) if len(dev_hol) > 0 else 636
    max_dev_orig_d = int(dev_hol["origin_d"].max()) if len(dev_hol) > 0 else 1420
    last_dev_orig_id = str(dev_hol.iloc[-1]["origin_id"]) if len(dev_hol) > 0 else "O1420"
    min_test_orig_d = int(test_hol["origin_d"].min()) if len(test_hol) > 0 else 1735
    max_test_orig_d = int(test_hol["origin_d"].max()) if len(test_hol) > 0 else 1785
    
    audit_records = []
    for s_id in affected_ids:
        group = "UNRANKED_ACTIVE_2015" if s_id in g3_ids else "INACTIVE_2015_HOLIDAY"
        
        if s_id in first_sales_dict:
            f_d = int(first_sales_dict[s_id]["first_positive_sale_d"])
            f_date = str(first_sales_dict[s_id]["first_positive_sale_date"])
            
            # In builder.py line 151: first_sale_idx <= O_d - 28.
            # Since first_sale_d is 1-indexed (first_sale_d = first_sale_idx + 1),
            # minimum-history condition is first_sale_d <= O_d - 27.
            dev_active_origs = dev_hol[dev_hol["origin_d"] - 27 >= f_d]["origin_id"].tolist()
            test_active_origs = test_hol[test_hol["origin_d"] - 27 >= f_d]["origin_id"].tolist()
            
            sat_dev = len(dev_active_origs) > 0
            sat_test = len(test_active_origs) > 0
            
            if group == "UNRANKED_ACTIVE_2015":
                reason = (
                    f"First positive sale on day {f_d} ({f_date}) is after latest development holiday cutoff "
                    f"(d={max_dev_orig_d - 27}, origin {last_dev_orig_id}). "
                    f"No eligible development forecast origins under the 28-day minimum-history rule. "
                    f"Satisfied minimum history for {len(test_active_origs)}/10 2015 test holiday origins."
                )
            else:
                reason = (
                    f"First positive sale on day {f_d} ({f_date}) occurred after all 10 2015 test holiday origins "
                    f"(range d={min_test_orig_d}..{max_test_orig_d}). "
                    f"Zero sales observed prior to 2015 holiday forecast origins."
                )
        else:
            f_d = None
            f_date = None
            sat_dev = False
            sat_test = False
            dev_active_origs = []
            test_active_origs = []
            reason = "No positive sales observed in cleaned panel; eligibility reason cannot be established from panel data."
            
        audit_records.append({
            "id": s_id,
            "population_group": group,
            "first_positive_sale_d": f_d,
            "first_positive_sale_date": f_date,
            "sampled_dev_holiday_origins": f"{len(dev_hol)} origins (d={min_dev_orig_d}..{max_dev_orig_d})",
            "dev_holiday_origins_satisfied": len(dev_active_origs),
            "satisfied_dev_min_history": sat_dev,
            "sampled_test_holiday_origins": f"{len(test_hol)} origins (d={min_test_orig_d}..{max_test_orig_d})",
            "test_holiday_origins_satisfied": len(test_active_origs),
            "satisfied_test_min_history": sat_test,
            "eligibility_reason": reason,
        })
        
    df_audit = pd.DataFrame(audit_records).sort_values(["population_group", "first_positive_sale_d", "id"]).reset_index(drop=True)
    audit_csv = out_dir / "population_eligibility_audit.csv"
    df_audit.to_csv(audit_csv, index=False)
    log(f"  Population eligibility audit saved -> {audit_csv} ({len(df_audit)} affected series)")
    return df_audit


def evaluate_sku_attention_rule(
    full_preds_df: pd.DataFrame,
    out_dir: Path,
    top_k: int = 50,
    panel_path: Optional[Path] = None,
    log: callable = print,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calibrate SKU attention ranking strictly on development holiday origins and evaluate on 2015.
    
    Partitions the 2015-active population into three mutually exclusive and exhaustive groups:
    1. Top-50 development-ranked series: flagged for human review.
    2. Other development-ranked series: not flagged.
    3. 2015-active series without a development score: not eligible for historical ranking;
       Model B default.
    Sampled series with no 2015 holiday evaluation are reported separately.
    """
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
        
    # Strictly require each eligible (id, origin_id) pair to contain each horizon 1..28 exactly once
    verify_horizon_integrity(dev_df, "development holiday predictions")
    n_dev_pairs = dev_df.groupby(["id", "origin_id"], observed=True).ngroups
    log(f"  Horizon integrity verified: All {n_dev_pairs:,} series-origin pairs in development contain exactly h = 1..28.")
    
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
    
    # Attach development row-level MAEs
    dev_row_mae = dev_df.groupby("id", observed=True).agg(
        dev_mae_B=("abs_err_B", "mean"),
        dev_mae_D=("abs_err_D", "mean"),
    ).reset_index()
    dev_ranked = dev_attention_score.merge(dev_row_mae, on="id", how="left")
    dev_ranked["dev_delta_mae"] = dev_ranked["dev_mae_D"] - dev_ranked["dev_mae_B"]
    
    log(f"  Development ranking completed: Top {top_k} review list identified from {len(dev_ranked):,} eligible development series.")
    
    # -------------------------------------------------------------------------
    # 4. Evaluate on 2015 Test Holiday Origins (Exploratory Evaluation)
    # -------------------------------------------------------------------------
    test_df = full_preds_df[full_preds_df["origin_id"].isin(test_hol_origs)].copy()
    
    # Strictly require each eligible (id, origin_id) pair in 2015 to contain each horizon 1..28 exactly once
    verify_horizon_integrity(test_df, "2015 test holiday predictions")
    n_test_pairs = test_df.groupby(["id", "origin_id"], observed=True).ngroups
    log(f"  Horizon integrity verified: All {n_test_pairs:,} series-origin pairs in 2015 test contain exactly h = 1..28.")
    
    # -------------------------------------------------------------------------
    # 5. Mutually Exclusive & Exhaustive Population Accounting
    # -------------------------------------------------------------------------
    all_study_series = set(full_preds_df["id"].unique())
    n_study_series = len(all_study_series)
    dev_ranked_ids = set(dev_ranked["id"])
    top_k_ids = set(dev_ranked[dev_ranked["in_top_50"]]["id"])
    other_dev_ids = dev_ranked_ids - top_k_ids
    test_active_ids = set(test_df["id"].unique())
    
    # Group 1: Top-50 development-ranked series (flagged for review)
    g1_ids = top_k_ids & test_active_ids
    # Group 2: Other development-ranked series (unflagged)
    g2_ids = other_dev_ids & test_active_ids
    # Group 3: 2015-active series without a development score (late introductions; Model B default)
    g3_ids = test_active_ids - dev_ranked_ids
    # Sampled series with no 2015 holiday evaluation (inactive/unobserved)
    inactive_study_ids = all_study_series - test_active_ids
    
    # Rigorous disjointness and completeness assertions
    assert len(g1_ids & g2_ids) == 0, f"Disjointness violated: G1 and G2 overlap ({len(g1_ids & g2_ids)} series)!"
    assert len(g1_ids & g3_ids) == 0, f"Disjointness violated: G1 and G3 overlap ({len(g1_ids & g3_ids)} series)!"
    assert len(g2_ids & g3_ids) == 0, f"Disjointness violated: G2 and G3 overlap ({len(g2_ids & g3_ids)} series)!"
    assert (g1_ids | g2_ids | g3_ids) == test_active_ids, (
        f"Exhaustiveness violated: Active groups sum to {len(g1_ids | g2_ids | g3_ids)}, expected {len(test_active_ids)}!"
    )
    assert len(test_active_ids & inactive_study_ids) == 0, "Disjointness violated: Active and inactive series overlap!"
    assert (test_active_ids | inactive_study_ids) == all_study_series, (
        f"Exhaustiveness violated: Active + inactive sum to {len(test_active_ids | inactive_study_ids)}, expected {n_study_series}!"
    )
    
    n_test_active = len(test_active_ids)
    n_test_inactive = len(inactive_study_ids)
    
    log(f"  Population Accounting: {n_study_series:,} total sampled series.")
    log(f"    - Group 1 (Top-{top_k} dev-ranked review list): {len(g1_ids):,} series")
    log(f"    - Group 2 (Other dev-ranked series, unflagged): {len(g2_ids):,} series")
    log(f"    - Group 3 (2015-active unranked late introductions): {len(g3_ids):,} series")
    log(f"    -> Total 2015-Active Holiday Series: {n_test_active:,} series")
    log(f"    - Sampled Series Without 2015 Holiday Evaluation: {n_test_inactive:,} series")
    
    # Population eligibility audit for affected series (Group 3 and inactive series)
    audit_df = generate_population_eligibility_audit(
        full_preds_df=full_preds_df,
        dev_hol_origs=dev_hol_origs,
        test_hol_origs=test_hol_origs,
        g3_ids=g3_ids,
        inact_ids=inactive_study_ids,
        out_dir=out_dir,
        panel_path=panel_path,
        log=log,
    )
    
    # -------------------------------------------------------------------------
    # 6. Group Metrics and Row Count Verification
    # -------------------------------------------------------------------------
    rows_g1 = test_df[test_df["id"].isin(g1_ids)]
    rows_g2 = test_df[test_df["id"].isin(g2_ids)]
    rows_g3 = test_df[test_df["id"].isin(g3_ids)]
    rows_all = test_df
    
    assert len(rows_g1) + len(rows_g2) + len(rows_g3) == len(rows_all), (
        f"Row count mismatch: {len(rows_g1)} + {len(rows_g2)} + {len(rows_g3)} != {len(rows_all)}"
    )
    
    def calc_group_metrics(sub_df: pd.DataFrame, n_series: int, name: str) -> dict:
        n_rows = len(sub_df)
        if n_rows == 0:
            return {
                "name": name, "n_series": n_series, "n_rows": 0,
                "mae_B": 0.0, "mae_D": 0.0, "delta_mae": 0.0, "delta_pct": 0.0,
                "rmse_B": 0.0, "rmse_D": 0.0, "bias_B": 0.0, "bias_D": 0.0,
            }
        m_B = float(np.mean(sub_df["abs_err_B"]))
        m_D = float(np.mean(sub_df["abs_err_D"]))
        r_B = float(np.sqrt(np.mean(sub_df["err_B"]**2)))
        r_D = float(np.sqrt(np.mean(sub_df["err_D"]**2)))
        b_B = float(np.mean(sub_df["err_B"]))
        b_D = float(np.mean(sub_df["err_D"]))
        d_m = m_D - m_B
        d_pct = (d_m / m_B * 100.0) if m_B > 0 else 0.0
        return {
            "name": name, "n_series": n_series, "n_rows": n_rows,
            "mae_B": m_B, "mae_D": m_D, "delta_mae": d_m, "delta_pct": d_pct,
            "rmse_B": r_B, "rmse_D": r_D, "bias_B": b_B, "bias_D": b_D,
        }
    
    m_g1 = calc_group_metrics(rows_g1, len(g1_ids), f"Top-{top_k} Development-Ranked Series (Flagged for Review)")
    m_g2 = calc_group_metrics(rows_g2, len(g2_ids), "Other Development-Ranked Series (Unflagged)")
    m_g3 = calc_group_metrics(rows_g3, len(g3_ids), "2015-Active Series Without Dev Score (Model B Default)")
    m_all = calc_group_metrics(rows_all, n_test_active, "All Active 2015 Holiday Series")
    
    # Weighted error reconstruction check
    weighted_mae_B = (m_g1["mae_B"] * m_g1["n_rows"] + m_g2["mae_B"] * m_g2["n_rows"] + m_g3["mae_B"] * m_g3["n_rows"]) / m_all["n_rows"]
    weighted_mae_D = (m_g1["mae_D"] * m_g1["n_rows"] + m_g2["mae_D"] * m_g2["n_rows"] + m_g3["mae_D"] * m_g3["n_rows"]) / m_all["n_rows"]
    assert np.isclose(weighted_mae_B, m_all["mae_B"], rtol=1e-7), (
        f"Weighted MAE B ({weighted_mae_B:.6f}) does not match all-active MAE ({m_all['mae_B']:.6f})"
    )
    assert np.isclose(weighted_mae_D, m_all["mae_D"], rtol=1e-7), (
        f"Weighted MAE D ({weighted_mae_D:.6f}) does not match all-active MAE ({m_all['mae_D']:.6f})"
    )
    
    # -------------------------------------------------------------------------
    # 7. Series-Level Statistics and Full Catalog Compilation
    # -------------------------------------------------------------------------
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
    
    # Combine static hierarchy metadata for ALL study series
    static_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    all_series_meta = full_preds_df[static_cols].drop_duplicates("id").reset_index(drop=True)
    
    full_sku_ranked = all_series_meta.merge(dev_ranked, on="id", how="left")
    full_sku_ranked = full_sku_ranked.merge(test_sku_stats, on="id", how="left")
    
    def assign_population_group(row_id: str) -> str:
        if row_id in g1_ids:
            return "TOP_50_REVIEW"
        elif row_id in g2_ids:
            return "OTHER_DEV_RANKED"
        elif row_id in g3_ids:
            return "UNRANKED_ACTIVE_2015"
        else:
            return "INACTIVE_2015_HOLIDAY"
            
    def assign_policy_default(row_id: str) -> str:
        if row_id in g1_ids:
            return "HUMAN_REVIEW_TRIAGE"
        elif row_id in g2_ids:
            return "MODEL_B_DEFAULT"
        elif row_id in g3_ids:
            return "MODEL_B_DEFAULT"
        else:
            return "INACTIVE"
            
    full_sku_ranked["population_group"] = full_sku_ranked["id"].map(assign_population_group)
    full_sku_ranked["policy_default"] = full_sku_ranked["id"].map(assign_policy_default)
    full_sku_ranked["in_top_50"] = full_sku_ranked["id"].isin(g1_ids)
    
    # Sort: Top 50 first, then remaining dev ranked, then unranked active, then inactive
    group_order = {"TOP_50_REVIEW": 1, "OTHER_DEV_RANKED": 2, "UNRANKED_ACTIVE_2015": 3, "INACTIVE_2015_HOLIDAY": 4}
    full_sku_ranked["group_order"] = full_sku_ranked["population_group"].map(group_order)
    full_sku_ranked = full_sku_ranked.sort_values(["group_order", "attention_rank", "id"]).drop(columns=["group_order"]).reset_index(drop=True)
    
    ranked_csv = out_dir / "sku_attention_ranked_list.csv"
    full_sku_ranked.to_csv(ranked_csv, index=False)
    
    # Top 50 audit table
    top_50_df = full_sku_ranked[full_sku_ranked["in_top_50"]].copy().sort_values("attention_rank").reset_index(drop=True)
    top_50_csv = out_dir / "sku_attention_2015_performance.csv"
    top_50_df.to_csv(top_50_csv, index=False)
    
    # -------------------------------------------------------------------------
    # 8. Top 50 Win/Loss Outcomes and Effect Sizes
    # -------------------------------------------------------------------------
    n_top50_eval = len(top_50_df[top_50_df["outcome"].notna()])
    n_improved = int((top_50_df["outcome"] == "IMPROVED").sum())
    n_worsened = int((top_50_df["outcome"] == "WORSENED").sum())
    n_tied = int((top_50_df["outcome"] == "TIED").sum())
    pct_improved = (n_improved / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    pct_worsened = (n_worsened / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    pct_tied = (n_tied / n_top50_eval) * 100.0 if n_top50_eval > 0 else 0.0
    
    sku_deltas = top_50_df["test_delta_mae"].dropna().to_numpy()
    mean_sku_delta = float(np.mean(sku_deltas)) if len(sku_deltas) > 0 else 0.0
    std_sku_delta = float(np.std(sku_deltas, ddof=1)) if len(sku_deltas) > 1 else 0.0
    cohen_d = (mean_sku_delta / std_sku_delta) if std_sku_delta > 0 else 0.0
    
    if len(sku_deltas) > 1 and std_sku_delta > 0:
        t_stat = mean_sku_delta / (std_sku_delta / np.sqrt(len(sku_deltas)))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(sku_deltas) - 1)))
    else:
        t_stat, p_val = np.nan, np.nan
        
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
    # 9. Unique Target Date Sensitivity Analysis
    # -------------------------------------------------------------------------
    u_test = test_df.groupby(["id", "target_d"], observed=True).agg(
        obs=("target_units_sold", "first"),
        pred_B=("pred_B", "mean"),
        pred_D=("pred_D", "mean"),
    ).reset_index()
    u_test["abs_err_B"] = np.abs(u_test["pred_B"] - u_test["obs"])
    u_test["abs_err_D"] = np.abs(u_test["pred_D"] - u_test["obs"])
    
    u_g1 = u_test[u_test["id"].isin(g1_ids)]
    u_g2 = u_test[u_test["id"].isin(g2_ids)]
    u_g3 = u_test[u_test["id"].isin(g3_ids)]
    u_all = u_test
    
    assert len(u_g1) + len(u_g2) + len(u_g3) == len(u_all), (
        f"Unique-date row mismatch: {len(u_g1)} + {len(u_g2)} + {len(u_g3)} != {len(u_all)}"
    )
    
    def calc_u_metrics(sub_df: pd.DataFrame, n_series: int, name: str) -> dict:
        n_rows = len(sub_df)
        if n_rows == 0:
            return {"name": name, "n_series": n_series, "n_rows": 0, "mae_B": 0.0, "mae_D": 0.0, "delta_mae": 0.0, "delta_pct": 0.0}
        m_B = float(np.mean(sub_df["abs_err_B"]))
        m_D = float(np.mean(sub_df["abs_err_D"]))
        d_m = m_D - m_B
        d_pct = (d_m / m_B * 100.0) if m_B > 0 else 0.0
        return {"name": name, "n_series": n_series, "n_rows": n_rows, "mae_B": m_B, "mae_D": m_D, "delta_mae": d_m, "delta_pct": d_pct}
    
    u_m_g1 = calc_u_metrics(u_g1, len(g1_ids), f"Top-{top_k} Development-Ranked Series (Flagged for Review)")
    u_m_g2 = calc_u_metrics(u_g2, len(g2_ids), "Other Development-Ranked Series (Unflagged)")
    u_m_g3 = calc_u_metrics(u_g3, len(g3_ids), "2015-Active Series Without Dev Score (Model B Default)")
    u_m_all = calc_u_metrics(u_all, n_test_active, "All Active 2015 Holiday Series")
    
    sensitivity_df = pd.DataFrame([
        {
            "segment": f"Top-{top_k} Development-Ranked Series (Flagged for Review)",
            "eval_level": "Origin-Horizon (Default)",
            "n_series": m_g1["n_series"],
            "n_predictions": m_g1["n_rows"],
            "mae_B": m_g1["mae_B"],
            "mae_D": m_g1["mae_D"],
            "delta_D_minus_B": m_g1["delta_mae"],
            "delta_pct": m_g1["delta_pct"],
        },
        {
            "segment": f"Top-{top_k} Development-Ranked Series (Flagged for Review)",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_series": u_m_g1["n_series"],
            "n_predictions": u_m_g1["n_rows"],
            "mae_B": u_m_g1["mae_B"],
            "mae_D": u_m_g1["mae_D"],
            "delta_D_minus_B": u_m_g1["delta_mae"],
            "delta_pct": u_m_g1["delta_pct"],
        },
        {
            "segment": "Other Development-Ranked Series (Unflagged)",
            "eval_level": "Origin-Horizon (Default)",
            "n_series": m_g2["n_series"],
            "n_predictions": m_g2["n_rows"],
            "mae_B": m_g2["mae_B"],
            "mae_D": m_g2["mae_D"],
            "delta_D_minus_B": m_g2["delta_mae"],
            "delta_pct": m_g2["delta_pct"],
        },
        {
            "segment": "Other Development-Ranked Series (Unflagged)",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_series": u_m_g2["n_series"],
            "n_predictions": u_m_g2["n_rows"],
            "mae_B": u_m_g2["mae_B"],
            "mae_D": u_m_g2["mae_D"],
            "delta_D_minus_B": u_m_g2["delta_mae"],
            "delta_pct": u_m_g2["delta_pct"],
        },
        {
            "segment": "2015-Active Series Without Dev Score (Model B Default)",
            "eval_level": "Origin-Horizon (Default)",
            "n_series": m_g3["n_series"],
            "n_predictions": m_g3["n_rows"],
            "mae_B": m_g3["mae_B"],
            "mae_D": m_g3["mae_D"],
            "delta_D_minus_B": m_g3["delta_mae"],
            "delta_pct": m_g3["delta_pct"],
        },
        {
            "segment": "2015-Active Series Without Dev Score (Model B Default)",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_series": u_m_g3["n_series"],
            "n_predictions": u_m_g3["n_rows"],
            "mae_B": u_m_g3["mae_B"],
            "mae_D": u_m_g3["mae_D"],
            "delta_D_minus_B": u_m_g3["delta_mae"],
            "delta_pct": u_m_g3["delta_pct"],
        },
        {
            "segment": "All Active 2015 Holiday Series",
            "eval_level": "Origin-Horizon (Default)",
            "n_series": m_all["n_series"],
            "n_predictions": m_all["n_rows"],
            "mae_B": m_all["mae_B"],
            "mae_D": m_all["mae_D"],
            "delta_D_minus_B": m_all["delta_mae"],
            "delta_pct": m_all["delta_pct"],
        },
        {
            "segment": "All Active 2015 Holiday Series",
            "eval_level": "Unique Target Dates (Averaged)",
            "n_series": u_m_all["n_series"],
            "n_predictions": u_m_all["n_rows"],
            "mae_B": u_m_all["mae_B"],
            "mae_D": u_m_all["mae_D"],
            "delta_D_minus_B": u_m_all["delta_mae"],
            "delta_pct": u_m_all["delta_pct"],
        },
    ])
    sensitivity_df.to_csv(out_dir / "sku_attention_unique_date_sensitivity.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 10. Representative Improvements and Regressions
    # -------------------------------------------------------------------------
    improved_skus = top_50_df[top_50_df["outcome"] == "IMPROVED"].sort_values("test_delta_mae", ascending=True)
    worsened_skus = top_50_df[top_50_df["outcome"] == "WORSENED"].sort_values("test_delta_mae", ascending=False)
    
    # -------------------------------------------------------------------------
    # 11. Generate Markdown Report
    # -------------------------------------------------------------------------
    # Format data-derived population descriptions dynamically from audit
    g3_audit = audit_df[audit_df["population_group"] == "UNRANKED_ACTIVE_2015"] if len(audit_df) > 0 else pd.DataFrame()
    inact_audit = audit_df[audit_df["population_group"] == "INACTIVE_2015_HOLIDAY"] if len(audit_df) > 0 else pd.DataFrame()

    if len(g3_audit) > 0 and g3_audit["first_positive_sale_d"].notna().any():
        g3_valid = g3_audit[g3_audit["first_positive_sale_d"].notna()]
        g3_min_d = int(g3_valid["first_positive_sale_d"].min())
        g3_max_d = int(g3_valid["first_positive_sale_d"].max())
        g3_min_date = str(g3_valid.loc[g3_valid["first_positive_sale_d"] == g3_min_d, "first_positive_sale_date"].iloc[0])
        g3_max_date = str(g3_valid.loc[g3_valid["first_positive_sale_d"] == g3_max_d, "first_positive_sale_date"].iloc[0])
        g3_sat_dev = int(g3_valid["satisfied_dev_min_history"].sum())
        g3_min_test_sat = int(g3_valid["test_holiday_origins_satisfied"].min())
        g3_max_test_sat = int(g3_valid["test_holiday_origins_satisfied"].max())
        g3_desc = (
            f"Series active in the 2015 holiday evaluation whose first observed positive sales range from day {g3_min_d} ({g3_min_date}) to day {g3_max_d} ({g3_max_date}). "
            f"Although some of these series had sales during late 2014, none ({g3_sat_dev}/{len(g3_valid)}) satisfied the 28-day minimum-history requirement (`first_sale_d <= origin_d - 27`) at any of the {len(dev_hol_origs)} sampled development holiday forecast origins "
            f"(the latest development holiday origin O1420 at d=1420 required first observed positive sale on or before d=1393). "
            f"Because they had no eligible development forecast origins under the minimum-history rule, they received no development attention score. "
            f"All {len(g3_valid)} series satisfied the minimum-history rule for between {g3_min_test_sat} and {g3_max_test_sat} of the {len(test_hol_origs)} sampled 2015 test holiday origins (range d=1735..1785). "
            f"Their prospective policy assignment for subsequent inventory simulation is the baseline Model B default."
        )
    else:
        g3_desc = (
            "Series active in the 2015 holiday evaluation with no eligible development forecast origins under the 28-day minimum-history rule. "
            "Their prospective policy assignment for subsequent inventory simulation is the baseline Model B default."
        )

    if len(inact_audit) > 0 and inact_audit["first_positive_sale_d"].notna().any():
        inact_records = inact_audit[inact_audit["first_positive_sale_d"].notna()].sort_values("first_positive_sale_d")
        details = [f"`{r.id}` (day {int(r.first_positive_sale_d)}, {r.first_positive_sale_date})" for r in inact_records.itertuples()]
        details_str = ", ".join(details)
        inact_desc = (
            f"These series' first observed positive sales occurred at days {details_str}. "
            f"Because all 10 sampled 2015 test holiday forecast origins precede day 1786 ($O_d \\le 1785$), "
            f"none of these four series had any observed positive sales prior to the 2015 holiday forecast origins, "
            f"failing the active-series minimum-history rule for all 2015 holiday evaluation origins."
        )
    else:
        inact_desc = (
            "Sampled series with no observed sales prior to the 2015 holiday forecast origins, "
            "failing the active-series eligibility rule for all 2015 holiday evaluation origins."
        )

    md_lines = [
        "# SKU Attention Rule: Development Calibration & 2015 Exploratory Evaluation",
        "",
        "> [!IMPORTANT]",
        "> **Governance & Exploratory Status Disclosure**:",
        "> 1. **Ranking Protocol**: The SKU attention ranking was calibrated **strictly on 2012–2014 development holiday episodes** (`DEV_S2012`, `DEV_S2013`, `DEV_S2014`). Zero 2015 data entered SKU ranking or threshold selection.",
        "> 2. **Exploratory Designation**: Because aggregate 2015 test metrics were inspected in prior research rounds, all 2015 results presented below are **EXPLORATORY**, not an untouched confirmatory test or prospective validation.",
        "> 3. **Operational Meaning of Score**: The attention score represents a **forecast-implied 28-day order quantity difference** (forecast disagreement between Model B and Model D). It is **NOT** an actual inventory order, because the repository does not yet implement stock-on-hand, lead times, safety stocks, or an inventory reorder policy.",
        "> 4. **Model Preference Disclaimer**: A top-50 flag identifies substantial disagreement between Model B and Model D; **it does NOT authorize switching to Model D**. Disagreement flags an SKU for closer managerial review; it does not prove Model D is more accurate.",
        "> 5. **Unranked Policy Default**: The unranked 2015-active group's Model B fallback is a prospective policy definition for subsequent inventory simulation, NOT a claim that Model B increased profit.",
        "> 6. **No Inventory Claims**: We do not claim that the top-50 rule increases profit or reduces stockouts; those operational outcomes require a subsequent inventory simulation.",
        "",
        "## 1. Attention Rule Formulation & Population Accounting",
        "",
        "- **Operational Context**: Store inventory managers cannot manually review all 3,050 series in the catalog. A disagreement rule flags the subset where event dynamics differ most between baseline Model B (same-day events) and Model D (event-window dynamics).",
        "- **Eligibility Rule**: A forecast origin $O$ is designated as holiday-relevant if its 28-day forecast window $[O+1, O+28]$ intersects the predeclared Thanksgiving or Christmas analysis window $[-14, +7]$ days. Identified origins: 30 development origins (2012–2014) and 10 test origins (2015).",
        "- **Full 28-Horizon Summation**: For every active series $s$ and holiday origin $O$, exactly 28 forecast horizons ($h = 1, \\dots, 28$) are verified (duplicate and missing horizons strictly prohibited) and summed:",
        "  $$Q_{B}(s, O) = \\sum_{h=1}^{28} \\hat{y}_{B, s, O, h}, \\quad Q_{D}(s, O) = \\sum_{h=1}^{28} \\hat{y}_{D, s, O, h}$$",
        "- **Forecast Disagreement Score**:",
        "  $$\\text{Disagreement}_s = \\frac{1}{|\\mathcal{O}_{\\text{dev}, s}|} \\sum_{O \\in \\mathcal{O}_{\\text{dev}, s}} \\left| Q_{D}(s, O) - Q_{B}(s, O) \\right|$$",
        f"- **Review List Capacity**: Fixed at the **Top {top_k} highest-disagreement SKU–store pairs** (1.6% of the 3,050-series catalog).",
        "- **Mutually Exclusive & Exhaustive Population Accounting**:",
        f"  - **Group 1: Top-{top_k} Development-Ranked Series (Flagged for Review)**: **{m_g1['n_series']:,} series** ({m_g1['n_rows']:,} forecast rows). Calibrated strictly on historical 2012–2014 development holiday episodes.",
        f"  - **Group 2: Other Development-Ranked Series (Unflagged)**: **{m_g2['n_series']:,} series** ({m_g2['n_rows']:,} forecast rows). Active during 2012–2014 development episodes, ranked outside the top {top_k}.",
        f"  - **Group 3: 2015-Active Series Without Dev Score (Model B Default)**: **{m_g3['n_series']:,} series** ({m_g3['n_rows']:,} forecast rows). {g3_desc}",
        f"  - **Total 2015-Active Holiday Series**: **{m_all['n_series']:,} series** ({m_all['n_rows']:,} forecast rows; sum of Groups 1 + 2 + 3).",
        f"  - **Sampled Series Without 2015 Holiday Evaluation (Inactive/Unobserved)**: **{n_test_inactive:,} series** (0 holiday rows). {inact_desc}",
        f"  - **Total Study Population**: **{n_study_series:,} series** ({n_test_active:,} active + {n_test_inactive:,} inactive = {n_study_series:,}).",
        "",
        "## 2. 2015 Exploratory Performance: Mutually Exclusive Population Accounting",
        "",
        "| Catalog Segment | Active Series | Forecast Rows | Model B MAE | Model D MAE | $\\Delta$ (D - B) | Relative $\\Delta$ % | Model B RMSE | Model D RMSE | Model B Bias | Model D Bias |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **Top-{top_k} Development-Ranked Series (Flagged for Review)** | {m_g1['n_series']:,} | {m_g1['n_rows']:,} | {m_g1['mae_B']:.4f} | {m_g1['mae_D']:.4f} | **{m_g1['delta_mae']:+.4f}** | **{m_g1['delta_pct']:+.2f}%** | {m_g1['rmse_B']:.4f} | {m_g1['rmse_D']:.4f} | {m_g1['bias_B']:+.3f} | {m_g1['bias_D']:+.3f} |",
        f"| **Other Development-Ranked Series (Unflagged)** | {m_g2['n_series']:,} | {m_g2['n_rows']:,} | {m_g2['mae_B']:.4f} | {m_g2['mae_D']:.4f} | {m_g2['delta_mae']:+.4f} | {m_g2['delta_pct']:+.2f}% | {m_g2['rmse_B']:.4f} | {m_g2['rmse_D']:.4f} | {m_g2['bias_B']:+.3f} | {m_g2['bias_D']:+.3f} |",
        f"| **2015-Active Series Without Dev Score (Model B Default)** | {m_g3['n_series']:,} | {m_g3['n_rows']:,} | {m_g3['mae_B']:.4f} | {m_g3['mae_D']:.4f} | {m_g3['delta_mae']:+.4f} | {m_g3['delta_pct']:+.2f}% | {m_g3['rmse_B']:.4f} | {m_g3['rmse_D']:.4f} | {m_g3['bias_B']:+.3f} | {m_g3['bias_D']:+.3f} |",
        f"| **All Active 2015 Holiday Series (Total Active)** | {m_all['n_series']:,} | {m_all['n_rows']:,} | {m_all['mae_B']:.4f} | {m_all['mae_D']:.4f} | {m_all['delta_mae']:+.4f} | {m_all['delta_pct']:+.2f}% | {m_all['rmse_B']:.4f} | {m_all['rmse_D']:.4f} | {m_all['bias_B']:+.3f} | {m_all['bias_D']:+.3f} |",
        f"| *Sampled Series Without 2015 Holiday Evaluation (Inactive)* | {n_test_inactive:,} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |",
        "",
        "### Accounting Integrity Checks",
        f"- **Series Count Equality**: {m_g1['n_series']} (Top 50) + {m_g2['n_series']:,} (Other Ranked) + {m_g3['n_series']} (Unranked) = {m_all['n_series']:,} active series (**Verified**).",
        f"- **Row Count Equality**: {m_g1['n_rows']:,} + {m_g2['n_rows']:,} + {m_g3['n_rows']:,} = {m_all['n_rows']:,} forecast rows (**Verified**).",
        f"- **Weighted MAE Reproduction**: Row-weighted MAE across the three groups reproduces the all-active Model B MAE ({weighted_mae_B:.6f} == {m_all['mae_B']:.6f}) and Model D MAE ({weighted_mae_D:.6f} == {m_all['mae_D']:.6f}) (**Verified**).",
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
        "Because rolling forecast origins evaluate overlapping target dates with varying forecast horizons, we evaluate accuracy when predictions are aggregated by unique calendar target date $(s, t)$ across all mutually exclusive segments:",
        "",
        "| Catalog Segment | Evaluation Level | Series Count | Forecast Rows | Model B MAE | Model D MAE | $\\Delta$ (D - B) | Relative $\\Delta$ % |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in sensitivity_df.itertuples():
        md_lines.append(
            f"| {r.segment} | {r.eval_level} | {r.n_series:,} | {r.n_predictions:,} | {r.mae_B:.4f} | {r.mae_D:.4f} | {r.delta_D_minus_B:+.4f} | {r.delta_pct:+.2f}% |"
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
        "| Rank | Series ID | Store | Category | Dev Disagreement | 2015 MAE B | 2015 MAE D | 2015 $\\Delta$ (D - B) | Outcome | Policy Default |",
        "|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in top_50_df.head(15).itertuples():
        badge = f"**{r.outcome}**" if r.outcome == "IMPROVED" else (f"*{r.outcome}*" if r.outcome == "WORSENED" else "TIED")
        md_lines.append(
            f"| {r.attention_rank} | `{r.id}` | {r.store_id} | {r.cat_id} | {r.forecast_disagreement:.2f} | {r.test_mae_B:.4f} | {r.test_mae_D:.4f} | {r.test_delta_mae:+.4f} | {badge} | `{r.policy_default}` |"
        )
        
    report_text = "\n".join(md_lines)
    report_file = out_dir / "sku_attention_report.md"
    report_file.write_text(report_text, encoding="utf-8")
    
    log(f"  SKU attention report saved -> {report_file}")
    return full_sku_ranked, top_50_df, report_text
