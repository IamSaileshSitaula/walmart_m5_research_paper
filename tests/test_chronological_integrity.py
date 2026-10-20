"""Unit and synthetic verification tests for chronological integrity, SKU attention formulation,
population accounting, and production data alignment contracts.
Built using unittest for standard execution without external test dependencies.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from m5_features.config import FeatureConfig
from m5_features.events import (
    PHASE_CATEGORIES,
    compute_fold_event_selection,
    event_date_table,
)
from m5_models.config import ModelConfig
from m5_models.data import (
    compute_fold_category_median_prices,
    load_partition_subset,
)
from m5_models.evaluate import predict_evaluation_origins
from m5_models.sku_attention import (
    evaluate_sku_attention_rule,
    generate_population_eligibility_audit,
    get_holiday_relevant_origins,
    is_holiday_relevant_origin,
    verify_horizon_integrity,
)


def _make_dummy_calendar(n_days: int = 1000) -> pd.DataFrame:
    """Generate minimal calendar DataFrame with Thanksgiving and Christmas markers."""
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    df = pd.DataFrame({
        "d_num": np.arange(1, n_days + 1),
        "d": [f"d_{i}" for i in range(1, n_days + 1)],
        "date": dates,
        "wm_yr_wk": 11101 + (np.arange(n_days) // 7),
        "wday": (np.arange(n_days) % 7) + 1,
        "month": dates.month,
        "year": dates.year,
        "event_name_1": [None] * n_days,
        "event_type_1": [None] * n_days,
        "event_name_2": [None] * n_days,
        "event_type_2": [None] * n_days,
    })
    if n_days >= 300:
        df.loc[df["d_num"] == 300, ["event_name_1", "event_type_1"]] = ["Thanksgiving", "National"]
    if n_days >= 664:
        df.loc[df["d_num"] == 664, ["event_name_1", "event_type_1"]] = ["Thanksgiving", "National"]
    if n_days >= 331:
        df.loc[df["d_num"] == 331, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    if n_days >= 697:
        df.loc[df["d_num"] == 697, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    if n_days >= 10:
        df.loc[df["d_num"] == 10, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    if n_days >= 375:
        df.loc[df["d_num"] == 375, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    if n_days >= 740:
        df.loc[df["d_num"] == 740, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    return df


class TestChronologicalIntegrity(unittest.TestCase):

    def test_post_cutoff_perturbation_invariance(self):
        """Test 1: Changing sales AFTER a fold's training cutoff does NOT change event selection,
        feature schema, or encoded day_C values.
        """
        n_days = 1000
        cutoff_d = 625  # DEV_S2012 cutoff
        cal = _make_dummy_calendar(n_days)
        ev = event_date_table(cal, test_start=800)
        
        cfg = FeatureConfig(
            panel_path=Path("dummy"),
            calendar_features_path=Path("dummy"),
            cleaning_manifest_path=Path("dummy"),
            out_dir=Path("dummy"),
        )
        
        np.random.seed(42)
        store_totals_original = np.random.poisson(lam=100.0, size=(10, n_days)).astype(np.float64)
        
        res_orig = compute_fold_event_selection(
            fold_id="DEV_S2012",
            cutoff_d=cutoff_d,
            store_daily_totals=store_totals_original,
            cal=cal,
            ev=ev,
            cfg=cfg,
        )
        
        store_totals_perturbed = store_totals_original.copy()
        store_totals_perturbed[:, cutoff_d:] = store_totals_perturbed[:, cutoff_d:] * 1000.0 + 99999.0
        
        res_perturbed = compute_fold_event_selection(
            fold_id="DEV_S2012",
            cutoff_d=cutoff_d,
            store_daily_totals=store_totals_perturbed,
            cal=cal,
            ev=ev,
            cfg=cfg,
        )
        
        pd.testing.assert_frame_equal(res_orig["sel_events"], res_perturbed["sel_events"])
        pd.testing.assert_frame_equal(res_orig["effect_profiles"], res_perturbed["effect_profiles"])
        pd.testing.assert_frame_equal(res_orig["day_C"], res_perturbed["day_C"])
        self.assertEqual(res_orig["named_events"], res_perturbed["named_events"])
        self.assertEqual(res_orig["feature_schema"], res_perturbed["feature_schema"])

    def test_sku_attention_28_horizons_and_hand_calculated_example(self):
        """Test 2: Exactly 28 horizons (h=1..28) required per SKU-origin;
        attention score matches hand-calculated arithmetic.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Holiday origin d=636 (origin O0636 intersects Thanksgiving d_664 at leads 1..28)
            self.assertTrue(is_holiday_relevant_origin(636, horizon=28, tg_days=[664], xm_days=[697]))
            
            # Build 28 rows for series s1 and s2 at origin O0636 (DEV_S2012)
            # s1: pred_B = 2.0 each horizon -> Q_B = 56.0
            #     pred_D = 3.5 each horizon -> Q_D = 98.0
            #     decision_diff = |98.0 - 56.0| = 42.0
            # s2: pred_B = 1.0 each horizon -> Q_B = 28.0
            #     pred_D = 1.5 each horizon -> Q_D = 42.0
            #     decision_diff = |42.0 - 28.0| = 14.0
            records = []
            for h in range(1, 29):
                records.append({
                    "id": "FOODS_1_001_CA_1", "item_id": "FOODS_1_001", "dept_id": "FOODS_1",
                    "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": f"2012-10-{h:02d}",
                    "target_units_sold": 2.0, "pred_B": 2.0, "pred_D": 3.5,
                    "abs_err_B": 0.0, "abs_err_D": 1.5, "err_B": 0.0, "err_D": 1.5,
                })
                records.append({
                    "id": "FOODS_1_002_CA_1", "item_id": "FOODS_1_002", "dept_id": "FOODS_1",
                    "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": f"2012-10-{h:02d}",
                    "target_units_sold": 1.0, "pred_B": 1.0, "pred_D": 1.5,
                    "abs_err_B": 0.0, "abs_err_D": 0.5, "err_B": 0.0, "err_D": 0.5,
                })
                
            # Add 2015 test predictions for origin O1735 (h=1..28)
            for h in range(1, 29):
                records.append({
                    "id": "FOODS_1_001_CA_1", "item_id": "FOODS_1_001", "dept_id": "FOODS_1",
                    "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": f"2015-10-{h:02d}",
                    "target_units_sold": 3.0, "pred_B": 2.5, "pred_D": 3.0,
                    "abs_err_B": 0.5, "abs_err_D": 0.0, "err_B": -0.5, "err_D": 0.0,
                })
                records.append({
                    "id": "FOODS_1_002_CA_1", "item_id": "FOODS_1_002", "dept_id": "FOODS_1",
                    "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": f"2015-10-{h:02d}",
                    "target_units_sold": 1.0, "pred_B": 1.0, "pred_D": 2.0,
                    "abs_err_B": 0.0, "abs_err_D": 1.0, "err_B": 0.0, "err_D": 1.0,
                })
                
            df = pd.DataFrame(records)
            
            ranked, top_k_df, _ = evaluate_sku_attention_rule(df, tmp_path, top_k=1, log=lambda m: None)
            
            # Verify hand calculation
            s1_row = ranked[ranked["id"] == "FOODS_1_001_CA_1"].iloc[0]
            s2_row = ranked[ranked["id"] == "FOODS_1_002_CA_1"].iloc[0]
            
            self.assertAlmostEqual(s1_row["forecast_disagreement"], 42.0, places=5)
            self.assertAlmostEqual(s2_row["forecast_disagreement"], 14.0, places=5)
            self.assertEqual(s1_row["attention_rank"], 1)
            self.assertEqual(s2_row["attention_rank"], 2)
            self.assertTrue(bool(s1_row["in_top_50"]))
            self.assertFalse(bool(s2_row["in_top_50"]))
            
            # Verify that dropping a horizon triggers ValueError
            df_incomplete = df.iloc[:-1].copy()
            with self.assertRaises(ValueError):
                evaluate_sku_attention_rule(df_incomplete, tmp_path, top_k=1, log=lambda m: None)

    def test_horizon_integrity_duplicate_and_missing_horizons_fail(self):
        """Test 3: Require each eligible SKU-origin pair to contain each horizon 1 through 28 exactly once;
        a duplicate horizon and a missing horizon must fail.
        """
        # Case A: Valid 28 horizons
        df_valid = pd.DataFrame({
            "id": ["s1"] * 28,
            "origin_id": ["O1"] * 28,
            "h": list(range(1, 29)),
        })
        # Should not raise
        verify_horizon_integrity(df_valid, "valid_test")
        
        # Case B: Duplicate horizon (h=14 appears twice, h=15 is missing; total count = 28, min=1, max=28)
        dup_horizons = list(range(1, 15)) + [14] + list(range(16, 29))
        self.assertEqual(len(dup_horizons), 28)
        self.assertEqual(min(dup_horizons), 1)
        self.assertEqual(max(dup_horizons), 28)
        df_dup = pd.DataFrame({
            "id": ["s1"] * 28,
            "origin_id": ["O1"] * 28,
            "h": dup_horizons,
        })
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_dup, "duplicate_test")
        self.assertIn("Duplicate", str(ctx.exception))
        
        # Case C: Missing horizon (only 27 horizons present)
        df_missing = pd.DataFrame({
            "id": ["s1"] * 27,
            "origin_id": ["O1"] * 27,
            "h": list(range(1, 28)),
        })
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_missing, "missing_test")
        self.assertIn("Horizon integrity failed", str(ctx.exception))

    def test_2015_label_perturbation_invariance(self):
        """Test 4: Changing 2015 observed labels cannot change development attention scores,
        ranking, or Top-50 membership.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            records = []
            for s in range(10):
                s_id = f"ITEM_{s:03d}_STORE_1"
                for h in range(1, 29):
                    records.append({
                        "id": s_id, "item_id": f"ITEM_{s:03d}", "dept_id": "DEPT_1",
                        "cat_id": "CAT_1", "store_id": "STORE_1", "state_id": "CA",
                        "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                        "target_d": 636 + h, "target_date": "2012-10-15",
                        "target_units_sold": float(s), "pred_B": float(s), "pred_D": float(s + (s % 3)),
                        "abs_err_B": 0.0, "abs_err_D": float(s % 3), "err_B": 0.0, "err_D": float(s % 3),
                    })
                    records.append({
                        "id": s_id, "item_id": f"ITEM_{s:03d}", "dept_id": "DEPT_1",
                        "cat_id": "CAT_1", "store_id": "STORE_1", "state_id": "CA",
                        "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                        "target_d": 1735 + h, "target_date": "2015-10-15",
                        "target_units_sold": float(s * 2), "pred_B": float(s), "pred_D": float(s),
                        "abs_err_B": float(s), "abs_err_D": float(s), "err_B": -float(s), "err_D": -float(s),
                    })
                    
            df_orig = pd.DataFrame(records)
            ranked_orig, _, _ = evaluate_sku_attention_rule(df_orig, tmp_path / "run1", top_k=3, log=lambda m: None)
            
            df_perturbed = df_orig.copy()
            test_mask = df_perturbed["fold_id"] == "FINAL_TEST"
            df_perturbed.loc[test_mask, "target_units_sold"] = df_perturbed.loc[test_mask, "target_units_sold"] * 50.0 + 1000.0
            df_perturbed.loc[test_mask, "abs_err_B"] = 999.0
            df_perturbed.loc[test_mask, "abs_err_D"] = 1.0
            
            ranked_pert, _, _ = evaluate_sku_attention_rule(df_perturbed, tmp_path / "run2", top_k=3, log=lambda m: None)
            
            pd.testing.assert_series_equal(ranked_orig["id"], ranked_pert["id"])
            pd.testing.assert_series_equal(ranked_orig["forecast_disagreement"], ranked_pert["forecast_disagreement"])
            pd.testing.assert_series_equal(ranked_orig["attention_rank"], ranked_pert["attention_rank"])
            pd.testing.assert_series_equal(ranked_orig["in_top_50"], ranked_pert["in_top_50"])

    def test_production_bcd_feature_assembly_coordinate_misorder(self):
        """Test 5: Exercise the production B/C/D feature assembly in load_partition_subset
        with deliberately misordered coordinates. Must fail on coordinate mismatch.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mat_B_dir = tmp_path / "matrix_B"
            mat_C_dir = tmp_path / "matrix_C"
            mat_D_dir = tmp_path / "matrix_D"
            (mat_B_dir / "origin_id=O001").mkdir(parents=True)
            (mat_C_dir / "origin_id=O001").mkdir(parents=True)
            
            # Base Matrix B for two series and two horizons
            df_B = pd.DataFrame({
                "id": ["s1", "s1", "s2", "s2"],
                "origin_id": ["O001", "O001", "O001", "O001"],
                "origin_d": [100, 100, 100, 100],
                "origin_date": ["2012-01-01"] * 4,
                "h": [1, 2, 1, 2],
                "target_d": [101, 102, 101, 102],
                "target_date": ["2012-01-02", "2012-01-03", "2012-01-02", "2012-01-03"],
                "target_units_sold": [1.0, 2.0, 3.0, 4.0],
                "cat_id": ["FOODS"] * 4,
                "feat_B": [10, 20, 30, 40],
            })
            pq.write_table(pa.Table.from_pandas(df_B), mat_B_dir / "origin_id=O001" / "data.parquet")
            
            cfg = ModelConfig(matrix_B_dir=mat_B_dir, matrix_C_dir=mat_C_dir)
            sampled = {"s1", "s2"}
            
            # 1. Aligned Matrix C -> Production assembly succeeds
            df_C_aligned = pd.DataFrame({
                "id": ["s1", "s1", "s2", "s2"],
                "h": [1, 2, 1, 2],
                "target_d": [101, 102, 101, 102],
                "ph_christmas": ["outside", "day_0", "outside", "day_0"],
            })
            pq.write_table(pa.Table.from_pandas(df_C_aligned), mat_C_dir / "origin_id=O001" / "data.parquet")
            
            loaded_df, coords_df, feature_cols, _ = load_partition_subset(
                matrix_dir=mat_D_dir,
                origin_ids=["O001"],
                sampled_ids=sampled,
                cfg=cfg,
                matrix_name="model_D",
            )
            self.assertEqual(len(loaded_df), 4)
            self.assertIn("ph_christmas", feature_cols)
            
            # 2. Deliberately misordered ID coordinates in Matrix C -> Production assembly must raise AssertionError
            df_C_bad_id = pd.DataFrame({
                "id": ["s2", "s2", "s1", "s1"],  # Inverted series order!
                "h": [1, 2, 1, 2],
                "target_d": [101, 102, 101, 102],
                "ph_christmas": ["outside", "day_0", "outside", "day_0"],
            })
            pq.write_table(pa.Table.from_pandas(df_C_bad_id), mat_C_dir / "origin_id=O001" / "data.parquet")
            with self.assertRaises(AssertionError) as ctx:
                load_partition_subset(
                    matrix_dir=mat_D_dir,
                    origin_ids=["O001"],
                    sampled_ids=sampled,
                    cfg=cfg,
                    matrix_name="model_D",
                )
            self.assertIn("Coordinate mismatch on 'id'", str(ctx.exception))
            
            # 3. Deliberately misordered Horizon coordinates in Matrix C -> Production assembly must raise AssertionError
            df_C_bad_h = pd.DataFrame({
                "id": ["s1", "s1", "s2", "s2"],
                "h": [2, 1, 2, 1],  # Inverted horizon order!
                "target_d": [101, 102, 101, 102],
                "ph_christmas": ["outside", "day_0", "outside", "day_0"],
            })
            pq.write_table(pa.Table.from_pandas(df_C_bad_h), mat_C_dir / "origin_id=O001" / "data.parquet")
            with self.assertRaises(AssertionError) as ctx:
                load_partition_subset(
                    matrix_dir=mat_D_dir,
                    origin_ids=["O001"],
                    sampled_ids=sampled,
                    cfg=cfg,
                    matrix_name="model_D",
                )
            self.assertIn("Coordinate mismatch on 'h'", str(ctx.exception))
            
            # 4. Deliberately misordered target_d coordinates in Matrix C -> Production assembly must raise AssertionError
            df_C_bad_target = pd.DataFrame({
                "id": ["s1", "s1", "s2", "s2"],
                "h": [1, 2, 1, 2],
                "target_d": [102, 101, 102, 101],  # Inverted target day order!
                "ph_christmas": ["outside", "day_0", "outside", "day_0"],
            })
            pq.write_table(pa.Table.from_pandas(df_C_bad_target), mat_C_dir / "origin_id=O001" / "data.parquet")
            with self.assertRaises(AssertionError) as ctx:
                load_partition_subset(
                    matrix_dir=mat_D_dir,
                    origin_ids=["O001"],
                    sampled_ids=sampled,
                    cfg=cfg,
                    matrix_name="model_D",
                )
            self.assertIn("Coordinate mismatch on 'target_d'", str(ctx.exception))

    def test_population_accounting_disjointness_and_coverage(self):
        """Test 6: Verify the groups are disjoint and cover every 2015-active series exactly once,
        and sampled inactive series are reported separately.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create synthetic data with 4 series:
            # - s1: in dev, will be Top-1 (Group 1)
            # - s2: in dev, will be Rank 2 (Group 2)
            # - s3: NOT in dev, active in 2015 (Group 3: late introduction)
            # - s4: NOT in dev, NOT active in 2015 (Inactive)
            records = []
            
            # Dev holiday origin O0636 (s1, s2 present; s3, s4 absent)
            for h in range(1, 29):
                records.append({
                    "id": "s1", "item_id": "i1", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": "2012-10-15",
                    "target_units_sold": 10.0, "pred_B": 5.0, "pred_D": 10.0,  # diff = 5.0 * 28 = 140
                    "abs_err_B": 5.0, "abs_err_D": 0.0, "err_B": -5.0, "err_D": 0.0,
                })
                records.append({
                    "id": "s2", "item_id": "i2", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": "2012-10-15",
                    "target_units_sold": 2.0, "pred_B": 2.0, "pred_D": 2.5,  # diff = 0.5 * 28 = 14
                    "abs_err_B": 0.0, "abs_err_D": 0.5, "err_B": 0.0, "err_D": 0.5,
                })
                
            # 2015 Test holiday origin O1735 (s1, s2, s3 present; s4 absent)
            for h in range(1, 29):
                records.append({
                    "id": "s1", "item_id": "i1", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 10.0, "pred_B": 6.0, "pred_D": 8.0,
                    "abs_err_B": 4.0, "abs_err_D": 2.0, "err_B": -4.0, "err_D": -2.0,
                })
                records.append({
                    "id": "s2", "item_id": "i2", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 3.0, "pred_B": 2.0, "pred_D": 4.0,
                    "abs_err_B": 1.0, "abs_err_D": 1.0, "err_B": -1.0, "err_D": 1.0,
                })
                records.append({
                    "id": "s3", "item_id": "i3", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 5.0, "pred_B": 3.0, "pred_D": 7.0,
                    "abs_err_B": 2.0, "abs_err_D": 2.0, "err_B": -2.0, "err_D": 2.0,
                })
                
            # s4 is only in an ordinary non-holiday origin O1885
            for h in range(1, 29):
                records.append({
                    "id": "s4", "item_id": "i4", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1885", "origin_d": 1885, "h": h,
                    "target_d": 1885 + h, "target_date": "2016-03-15",
                    "target_units_sold": 1.0, "pred_B": 1.0, "pred_D": 1.0,
                    "abs_err_B": 0.0, "abs_err_D": 0.0, "err_B": 0.0, "err_D": 0.0,
                })
                
            df = pd.DataFrame(records)
            ranked, top_k_df, md_report = evaluate_sku_attention_rule(df, tmp_path, top_k=1, log=lambda m: None)
            
            # Verify 4 series in ranked table
            self.assertEqual(len(ranked), 4)
            
            # Group assignments
            g1 = set(ranked[ranked["population_group"] == "TOP_50_REVIEW"]["id"])
            g2 = set(ranked[ranked["population_group"] == "OTHER_DEV_RANKED"]["id"])
            g3 = set(ranked[ranked["population_group"] == "UNRANKED_ACTIVE_2015"]["id"])
            ginact = set(ranked[ranked["population_group"] == "INACTIVE_2015_HOLIDAY"]["id"])
            
            self.assertEqual(g1, {"s1"})
            self.assertEqual(g2, {"s2"})
            self.assertEqual(g3, {"s3"})
            self.assertEqual(ginact, {"s4"})
            
            # Disjointness
            self.assertEqual(len(g1 & g2), 0)
            self.assertEqual(len(g1 & g3), 0)
            self.assertEqual(len(g2 & g3), 0)
            self.assertEqual(len(g1 & ginact), 0)
            self.assertEqual(len(g2 & ginact), 0)
            self.assertEqual(len(g3 & ginact), 0)
            
            # Exhaustiveness
            self.assertEqual(g1 | g2 | g3, {"s1", "s2", "s3"})
            self.assertEqual(g1 | g2 | g3 | ginact, {"s1", "s2", "s3", "s4"})
            
            # Policy defaults
            self.assertEqual(ranked.loc[ranked["id"] == "s1", "policy_default"].iloc[0], "HUMAN_REVIEW_TRIAGE")
            self.assertEqual(ranked.loc[ranked["id"] == "s2", "policy_default"].iloc[0], "MODEL_B_DEFAULT")
            self.assertEqual(ranked.loc[ranked["id"] == "s3", "policy_default"].iloc[0], "MODEL_B_DEFAULT")
            self.assertEqual(ranked.loc[ranked["id"] == "s4", "policy_default"].iloc[0], "INACTIVE")

    def test_group_row_counts_and_weighted_mae_reproduction(self):
        """Test 7: Verify group row counts sum to all-active row count and their weighted errors
        reproduce all-active MAE.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            records = []
            # Dev holiday origin O0636
            for h in range(1, 29):
                records.append({
                    "id": "s1", "item_id": "i1", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": "2012-10-15",
                    "target_units_sold": 5.0, "pred_B": 1.0, "pred_D": 5.0,
                    "abs_err_B": 4.0, "abs_err_D": 0.0, "err_B": -4.0, "err_D": 0.0,
                })
                records.append({
                    "id": "s2", "item_id": "i2", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                    "target_d": 636 + h, "target_date": "2012-10-15",
                    "target_units_sold": 2.0, "pred_B": 2.0, "pred_D": 2.1,
                    "abs_err_B": 0.0, "abs_err_D": 0.1, "err_B": 0.0, "err_D": 0.1,
                })
                
            # 2015 Test holiday origin O1735 (s1, s2, and unranked s3)
            for h in range(1, 29):
                records.append({
                    "id": "s1", "item_id": "i1", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 8.0, "pred_B": 4.0, "pred_D": 6.0,
                    "abs_err_B": 4.0, "abs_err_D": 2.0, "err_B": -4.0, "err_D": -2.0,
                })
                records.append({
                    "id": "s2", "item_id": "i2", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 3.0, "pred_B": 2.0, "pred_D": 2.5,
                    "abs_err_B": 1.0, "abs_err_D": 0.5, "err_B": -1.0, "err_D": -0.5,
                })
                records.append({
                    "id": "s3", "item_id": "i3", "dept_id": "d1", "cat_id": "c1", "store_id": "st1", "state_id": "CA",
                    "fold_id": "FINAL_TEST", "origin_id": "O1735", "origin_d": 1735, "h": h,
                    "target_d": 1735 + h, "target_date": "2015-10-15",
                    "target_units_sold": 6.0, "pred_B": 3.0, "pred_D": 9.0,
                    "abs_err_B": 3.0, "abs_err_D": 3.0, "err_B": -3.0, "err_D": 3.0,
                })
                
            df = pd.DataFrame(records)
            evaluate_sku_attention_rule(df, tmp_path, top_k=1, log=lambda m: None)
            
            # Read sensitivity table
            sens_df = pd.read_csv(tmp_path / "sku_attention_unique_date_sensitivity.csv")
            default_sens = sens_df[sens_df["eval_level"] == "Origin-Horizon (Default)"]
            
            g1_row = default_sens[default_sens["segment"].str.contains("Top-1")].iloc[0]
            g2_row = default_sens[default_sens["segment"].str.contains("Other Development")].iloc[0]
            g3_row = default_sens[default_sens["segment"].str.contains("2015-Active Series")].iloc[0]
            all_row = default_sens[default_sens["segment"].str.contains("All Active")].iloc[0]
            
            # Row sum verification
            self.assertEqual(g1_row["n_predictions"] + g2_row["n_predictions"] + g3_row["n_predictions"], all_row["n_predictions"])
            self.assertEqual(all_row["n_predictions"], 84)
            
            # Series sum verification
            self.assertEqual(g1_row["n_series"] + g2_row["n_series"] + g3_row["n_series"], all_row["n_series"])
            self.assertEqual(all_row["n_series"], 3)
            
            # Weighted MAE verification
            expected_B = (g1_row["mae_B"] * g1_row["n_predictions"] + g2_row["mae_B"] * g2_row["n_predictions"] + g3_row["mae_B"] * g3_row["n_predictions"]) / all_row["n_predictions"]
            expected_D = (g1_row["mae_D"] * g1_row["n_predictions"] + g2_row["mae_D"] * g2_row["n_predictions"] + g3_row["mae_D"] * g3_row["n_predictions"]) / all_row["n_predictions"]
            self.assertAlmostEqual(all_row["mae_B"], expected_B, places=6)
            self.assertAlmostEqual(all_row["mae_D"], expected_D, places=6)

    def test_training_price_normalization_zero_lookahead(self):
        """Test 8: Category price median is computed strictly from training origins
        and applied to validation origins without look-ahead or contamination from future prices.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mat_A = tmp_path / "matrix_A"
            (mat_A / "origin_id=O001").mkdir(parents=True)
            (mat_A / "origin_id=O002").mkdir(parents=True)

            df_train = pd.DataFrame({
                "id": ["s1", "s2"], "cat_id": ["FOODS", "FOODS"],
                "last_price_known_at_origin": [2.0, 4.0],
                "origin_id": ["O001", "O001"], "origin_d": [100, 100], "origin_date": ["2012-01-01", "2012-01-01"],
                "h": [1, 1], "target_d": [101, 101], "target_date": ["2012-01-02", "2012-01-02"],
                "target_units_sold": [1.0, 2.0],
            })
            pq.write_table(pa.Table.from_pandas(df_train), mat_A / "origin_id=O001" / "data.parquet")

            df_val = pd.DataFrame({
                "id": ["s1", "s2"], "cat_id": ["FOODS", "FOODS"],
                "last_price_known_at_origin": [99.0, 150.0],
                "origin_id": ["O002", "O002"], "origin_d": [128, 128], "origin_date": ["2012-01-29", "2012-01-29"],
                "h": [1, 1], "target_d": [129, 129], "target_date": ["2012-01-30", "2012-01-30"],
                "target_units_sold": [0.0, 1.0],
            })
            pq.write_table(pa.Table.from_pandas(df_val), mat_A / "origin_id=O002" / "data.parquet")

            cfg = ModelConfig(matrix_A_dir=mat_A)
            sampled = {"s1", "s2"}

            train_medians = compute_fold_category_median_prices(["O001"], sampled, cfg)
            self.assertEqual(train_medians["FOODS"], 3.0)

            df_val_loaded, _, _, _ = load_partition_subset(
                mat_A, ["O002"], sampled, cfg, fold_cat_medians=train_medians
            )

            expected_ratios = np.array([99.0 / 3.0, 150.0 / 3.0], dtype=np.float32)
            actual_ratios = df_val_loaded["price_ratio_to_cat_median"].to_numpy()
            np.testing.assert_allclose(actual_ratios, expected_ratios, rtol=1e-5)

    def test_exact_horizon_validation_rejection_of_floats_and_nulls(self):
        """Test 9: Exact horizon validation strictly requires integer membership in {1, ..., 28},
        non-null coordinate keys, and rejection of float values like 14.5.
        """
        # Baseline valid dataframe with 28 horizons
        records = []
        for h in range(1, 29):
            records.append({"id": "s1", "origin_id": "O1000", "h": h})
        df_valid = pd.DataFrame(records)
        
        # Valid should pass without exception
        verify_horizon_integrity(df_valid, context_name="test_valid")
        
        # Case 1: Float 14.5 replaces integer 14 (count is 28, min is 1, max is 28)
        df_float = df_valid.astype({"h": float})
        df_float.loc[df_float["h"] == 14.0, "h"] = 14.5
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_float, context_name="test_float")
        self.assertIn("Non-integer horizon values", str(ctx.exception))
        
        # Case 2: Null key in 'h'
        df_null_h = df_valid.copy()
        df_null_h.loc[0, "h"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_null_h, context_name="test_null_h")
        self.assertIn("Null values detected", str(ctx.exception))
        
        # Case 3: Null key in 'id'
        df_null_id = df_valid.copy()
        df_null_id.loc[0, "id"] = None
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_null_id, context_name="test_null_id")
        self.assertIn("Null values detected", str(ctx.exception))
        
        # Case 4: Missing required column
        df_missing_col = df_valid.drop(columns=["origin_id"])
        with self.assertRaises(KeyError) as ctx:
            verify_horizon_integrity(df_missing_col, context_name="test_missing_col")
        self.assertIn("Missing required column 'origin_id'", str(ctx.exception))
        
        # Case 5: Horizon outside {1, ..., 28} (e.g. 29 replacing 14)
        df_out_of_range = df_valid.copy()
        df_out_of_range.loc[df_out_of_range["h"] == 14, "h"] = 29
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_out_of_range, context_name="test_out_of_range")
        self.assertIn("Horizons outside valid set", str(ctx.exception))

        # Case 6: Duplicate horizon (15 duplicated, 14 missing)
        df_dup = df_valid.copy()
        df_dup.loc[df_dup["h"] == 14, "h"] = 15
        with self.assertRaises(ValueError) as ctx:
            verify_horizon_integrity(df_dup, context_name="test_dup")
        self.assertIn("Duplicate", str(ctx.exception))

    def test_evaluator_coordinate_alignment_with_fold_day_c(self):
        """Test 10: Calling the actual predict_evaluation_origins evaluator with fold_day_C supplied
        across 5 conditions:
        1. Correctly aligned inputs
        2. Reordered SKU rows (evaluator restores alignment; never misassigns forecasts)
        3. Reordered horizons within one SKU (evaluator restores alignment)
        4. Incorrect target days (raises clear error)
        5. Missing or duplicate keys (raises clear error)
        """
        class MockBooster:
            def predict(self, df):
                return df["f1"].to_numpy(dtype=np.float64)

        class MockBenchmark:
            def predict(self, coords):
                n = len(coords)
                return np.zeros(n, dtype=np.float64), np.ones(n, dtype=bool)

        orig_id = "O1000"
        orig_d = 1000
        target_days = list(range(orig_d + 1, orig_d + 29))  # 1001..1028
        
        fold_day_C = pd.DataFrame({
            "d_num": target_days,
            "c_feat": [1.0] * 28,
            "ph_thanksgiving": ["NONE"] * 28,
            "k_thanksgiving": [999] * 28,
        })
        
        models = {
            "model_A": {"booster": MockBooster(), "feature_cols": ["f1"], "cat_cols": []},
            "model_B": {"booster": MockBooster(), "feature_cols": ["f1"], "cat_cols": []},
            "model_C": {"booster": MockBooster(), "feature_cols": ["f1", "k_thanksgiving"], "cat_cols": []},
            "model_D": {"booster": MockBooster(), "feature_cols": ["f1", "k_thanksgiving"], "cat_cols": []},
        }
        sampled_ids = {"SKU_1", "SKU_2"}
        
        def build_matrix_A_df():
            rows = []
            for s_id, base_val in [("SKU_1", 100.0), ("SKU_2", 200.0)]:
                for h in range(1, 29):
                    rows.append({
                        "id": s_id, "item_id": s_id, "dept_id": "D1", "cat_id": "C1",
                        "store_id": "ST1", "state_id": "CA",
                        "origin_id": orig_id, "origin_d": orig_d, "h": h,
                        "target_d": orig_d + h, "target_date": f"2013-10-{h:02d}",
                        "target_units_sold": 5.0, "sales_last_O": 5.0,
                        "sales_zero_ratio_28d": 0.0, "sales_mean_28d": 5.0,
                        "f1": base_val + float(h),
                    })
            return pd.DataFrame(rows)

        def build_matrix_B_df():
            rows = []
            for s_id, base_val in [("SKU_1", 100.0), ("SKU_2", 200.0)]:
                for h in range(1, 29):
                    rows.append({
                        "id": s_id, "item_id": s_id, "dept_id": "D1", "cat_id": "C1",
                        "store_id": "ST1", "state_id": "CA",
                        "origin_id": orig_id, "origin_d": orig_d, "h": h,
                        "target_d": orig_d + h, "target_date": f"2013-10-{h:02d}",
                        "t_event_name_1": "None", "t_no_event": 1,
                        "f1": base_val + float(h),
                    })
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mat_A = tmp_path / "matrix_A"
            mat_B = tmp_path / "matrix_B"
            (mat_A / f"origin_id={orig_id}").mkdir(parents=True)
            (mat_B / f"origin_id={orig_id}").mkdir(parents=True)
            
            df_A = build_matrix_A_df()
            pq.write_table(pa.Table.from_pandas(df_A), mat_A / f"origin_id={orig_id}" / "data.parquet")
            
            cfg = ModelConfig(
                matrix_A_dir=mat_A,
                matrix_B_dir=mat_B,
                features_dir=tmp_path,
                clip_nonnegative=True,
            )
            
            # --- Condition 1: Correctly aligned inputs ---
            df_B_aligned = build_matrix_B_df()
            pq.write_table(pa.Table.from_pandas(df_B_aligned), mat_B / f"origin_id={orig_id}" / "data.parquet")
            res1 = predict_evaluation_origins(
                models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                fold_day_C=fold_day_C, log=lambda m: None,
            )
            self.assertEqual(len(res1), 56)
            s1_preds = res1[res1["id"] == "SKU_1"].sort_values("h")
            s2_preds = res1[res1["id"] == "SKU_2"].sort_values("h")
            np.testing.assert_array_equal(s1_preds["pred_B"].to_numpy(), 100.0 + np.arange(1, 29))
            np.testing.assert_array_equal(s2_preds["pred_B"].to_numpy(), 200.0 + np.arange(1, 29))
            np.testing.assert_array_equal(s1_preds["pred_C"].to_numpy(), 100.0 + np.arange(1, 29))
            np.testing.assert_array_equal(s1_preds["pred_D"].to_numpy(), 100.0 + np.arange(1, 29))

            # --- Condition 2: Reordered SKU rows in Matrix B ---
            # Matrix B has SKU_2 first, then SKU_1 (reversed SKU order)
            df_B_reordered_skus = pd.concat([
                df_B_aligned[df_B_aligned["id"] == "SKU_2"],
                df_B_aligned[df_B_aligned["id"] == "SKU_1"],
            ], ignore_index=True)
            pq.write_table(pa.Table.from_pandas(df_B_reordered_skus), mat_B / f"origin_id={orig_id}" / "data.parquet")
            
            res2 = predict_evaluation_origins(
                models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                fold_day_C=fold_day_C, log=lambda m: None,
            )
            self.assertEqual(len(res2), 56)
            # Crucial assertion: Evaluator MUST restore alignment to reference coordinates (df_A order: SKU_1 first, then SKU_2)
            self.assertEqual(list(res2["id"][:28]), ["SKU_1"] * 28)
            self.assertEqual(list(res2["id"][28:]), ["SKU_2"] * 28)
            # And predictions must NEVER be misassigned to the wrong SKU!
            s1_preds2 = res2[res2["id"] == "SKU_1"].sort_values("h")
            s2_preds2 = res2[res2["id"] == "SKU_2"].sort_values("h")
            np.testing.assert_array_equal(s1_preds2["pred_B"].to_numpy(), 100.0 + np.arange(1, 29))
            np.testing.assert_array_equal(s2_preds2["pred_B"].to_numpy(), 200.0 + np.arange(1, 29))

            # --- Condition 3: Reordered horizons within one SKU in Matrix B ---
            # Reverse horizons for SKU_1
            df_B_reordered_h = pd.concat([
                df_B_aligned[df_B_aligned["id"] == "SKU_1"].iloc[::-1],
                df_B_aligned[df_B_aligned["id"] == "SKU_2"],
            ], ignore_index=True)
            pq.write_table(pa.Table.from_pandas(df_B_reordered_h), mat_B / f"origin_id={orig_id}" / "data.parquet")
            
            res3 = predict_evaluation_origins(
                models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                fold_day_C=fold_day_C, log=lambda m: None,
            )
            self.assertEqual(len(res3), 56)
            s1_preds3 = res3[res3["id"] == "SKU_1"].sort_values("h")
            np.testing.assert_array_equal(s1_preds3["pred_B"].to_numpy(), 100.0 + np.arange(1, 29))

            # --- Condition 4: Incorrect target days in Matrix B ---
            df_B_bad_target = df_B_aligned.copy()
            df_B_bad_target.loc[0, "target_d"] = 9999
            pq.write_table(pa.Table.from_pandas(df_B_bad_target), mat_B / f"origin_id={orig_id}" / "data.parquet")
            with self.assertRaises(ValueError) as ctx:
                predict_evaluation_origins(
                    models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                    fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                    fold_day_C=fold_day_C, log=lambda m: None,
                )
            self.assertIn("Coordinate keys mismatch", str(ctx.exception))

            # --- Condition 5a: Missing required key in Matrix B ---
            df_B_missing_key = df_B_aligned.drop(columns=["h"])
            pq.write_table(pa.Table.from_pandas(df_B_missing_key), mat_B / f"origin_id={orig_id}" / "data.parquet")
            with self.assertRaises(KeyError) as ctx:
                predict_evaluation_origins(
                    models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                    fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                    fold_day_C=fold_day_C, log=lambda m: None,
                )
            self.assertIn("Missing required key column 'h'", str(ctx.exception))

            # --- Condition 5b: Duplicate keys in Matrix B ---
            df_B_dup_key = pd.concat([df_B_aligned.iloc[[0]], df_B_aligned], ignore_index=True)
            pq.write_table(pa.Table.from_pandas(df_B_dup_key), mat_B / f"origin_id={orig_id}" / "data.parquet")
            with self.assertRaises(ValueError) as ctx:
                predict_evaluation_origins(
                    models=models, benchmark=MockBenchmark(), eval_origs=[orig_id],
                    fold_id="DEV_S2013", sampled_ids=sampled_ids, cfg=cfg,
                    fold_day_C=fold_day_C, log=lambda m: None,
                )
            self.assertTrue("Duplicate coordinate keys detected" in str(ctx.exception) or "Row count mismatch" in str(ctx.exception))

    def test_population_eligibility_audit_synthetic(self):
        """Test 11: Population eligibility audit correctly calculates first observed positive sales,
        checks 28-day minimum history (first_sale_d <= origin_d - 27) against dev and test holiday origins,
        and derives audit reasons accurately.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            panel_p = tmp_path / "panel.parquet"
            
            # Create synthetic panel with _evaluation suffix
            panel_data = pd.DataFrame({
                "id": ["s_unranked_evaluation", "s_inactive_evaluation"],
                "d_num": [1410, 1798],
                "date": ["2014-12-08", "2015-12-31"],
                "units_sold": [5.0, 3.0],
            })
            pq.write_table(pa.Table.from_pandas(panel_data), panel_p)
            
            # Synthetic origins metadata
            dev_hol_origs = ["O1420"]
            test_hol_origs = ["O1735", "O1785"]
            full_preds_df = pd.DataFrame({
                "origin_id": ["O1420", "O1735", "O1785"],
                "origin_d": [1420, 1735, 1785],
                "fold_id": ["DEV_S2014", "FINAL_TEST", "FINAL_TEST"],
            })
            
            audit_df = generate_population_eligibility_audit(
                full_preds_df=full_preds_df,
                dev_hol_origs=dev_hol_origs,
                test_hol_origs=test_hol_origs,
                g3_ids={"s_unranked"},
                inact_ids={"s_inactive"},
                out_dir=tmp_path,
                panel_path=panel_p,
                log=lambda m: None,
            )
            
            self.assertEqual(len(audit_df), 2)
            unranked_row = audit_df[audit_df["id"] == "s_unranked"].iloc[0]
            self.assertEqual(unranked_row["population_group"], "UNRANKED_ACTIVE_2015")
            self.assertEqual(unranked_row["first_positive_sale_d"], 1410)
            self.assertEqual(unranked_row["dev_holiday_origins_satisfied"], 0)
            self.assertFalse(unranked_row["satisfied_dev_min_history"])
            self.assertEqual(unranked_row["test_holiday_origins_satisfied"], 2)  # 1735 - 27 = 1708 >= 1410 and 1785 - 27 = 1758 >= 1410
            self.assertTrue(unranked_row["satisfied_test_min_history"])
            self.assertIn("No eligible development forecast origins under the 28-day minimum-history rule", unranked_row["eligibility_reason"])

            inactive_row = audit_df[audit_df["id"] == "s_inactive"].iloc[0]
            self.assertEqual(inactive_row["population_group"], "INACTIVE_2015_HOLIDAY")
            self.assertEqual(inactive_row["first_positive_sale_d"], 1798)
            self.assertEqual(inactive_row["test_holiday_origins_satisfied"], 0)
            self.assertFalse(inactive_row["satisfied_test_min_history"])
            self.assertIn("Zero sales observed prior to 2015 holiday forecast origins", inactive_row["eligibility_reason"])


if __name__ == "__main__":
    unittest.main()

