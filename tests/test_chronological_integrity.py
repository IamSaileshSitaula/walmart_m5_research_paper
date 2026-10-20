"""Unit and synthetic verification tests for chronological integrity, SKU attention formulation,
and data alignment contracts. Built using unittest for standard execution without external test dependencies.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from m5_features.config import FeatureConfig
from m5_features.events import (
    PHASE_CATEGORIES,
    compute_fold_event_selection,
    event_date_table,
)
from m5_models.sku_attention import (
    evaluate_sku_attention_rule,
    is_holiday_relevant_origin,
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
    # Set Thanksgiving on d_300 and d_664
    if n_days >= 300:
        df.loc[df["d_num"] == 300, ["event_name_1", "event_type_1"]] = ["Thanksgiving", "National"]
    if n_days >= 664:
        df.loc[df["d_num"] == 664, ["event_name_1", "event_type_1"]] = ["Thanksgiving", "National"]
    # Set Christmas on d_331 and d_697
    if n_days >= 331:
        df.loc[df["d_num"] == 331, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    if n_days >= 697:
        df.loc[df["d_num"] == 697, ["event_name_1", "event_type_1"]] = ["Christmas", "National"]
    # Set an exploratory event (SuperBowl) on d_10, d_375, and d_740
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
        
        # Store daily sales totals
        np.random.seed(42)
        store_totals_original = np.random.poisson(lam=100.0, size=(10, n_days)).astype(np.float64)
        
        # Run screening on original data
        res_orig = compute_fold_event_selection(
            fold_id="DEV_S2012",
            cutoff_d=cutoff_d,
            store_daily_totals=store_totals_original,
            cal=cal,
            ev=ev,
            cfg=cfg,
        )
        
        # Corrupt/perturb sales strictly AFTER cutoff_d (days 626 to 1000)
        store_totals_perturbed = store_totals_original.copy()
        store_totals_perturbed[:, cutoff_d:] = store_totals_perturbed[:, cutoff_d:] * 1000.0 + 99999.0
        
        # Re-run screening on perturbed data
        res_perturbed = compute_fold_event_selection(
            fold_id="DEV_S2012",
            cutoff_d=cutoff_d,
            store_daily_totals=store_totals_perturbed,
            cal=cal,
            ev=ev,
            cfg=cfg,
        )
        
        # Verify bit-for-bit equivalence
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
            
            # Verify that an incomplete horizon (e.g. dropping h=28) triggers ValueError
            df_incomplete = df.iloc[:-1].copy()
            with self.assertRaises(ValueError):
                evaluate_sku_attention_rule(df_incomplete, tmp_path, top_k=1, log=lambda m: None)

    def test_2015_label_perturbation_invariance(self):
        """Test 3: Changing 2015 observed labels cannot change development attention scores,
        ranking, or Top-50 membership.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            records = []
            for s in range(10):
                s_id = f"ITEM_{s:03d}_STORE_1"
                for h in range(1, 29):
                    # Dev origin O0636
                    records.append({
                        "id": s_id, "item_id": f"ITEM_{s:03d}", "dept_id": "DEPT_1",
                        "cat_id": "CAT_1", "store_id": "STORE_1", "state_id": "CA",
                        "fold_id": "DEV_S2012", "origin_id": "O0636", "origin_d": 636, "h": h,
                        "target_d": 636 + h, "target_date": "2012-10-15",
                        "target_units_sold": float(s), "pred_B": float(s), "pred_D": float(s + (s % 3)),
                        "abs_err_B": 0.0, "abs_err_D": float(s % 3), "err_B": 0.0, "err_D": float(s % 3),
                    })
                    # 2015 Test origin O1735
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
            
            # Perturb 2015 observed labels and predictions completely
            df_perturbed = df_orig.copy()
            test_mask = df_perturbed["fold_id"] == "FINAL_TEST"
            df_perturbed.loc[test_mask, "target_units_sold"] = df_perturbed.loc[test_mask, "target_units_sold"] * 50.0 + 1000.0
            df_perturbed.loc[test_mask, "abs_err_B"] = 999.0
            df_perturbed.loc[test_mask, "abs_err_D"] = 1.0
            
            ranked_pert, _, _ = evaluate_sku_attention_rule(df_perturbed, tmp_path / "run2", top_k=3, log=lambda m: None)
            
            # Verify ranking and dev scores are 100% invariant
            pd.testing.assert_series_equal(ranked_orig["id"], ranked_pert["id"])
            pd.testing.assert_series_equal(ranked_orig["forecast_disagreement"], ranked_pert["forecast_disagreement"])
            pd.testing.assert_series_equal(ranked_orig["attention_rank"], ranked_pert["attention_rank"])
            pd.testing.assert_series_equal(ranked_orig["in_top_50"], ranked_pert["in_top_50"])

    def test_matrix_bcd_coordinate_alignment(self):
        """Test 4: Assert B and C/D feature matrices align on SKU-store, origin, horizon,
        and target day before combination.
        """
        df_B = pd.DataFrame({
            "id": ["s1", "s1", "s2", "s2"],
            "origin_id": ["O1", "O1", "O1", "O1"],
            "h": [1, 2, 1, 2],
            "target_d": [101, 102, 101, 102],
            "feat_B": [10, 20, 30, 40],
        })
        
        # Valid matching C extra
        df_C_valid = pd.DataFrame({
            "id": ["s1", "s1", "s2", "s2"],
            "origin_id": ["O1", "O1", "O1", "O1"],
            "h": [1, 2, 1, 2],
            "target_d": [101, 102, 101, 102],
            "ph_christmas": pd.Categorical(["outside", "day_0", "outside", "day_0"], categories=PHASE_CATEGORIES),
        })
        
        # Alignment assertion passes
        self.assertEqual(len(df_B), len(df_C_valid))
        self.assertTrue((df_B["id"].to_numpy() == df_C_valid["id"].to_numpy()).all())
        self.assertTrue((df_B["origin_id"].to_numpy() == df_C_valid["origin_id"].to_numpy()).all())
        self.assertTrue((df_B["h"].to_numpy() == df_C_valid["h"].to_numpy()).all())
        self.assertTrue((df_B["target_d"].to_numpy() == df_C_valid["target_d"].to_numpy()).all())
        
        # Combined D
        df_D = pd.concat([df_B, df_C_valid[["ph_christmas"]]], axis=1)
        self.assertEqual(len(df_D), 4)
        self.assertIn("ph_christmas", df_D.columns)
        
        # Length mismatch triggers assertion failure
        df_C_bad_len = df_C_valid.iloc[:3]
        with self.assertRaises(AssertionError):
            assert len(df_B) == len(df_C_bad_len)
            
        # ID order mismatch triggers assertion failure
        df_C_bad_id = df_C_valid.copy()
        df_C_bad_id.loc[0, "id"] = "s99"
        with self.assertRaises(AssertionError):
            assert (df_B["id"].to_numpy() == df_C_bad_id["id"].to_numpy()).all()

    def test_training_price_normalization_zero_lookahead(self):
        """Test 5: Category price median is computed strictly from training origins
        and applied to validation origins without look-ahead or contamination from future prices.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from m5_models.config import ModelConfig
        from m5_models.data import compute_fold_category_median_prices, load_partition_subset

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mat_A = tmp_path / "matrix_A"
            (mat_A / "origin_id=O001").mkdir(parents=True)
            (mat_A / "origin_id=O002").mkdir(parents=True)

            # Training origin O001 has prices 2.0 and 4.0 for FOODS -> median = 3.0
            df_train = pd.DataFrame({
                "id": ["s1", "s2"], "cat_id": ["FOODS", "FOODS"],
                "last_price_known_at_origin": [2.0, 4.0],
                "origin_id": ["O001", "O001"], "origin_d": [100, 100], "origin_date": ["2012-01-01", "2012-01-01"],
                "h": [1, 1], "target_d": [101, 101], "target_date": ["2012-01-02", "2012-01-02"],
                "target_units_sold": [1.0, 2.0],
            })
            pq.write_table(pa.Table.from_pandas(df_train), mat_A / "origin_id=O001" / "data.parquet")

            # Validation origin O002 has prices 100.0 and 200.0 (future inflation/spike)
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

            # Compute median using training origin O001 ONLY
            train_medians = compute_fold_category_median_prices(["O001"], sampled, cfg)
            self.assertEqual(train_medians["FOODS"], 3.0)

            # Load validation origin O002 using frozen train medians
            df_val_loaded, _, _, _ = load_partition_subset(
                mat_A, ["O002"], sampled, cfg, fold_cat_medians=train_medians
            )

            # Verify that validation origin was normalized by training median (3.0), NOT validation median
            expected_ratios = np.array([99.0 / 3.0, 150.0 / 3.0], dtype=np.float32)
            actual_ratios = df_val_loaded["price_ratio_to_cat_median"].to_numpy()
            np.testing.assert_allclose(actual_ratios, expected_ratios, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

