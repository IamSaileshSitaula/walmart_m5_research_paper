"""Seasonal-naive benchmark calculation and coverage reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from m5_features.inputs import load_wide
from .config import ModelConfig


class SeasonalNaiveBenchmark:
    """Calculates seasonal-naive forecasts y(t - 28) with strict temporal barrier."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # Load wide sales matrix (30,490 series x 1,941 days)
        wide_data = load_wide(cfg.cleaned_panel_path, log=lambda _: None)
        self.Y = wide_data["Y"]  # int16 array of shape (S, D)
        static = wide_data["static"]
        
        # Build series id to index mapping (supporting item_store and item_store_validation)
        self.id_to_idx = {}
        for idx, (it, st) in enumerate(zip(static["item_id"], static["store_id"])):
            self.id_to_idx[f"{it}_{st}"] = idx
            self.id_to_idx[f"{it}_{st}_validation"] = idx
        
        # Compute first positive sale day for each series (1-indexed)
        # S x D
        has_sold = self.Y > 0
        first_sale_idx = np.where(has_sold.any(axis=1), has_sold.argmax(axis=1), self.Y.shape[1])
        self.first_sale_d = first_sale_idx + 1  # 1-indexed day

    def predict(self, df_coords: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Compute seasonal-naive predictions and availability flags.
        
        For target day t: reference day is t_ref = t - 28.
        Since h in [1, 28], t_ref = O + h - 28 <= O.
        Benchmark is unavailable (NaN) if t_ref < 1 or t_ref < first_sale_d.
        """
        n = len(df_coords)
        preds = np.full(n, np.nan, dtype=np.float32)
        available = np.zeros(n, dtype=bool)

        series_ids = df_coords["id"].to_numpy()
        target_ds = df_coords["target_d"].to_numpy()
        
        # Map series IDs to indices
        s_indices = np.array([self.id_to_idx.get(sid, -1) for sid in series_ids])
        
        ref_ds = target_ds - 28  # 1-indexed reference days
        
        # Eligibility mask
        valid_s = s_indices >= 0
        valid_t = ref_ds >= 1
        
        # Check against first positive sale day
        first_ds = np.where(valid_s, self.first_sale_d[s_indices], 999999)
        eligible = valid_s & valid_t & (ref_ds >= first_ds)
        
        # Extract values for eligible coordinates
        el_s = s_indices[eligible]
        el_ref_idx = ref_ds[eligible] - 1  # 0-indexed day column
        
        preds[eligible] = self.Y[el_s, el_ref_idx].astype(np.float32)
        available[eligible] = True
        
        return preds, available
