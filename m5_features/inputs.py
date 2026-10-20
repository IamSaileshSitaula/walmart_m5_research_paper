"""Inspect, pin and load the cleaned inputs (read-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import FIELD_MAP, REQUIRED_PANEL_COLUMNS


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def inspect_panel(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    md = pf.metadata
    schema = pf.schema_arrow
    nulls = {n: 0 for n in schema.names}
    mins, maxs = {}, {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for c in range(g.num_columns):
            col = g.column(c)
            name = col.path_in_schema
            s = col.statistics
            if s is None:
                continue
            nulls[name] += s.null_count or 0
            if s.has_min_max and name in ("date", "d_num", "units_sold", "sell_price_raw", "wm_yr_wk"):
                mins[name] = s.min if name not in mins else min(mins[name], s.min)
                maxs[name] = s.max if name not in maxs else max(maxs[name], s.max)
    missing = [c for c in REQUIRED_PANEL_COLUMNS if c not in schema.names]
    return {
        "path": str(path), "size_bytes": path.stat().st_size, "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "columns": [{"name": f.name, "arrow_type": str(f.type), "null_count": nulls.get(f.name, 0)} for f in schema],
        "date_min": str(mins.get("date")), "date_max": str(maxs.get("date")),
        "d_num_min": mins.get("d_num"), "d_num_max": maxs.get("d_num"),
        "units_sold_min": mins.get("units_sold"), "units_sold_max": maxs.get("units_sold"),
        "sell_price_min": mins.get("sell_price_raw"), "sell_price_max": maxs.get("sell_price_raw"),
        "missing_required_columns": missing,
        "field_map": FIELD_MAP,
    }


def load_wide(path: Path, log) -> dict:
    """Load the panel into dense item-store x day matrices indexed by TRUE
    calendar day (d_num - 1). Fails if any (series, day) cell is missing or
    duplicated, so offsets such as O-7 are genuine calendar-day lags."""
    cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "d_num",
            "units_sold", "units_sold_raw", "units_corrected_flag", "sell_price_raw"]
    t = pq.read_table(path, columns=cols,
                      read_dictionary=["item_id", "dept_id", "cat_id", "store_id", "state_id"])
    t = t.unify_dictionaries().combine_chunks()
    n_rows = t.num_rows
    item_idx = t["item_id"].chunk(0).indices.to_numpy().astype(np.int64)
    store_idx = t["store_id"].chunk(0).indices.to_numpy().astype(np.int64)
    item_lv = np.array(t["item_id"].chunk(0).dictionary.to_pylist())
    store_lv = np.array(t["store_id"].chunk(0).dictionary.to_pylist())
    d_num = t["d_num"].to_numpy().astype(np.int64)
    D = int(d_num.max())
    if int(d_num.min()) != 1:
        raise ValueError("panel does not start at d_1")
    key = store_idx * len(item_lv) + item_idx
    uniq, first_row, sidx = np.unique(key, return_index=True, return_inverse=True)
    S = len(uniq)
    if n_rows != S * D:
        raise ValueError(f"panel rows {n_rows} != series {S} x days {D}: incomplete daily grid")
    Y = np.full((S, D), -1, dtype=np.int32)
    Y[sidx, d_num - 1] = t["units_sold"].to_numpy()
    if (Y < 0).any():
        raise ValueError("missing or duplicated (series, day) cells after calendar-day reindex")
    Y = Y.astype(np.int16) if Y.max() < 32767 else Y
    P = np.full((S, D), np.nan, dtype=np.float32)
    P[sidx, d_num - 1] = t["sell_price_raw"].to_numpy(zero_copy_only=False)
    n_corr = int(pc.sum(t["units_corrected_flag"]).as_py() or 0)
    n_diff = int(pc.sum(pc.not_equal(t["units_sold"], t["units_sold_raw"])).as_py() or 0)

    # static attributes: first occurrence per series
    static = pd.DataFrame({
        "store_id": store_lv[store_idx[first_row]], "item_id": item_lv[item_idx[first_row]],
    })
    for c in ["dept_id", "cat_id", "state_id"]:
        arr = t[c].chunk(0)
        static[c] = np.array(arr.dictionary.to_pylist())[arr.indices.to_numpy()[first_row]]
    log(f"Loaded panel into {S:,} series x {D:,} days (calendar-day indexed); "
        f"units_sold != units_sold_raw in {n_diff} rows; logged corrections {n_corr}")
    del t
    return {"Y": Y, "P": P, "static": static, "S": S, "D": D,
            "n_units_diff": n_diff, "n_corrections": n_corr}


def load_calendar(path: Path) -> pd.DataFrame:
    cal = pd.read_parquet(path)
    cal = cal.sort_values("d_num").reset_index(drop=True)
    if not (cal["d_num"].to_numpy() == np.arange(1, len(cal) + 1)).all():
        raise ValueError("calendar features are not a contiguous d_1..d_N sequence")
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def verify_panel_calendar_consistency(panel_path: Path, cal: pd.DataFrame) -> dict:
    """Panel event slots and snap_state must match the calendar for every day
    (checked on the first item of every store, which spans all days)."""
    first_item = pq.read_table(panel_path, columns=["item_id"]).column(0)[0].as_py()
    p = pq.read_table(panel_path, columns=["store_id", "state_id", "d_num", "event_name_1", "event_type_1",
                                           "event_name_2", "event_type_2", "snap_state"],
                      filters=[("item_id", "==", first_item)]).to_pandas()
    m = p.merge(cal, on="d_num", how="left", suffixes=("", "_cal"))
    mism = 0
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        a, b = m[c].astype("string"), m[f"{c}_cal"].astype("string")
        mism += int((a.fillna("<NA>") != b.fillna("<NA>")).sum())
    snap_exp = np.select([m["state_id"] == s for s in ["CA", "TX", "WI"]],
                         [m["snap_CA"], m["snap_TX"], m["snap_WI"]], default=-1)
    snap_mism = int((m["snap_state"].to_numpy() != snap_exp).sum())
    return {"rows_checked": len(m), "event_slot_mismatches": mism, "snap_mismatches": snap_mism,
            "probe_item": first_item}


def pin_inputs(paths: dict, pin_file: Path, repin: bool, log) -> dict:
    current = {k: {"path": str(p), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
               for k, p in paths.items()}
    if pin_file.exists() and not repin:
        pinned = json.loads(pin_file.read_text())
        diffs = [k for k in current if pinned.get(k, {}).get("sha256") != current[k]["sha256"]]
        status = "match" if not diffs else f"MISMATCH: {diffs}"
        log(f"Input pin check: {status}")
        return {"pinned": pinned, "current": current, "status": status}
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    pin_file.write_text(json.dumps(current, indent=2))
    log(f"Pinned input hashes -> {pin_file}")
    return {"pinned": current, "current": current, "status": "pinned_now"}
