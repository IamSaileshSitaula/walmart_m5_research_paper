"""Origin sampling + chronological split manifest.

Written BEFORE any data-derived feature is constructed. Uses only the
calendar (event dates) and the last observed day - never sales values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def event_dates(cal: pd.DataFrame, name: str) -> list[int]:
    m = (cal["event_name_1"] == name) | (cal["event_name_2"] == name)
    return cal.loc[m, "d_num"].astype(int).tolist()


def season_blocks(cal: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Holiday season Y = first target of the longest-lead Thanksgiving origin
    through the last target of the shortest-lead Christmas origin. It crosses
    the calendar-year boundary (Christmas + 21 days falls in January of Y+1);
    seasons are therefore labelled by the Thanksgiving year, not calendar year."""
    tg = {cal.loc[cal.d_num == d, "date"].iloc[0].year: d for d in event_dates(cal, "Thanksgiving")}
    xm = {cal.loc[cal.d_num == d, "date"].iloc[0].year: d for d in event_dates(cal, "Christmas")}
    rows = []
    for y in sorted(set(tg) & set(xm)):
        start = tg[y] - max(cfg.pre_event_leads) + 1
        end = xm[y] - min(cfg.pre_event_leads) + cfg.horizon
        rows.append({"season": y, "thanksgiving_d": tg[y], "christmas_d": xm[y],
                     "block_start_d": start, "block_end_d": end})
    b = pd.DataFrame(rows)
    date_of = dict(zip(cal["d_num"].astype(int), cal["date"]))
    for c in ["thanksgiving_d", "christmas_d", "block_start_d", "block_end_d"]:
        b[c.replace("_d", "_date")] = b[c].map(date_of)
    b["crosses_year_boundary"] = b["block_start_date"].dt.year != b["block_end_date"].dt.year
    return b


def build_origins(cal: pd.DataFrame, last_d: int, cfg: FeatureConfig) -> pd.DataFrame:
    H = cfg.horizon
    rec: dict[int, dict] = {}
    O = last_d - cfg.base_anchor_offset
    while O >= cfg.min_origin_d:
        rec[O] = {"origin_d": O, "is_base_grid": True, "pre_event_tags": []}
        O -= cfg.base_step
    for name in cfg.pre_event_holidays:
        for E in event_dates(cal, name):
            for L in cfg.pre_event_leads:
                O = E - L
                if O < cfg.min_origin_d or O + H > last_d:
                    continue
                r = rec.setdefault(O, {"origin_d": O, "is_base_grid": False, "pre_event_tags": []})
                r["pre_event_tags"].append(f"{name}@L{L}")
    o = pd.DataFrame(sorted(rec.values(), key=lambda r: r["origin_d"]))
    o["origin_type"] = np.where(o["is_base_grid"] & o["pre_event_tags"].str.len().gt(0), "base_grid+pre_event",
                                np.where(o["is_base_grid"], "base_grid", "pre_event"))
    o["pre_event_tags"] = o["pre_event_tags"].map(lambda t: ";".join(t))
    o["first_target_d"] = o["origin_d"] + 1
    o["last_target_d"] = o["origin_d"] + H
    date_of = dict(zip(cal["d_num"].astype(int), cal["date"]))
    for c in ["origin_d", "first_target_d", "last_target_d"]:
        o[c.replace("_d", "_date")] = o[c].map(date_of)
    o["window_crosses_year"] = o["first_target_date"].dt.year != o["last_target_date"].dt.year
    occ = pd.concat([cal[["d_num", "event_name_1"]].rename(columns={"event_name_1": "e"}),
                     cal[["d_num", "event_name_2"]].rename(columns={"event_name_2": "e"})]).dropna()
    o["events_in_window"] = [
        "; ".join(f"{r.e}(h={int(r.d_num) - O})" for r in occ[(occ.d_num > O) & (occ.d_num <= O + H)]
                  .sort_values("d_num").itertuples()) for O in o["origin_d"]]
    o.insert(0, "origin_id", [f"O{d:04d}" for d in o["origin_d"]])
    return o


def assign_folds(origins: pd.DataFrame, seasons: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs, fe = origins["first_target_d"].to_numpy(), origins["last_target_d"].to_numpy()
    folds, assign = [], []

    def add(fold_id, kind, eval_role, eval_mask, eval_start, eval_end, desc):
        train = fe < eval_start
        role = np.where(eval_mask, eval_role, np.where(train, "train",
                        np.where(fs > eval_end, "future_unused", "excluded_overlap")))
        for oid, r in zip(origins["origin_id"], role):
            assign.append({"fold_id": fold_id, "origin_id": oid, "role": r})
        folds.append({"fold_id": fold_id, "fold_kind": kind, "eval_role": eval_role,
                      "eval_target_start_d": int(eval_start), "eval_target_end_d": int(eval_end),
                      "n_train_origins": int(train.sum()), "n_eval_origins": int(eval_mask.sum()),
                      "description": desc})

    sb = seasons.set_index("season")
    test = sb.loc[cfg.test_season]
    test_start = int(test["block_start_d"])
    for y in cfg.dev_seasons:
        s, e = int(sb.loc[y, "block_start_d"]), int(sb.loc[y, "block_end_d"])
        add(f"DEV_S{y}", "development_holiday_season", "validation", (fs >= s) & (fe <= e), s, e,
            f"Holiday season {y}-{(y + 1) % 100:02d}: Thanksgiving {y} lead origins through Christmas {y} + 21d")
    s_ord = int(sb.loc[max(cfg.dev_seasons), "block_end_d"]) + 1
    mask_ord = (fs >= s_ord) & (fe < test_start)
    add("DEV_ORD", "development_ordinary_and_other_events", "validation", mask_ord,
        int(fs[mask_ord].min()), int(fe[mask_ord].max()),
        "Non-overlapping base origins between the last development season and the final-test block")
    add("FINAL_TEST", "final_test", "test", fs >= test_start, test_start, int(fe.max()),
        f"Holiday season {cfg.test_season} (last complete Thanksgiving/Christmas) and all later origins "
        "incl. the M5 public window d_1914-d_1941; model frozen on targets < test start")
    return pd.DataFrame(folds), pd.DataFrame(assign)


def check_splits(origins, folds, assign, test_start: int) -> dict:
    a = assign.merge(origins[["origin_id", "first_target_d", "last_target_d"]], on="origin_id")
    out = {}
    cross = 0
    for f in folds.itertuples():
        g = a[a.fold_id == f.fold_id]
        tr, ev = g[g.role == "train"], g[g.role == f.eval_role]
        if len(tr) and len(ev):
            cross += int(tr["last_target_d"].max() >= ev["first_target_d"].min())
    out["train_target_crosses_eval_labels"] = cross
    dev = a[a.fold_id.str.startswith("DEV") & a.role.isin(["train", "validation"])]
    out["dev_targets_in_final_test_period"] = int((dev["last_target_d"] >= test_start).sum())
    out["final_test_train_targets_at_or_after_test_start"] = int(
        (a[(a.fold_id == "FINAL_TEST") & (a.role == "train")]["last_target_d"] >= test_start).sum())
    out["random_split_used"] = 0
    return out


def repeated_target_dates(origins: pd.DataFrame, H: int) -> pd.DataFrame:
    days = np.concatenate([np.arange(o + 1, o + H + 1) for o in origins["origin_d"]])
    vc = pd.Series(days).value_counts().sort_index()
    return vc.rename_axis("target_d").reset_index(name="n_origins")
