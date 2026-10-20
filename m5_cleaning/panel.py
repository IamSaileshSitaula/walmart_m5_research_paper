"""Construct the daily item-store panel, store by store, and stream it to Parquet.

Memory strategy: the wide sales matrix of ONE store (~3,049 items x 1,941
days = ~5.9M rows) is reshaped, joined and written as row groups before the
next store is processed, so the ~59M-row table never sits in memory at once.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .anomalies import series_extreme_flags, store_day_low_flags, context_label
from .audit import AuditContext
from .features import check_origin_availability, week_end_index

PANEL_SCHEMA = pa.schema([
    ("id", pa.string()), ("item_id", pa.string()), ("dept_id", pa.string()),
    ("cat_id", pa.string()), ("store_id", pa.string()), ("state_id", pa.string()),
    ("d", pa.string()), ("d_num", pa.int16()), ("date", pa.date32()),
    ("wm_yr_wk", pa.int32()), ("weekday", pa.string()), ("wday", pa.int8()),
    ("month", pa.int8()), ("year", pa.int16()),
    ("event_name_1", pa.string()), ("event_type_1", pa.string()),
    ("event_name_2", pa.string()), ("event_type_2", pa.string()),
    ("calendar_join_ok", pa.bool_()), ("event_status", pa.string()),
    ("snap_state", pa.int8()),
    ("units_sold_raw", pa.int32()), ("units_sold", pa.int32()), ("units_corrected_flag", pa.bool_()),
    ("sell_price_raw", pa.float64()), ("price_listed_flag", pa.bool_()), ("price_status", pa.string()),
    ("qf_zero_units_sold", pa.bool_()), ("qf_positive_sales_no_price", pa.bool_()),
    ("audit_extreme_review_flag", pa.bool_()), ("audit_extreme_reasons", pa.string()),
    ("audit_store_day_low_flag", pa.bool_()),
])

PRICE_STATUS = ["listed", "unpriced_before_first_listed", "unpriced_within_listed_span",
                "unpriced_after_last_listed", "series_never_priced", "calendar_join_failed"]
EVENT_STATUS = ["listed_event", "no_listed_event", "calendar_join_failed"]
CAL_JOIN_COLS = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                 "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                 "snap_CA", "snap_TX", "snap_WI"]


class JoinInvariantError(AssertionError):
    pass


def _cat_rep(values: pd.Series, rep: np.ndarray) -> pd.Categorical:
    codes, uniq = pd.factorize(values)
    return pd.Categorical.from_codes(codes[rep], categories=uniq)


def load_corrections(path: Path | None) -> pd.DataFrame:
    cols = ["id", "d", "original_value", "revised_value", "reason", "evidence", "approved_by"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    c = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(cols) - set(c.columns)
    if missing:
        raise ValueError(f"corrections file missing columns {missing}")
    if (c["reason"].str.strip() == "").any() or (c["evidence"].str.strip() == "").any():
        raise ValueError("every correction needs a non-empty reason and evidence")
    return c


def build_panel(sales: pd.DataFrame, d_cols: list[str], cal: pd.DataFrame, prices: pd.DataFrame,
                corrections: pd.DataFrame, folds: pd.DataFrame, day_ctx: dict,
                out_path: Path, ctx: AuditContext) -> dict:
    cfg, log = ctx.cfg, ctx.log
    D = len(d_cols)
    d_nums = np.array([int(c[2:]) for c in d_cols], dtype=np.int16)
    d_cat = pd.CategoricalDtype(categories=list(dict.fromkeys(list(d_cols) + cal["d"].tolist())))
    d_codes = d_cat.categories.get_indexer(d_cols)

    cal_join = cal[CAL_JOIN_COLS].copy()
    cal_join["d"] = cal_join["d"].astype(d_cat)
    for c in ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]:
        cal_join[c] = cal_join[c].astype("int32") if not cal_join[c].isna().any() else cal_join[c]
    for c in ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal_join[c] = cal_join[c].astype("category")
    sales_d_set = set(d_cols)
    ctx.findings["calendar_days_without_sales"] = int((~cal["d"].isin(sales_d_set)).sum())
    ctx.findings["calendar_days_without_sales_range"] = (
        f"{cal.loc[~cal['d'].isin(sales_d_set), 'd'].iloc[0]}..{cal.loc[~cal['d'].isin(sales_d_set), 'd'].iloc[-1]}"
        if (~cal["d"].isin(sales_d_set)).any() else "none")
    last_sales_wk = int(cal.loc[cal["d"] == d_cols[-1], "wm_yr_wk"].iloc[0])

    wk_full = cal["wm_yr_wk"].to_numpy().astype(int)
    week_end = week_end_index(wk_full)[:D]
    wday_idx = cal["wday"].to_numpy().astype(int)[:D]
    origins = [int(T) - 1 for T in folds.loc[folds["included"], "_T"]]

    prices = prices.copy()
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype(cal_join["wm_yr_wk"].dtype)
    prices_by_store = {k: v for k, v in prices.groupby("store_id", sort=False)}

    writer = pq.ParquetWriter(out_path, PANEL_SCHEMA, compression="zstd")
    st = {k: 0 for k in ["rows_before", "rows_after_calendar", "rows_after_price",
                         "units_before", "units_after_calendar", "units_after_price",
                         "calendar_unmatched_rows", "price_rows_matched", "price_rows_unmatched",
                         "price_rows_unmatched_after_sales_end", "price_rows_unmatched_within_sales_span",
                         "price_rows_item_not_in_store", "positive_sales_no_price_rows",
                         "positive_sales_no_price_units", "extreme_flags", "extreme_isolated",
                         "store_day_low", "corrections_applied"]}
    status_rows = np.zeros(len(PRICE_STATUS), dtype=np.int64)
    status_pos_rows = np.zeros(len(PRICE_STATUS), dtype=np.int64)
    status_units = np.zeros(len(PRICE_STATUS), dtype=np.int64)
    status_series_weeks = np.zeros(len(PRICE_STATUS), dtype=np.int64)
    series_status_any = {s: 0 for s in PRICE_STATUS}
    zero_tables, gap_tables, avail_rows, per_store = [], [], [], []
    corr_log = []

    for store in sorted(sales["store_id"].unique()):
        s = sales.loc[sales["store_id"] == store].sort_values("item_id").reset_index(drop=True)
        n_i = len(s)
        Y = s[d_cols].to_numpy(dtype=np.int32)
        rep = np.repeat(np.arange(n_i), D)
        long = pd.DataFrame({
            "_row": np.arange(n_i * D, dtype=np.int64),
            "id": _cat_rep(s["id"], rep), "item_id": _cat_rep(s["item_id"], rep),
            "dept_id": _cat_rep(s["dept_id"], rep), "cat_id": _cat_rep(s["cat_id"], rep),
            "store_id": _cat_rep(s["store_id"], rep), "state_id": _cat_rep(s["state_id"], rep),
            "d": pd.Categorical.from_codes(np.tile(d_codes, n_i), dtype=d_cat),
            "d_num": np.tile(d_nums, n_i),
            "units_sold_raw": Y.ravel(),
        })
        n0, u0 = len(long), int(Y.sum(dtype=np.int64))

        # ---- Join 1: calendar on d ---------------------------------------
        m = long.merge(cal_join, on="d", how="left", validate="many_to_one", indicator="_cal")
        n1, u1 = len(m), int(m["units_sold_raw"].to_numpy().sum(dtype=np.int64))
        if n1 != n0 or u1 != u0:
            raise JoinInvariantError(f"{store}: calendar join changed rows {n0}->{n1} or units {u0}->{u1}")
        if not m["_row"].is_monotonic_increasing:
            m = m.sort_values("_row", ignore_index=True)
        cal_ok = (m["_cal"] == "both").to_numpy()

        # ---- Join 2: prices on (store_id, item_id, wm_yr_wk) -------------
        ps = prices_by_store.get(store, prices.iloc[0:0]).copy()
        ps["item_id"] = pd.Categorical(ps["item_id"], categories=m["item_id"].cat.categories)
        ps["store_id"] = pd.Categorical(ps["store_id"], categories=m["store_id"].cat.categories)
        st["price_rows_item_not_in_store"] += int(ps["item_id"].isna().sum())
        ps = ps.dropna(subset=["item_id"]).rename(columns={"sell_price": "sell_price_raw"})
        m = m.merge(ps, on=["store_id", "item_id", "wm_yr_wk"], how="left",
                    validate="many_to_one", indicator="_price")
        n2, u2 = len(m), int(m["units_sold_raw"].to_numpy().sum(dtype=np.int64))
        if n2 != n0 or u2 != u0:
            raise JoinInvariantError(f"{store}: price join changed rows {n0}->{n2} or units {u0}->{u2}")
        if not m["_row"].is_monotonic_increasing:
            m = m.sort_values("_row", ignore_index=True)

        # Unmatched price keys (price rows not used by any sales day)
        used = m.loc[m["_price"] == "both", ["item_id", "wm_yr_wk"]].drop_duplicates()
        chk = ps[["item_id", "wm_yr_wk"]].merge(used, how="left", indicator=True)
        unmatched = chk["_merge"] == "left_only"
        st["price_rows_matched"] += int((~unmatched).sum())
        st["price_rows_unmatched"] += int(unmatched.sum())
        st["price_rows_unmatched_after_sales_end"] += int((unmatched & (chk["wm_yr_wk"] > last_sales_wk)).sum())
        st["price_rows_unmatched_within_sales_span"] += int((unmatched & (chk["wm_yr_wk"] <= last_sales_wk)).sum())

        # ---- Missingness classification ----------------------------------
        price = m["sell_price_raw"].to_numpy(dtype=np.float64)
        listed = ~np.isnan(price)
        span = ps.groupby("item_id", observed=False)["wm_yr_wk"].agg(["min", "max"]).astype(float)
        first = span["min"].to_numpy()[rep]
        last = span["max"].to_numpy()[rep]
        wk = m["wm_yr_wk"].to_numpy(dtype=np.float64)
        never = np.isnan(first)
        with np.errstate(invalid="ignore"):
            code = np.select([~cal_ok, listed, never, wk < first, wk > last], [5, 0, 4, 1, 3], default=2)
        units = Y.ravel()
        status_rows += np.bincount(code, minlength=6)
        status_pos_rows += np.bincount(code, weights=(units > 0), minlength=6).astype(np.int64)
        status_units += np.bincount(code, weights=units, minlength=6).astype(np.int64)
        item_code = rep.astype(np.int64)
        wk_key = item_code * 100_000 + np.nan_to_num(wk, nan=0).astype(np.int64)
        _, first_idx = np.unique(wk_key, return_index=True)
        status_series_weeks += np.bincount(code[first_idx], minlength=6)
        code2d = code.reshape(n_i, D)
        for k, name in enumerate(PRICE_STATUS):
            series_status_any[name] += int((code2d == k).any(axis=1).sum())

        # within-span gaps at series-week level, for review
        gap = code == 2
        if gap.any():
            g = pd.DataFrame({"store_id": store, "item_id": m["item_id"].to_numpy()[gap],
                              "wm_yr_wk": wk[gap].astype(int), "units": units[gap]})
            gap_tables.append(g.groupby(["store_id", "item_id", "wm_yr_wk"], observed=True)
                              .agg(days=("units", "size"), units_sold_raw=("units", "sum")).reset_index())

        # ---- Documented corrections (none unless evidence supplied) -------
        units_clean = units.copy()
        corrected = np.zeros(len(units), dtype=bool)
        cs = corrections[corrections["id"].isin(set(s["id"]))]
        if len(cs):
            pos_of = {v: i for i, v in enumerate(s["id"])}
            for c in cs.itertuples():
                ridx = pos_of[c.id] * D + (int(c.d[2:]) - 1)
                if int(units[ridx]) != int(c.original_value):
                    raise ValueError(f"correction {c.id} {c.d}: original_value {c.original_value} "
                                     f"!= raw {units[ridx]}")
                units_clean[ridx] = int(c.revised_value)
                corrected[ridx] = True
                corr_log.append({"correction_id": f"SALES-{c.id}-{c.d}", "level": "sales_cell",
                                 "source_file": "sales", "record_key": f"{c.id}|{c.d}", "field": "units_sold",
                                 "original_value": c.original_value, "revised_value": c.revised_value,
                                 "values_changed": 1, "reason": c.reason,
                                 "evidence": f"{c.evidence} (approved_by={c.approved_by})"})
            st["corrections_applied"] += len(cs)

        # ---- Review flags (no values modified) ---------------------------
        P = price.reshape(n_i, D)
        flag2d, cand = series_extreme_flags(Y, P, wday_idx, cfg)
        reasons = np.full(n_i * D, None, dtype=object)
        if len(cand):
            flat = cand["row_item"].to_numpy() * D + cand["day_idx"].to_numpy()
            reasons[flat] = cand["reasons"].to_numpy()
            st["extreme_flags"] += len(cand)
            st["extreme_isolated"] += int(cand["isolated_spike"].sum())
            ev, nn, nr = day_ctx["events"], day_ctx["near_name"], day_ctx["near_rel"]
            snap_arr = cal_join[f"snap_{s['state_id'].iloc[0]}"].to_numpy()[:D]
            di = cand["day_idx"].to_numpy()
            cand = cand.assign(
                id=s["id"].to_numpy()[cand["row_item"]], item_id=s["item_id"].to_numpy()[cand["row_item"]],
                dept_id=s["dept_id"].to_numpy()[cand["row_item"]], store_id=store,
                d=np.array(d_cols)[di], date=cal["date"].to_numpy()[di],
                weekday=cal["weekday"].to_numpy()[di], wm_yr_wk=wk_full[di],
                events_on_day=ev[di], nearest_event_pm7=nn[di], nearest_event_rel_day=nr[di],
                snap_state=snap_arr[di],
                price_status=np.array(PRICE_STATUS)[code2d[cand["row_item"], di]],
            )
            cand["review_context"] = [context_label(e, n_, r_, sn, pr, cfg) for e, n_, r_, sn, pr in zip(
                cand["events_on_day"], cand["nearest_event_pm7"], cand["nearest_event_rel_day"],
                cand["snap_state"], cand["price_ratio_to_series_median"])]
            cand["proposed_action"] = "retain_unchanged; human review (no evidence of invalid source record)"
            ctx.add_exceptions(cand.drop(columns=["row_item", "day_idx"]), "extreme_sales_review", "panel")

        low_day, tot, med = store_day_low_flags(Y, cfg)
        st["store_day_low"] += int(low_day.sum())
        if low_day.any():
            di = np.nonzero(low_day)[0]
            ctx.add_exceptions(pd.DataFrame({
                "store_id": store, "d": np.array(d_cols)[di], "date": cal["date"].to_numpy()[di],
                "weekday": cal["weekday"].to_numpy()[di], "events_on_day": day_ctx["events"][di],
                "store_total_units": tot[di].astype(int), "local_median_total_pm14": med[di],
                "proposed_action": "retain_unchanged; likely store closure/partial day - recorded zeros are "
                                   "units sold, not demand"}), "store_day_low_total", "panel")

        pos_no_price = (units > 0) & ~listed
        st["positive_sales_no_price_rows"] += int(pos_no_price.sum())
        st["positive_sales_no_price_units"] += int(units[pos_no_price].sum())
        if pos_no_price.any():
            ctx.add_exceptions(pd.DataFrame({
                "id": m["id"].to_numpy()[pos_no_price], "item_id": m["item_id"].to_numpy()[pos_no_price],
                "store_id": store, "d": m["d"].to_numpy()[pos_no_price],
                "date": m["date"].to_numpy()[pos_no_price], "wm_yr_wk": wk[pos_no_price],
                "units_sold_raw": units[pos_no_price],
                "price_status": np.array(PRICE_STATUS)[code[pos_no_price]],
                "proposed_action": "retain; price stays missing; review"}), "positive_sales_no_listed_price", "panel")

        # ---- Assemble output chunk --------------------------------------
        has_event = m["event_name_1"].notna().to_numpy() | m["event_name_2"].notna().to_numpy()
        ev_code = np.select([~cal_ok, has_event], [2, 0], default=1)
        state = s["state_id"].iloc[0]
        out = pd.DataFrame({
            "id": m["id"], "item_id": m["item_id"], "dept_id": m["dept_id"], "cat_id": m["cat_id"],
            "store_id": m["store_id"], "state_id": m["state_id"], "d": m["d"], "d_num": m["d_num"],
            "date": m["date"], "wm_yr_wk": m["wm_yr_wk"], "weekday": m["weekday"], "wday": m["wday"],
            "month": m["month"], "year": m["year"],
            "event_name_1": m["event_name_1"], "event_type_1": m["event_type_1"],
            "event_name_2": m["event_name_2"], "event_type_2": m["event_type_2"],
            "calendar_join_ok": cal_ok,
            "event_status": pd.Categorical.from_codes(ev_code, categories=EVENT_STATUS),
            "snap_state": m[f"snap_{state}"],
            "units_sold_raw": units, "units_sold": units_clean, "units_corrected_flag": corrected,
            "sell_price_raw": price, "price_listed_flag": listed,
            "price_status": pd.Categorical.from_codes(code, categories=PRICE_STATUS),
            "qf_zero_units_sold": units == 0, "qf_positive_sales_no_price": pos_no_price,
            "audit_extreme_review_flag": flag2d.ravel(),
            "audit_extreme_reasons": pd.Categorical(reasons),
            "audit_store_day_low_flag": np.tile(low_day, n_i),
        })
        table = pa.Table.from_pandas(out, preserve_index=False).cast(PANEL_SCHEMA)
        writer.write_table(table, row_group_size=cfg.row_group_size)

        # ---- Stats ----------------------------------------------------------
        st["rows_before"] += n0; st["rows_after_calendar"] += n1; st["rows_after_price"] += n2
        st["units_before"] += u0; st["units_after_calendar"] += u1; st["units_after_price"] += u2
        st["calendar_unmatched_rows"] += int((~cal_ok).sum())
        zt = out.groupby(["store_id", "cat_id", "price_status", "event_status"], observed=True).agg(
            rows=("units_sold_raw", "size"), zero_rows=("qf_zero_units_sold", "sum"),
            units=("units_sold_raw", "sum")).reset_index()
        zero_tables.append(zt)
        res = check_origin_availability(Y, P, week_end, wday_idx, origins, seed=zlib.crc32(store.encode()))
        for r in res:
            r["store_id"] = store
        avail_rows += res
        per_store.append({"store_id": store, "series": n_i, "rows": n0, "units": u0,
                          "extreme_flags": len(cand), "store_day_low": int(low_day.sum())})
        log(f"  {store}: {n0:,} rows, {u0:,} units, joins OK, extreme flags={len(cand)}, "
            f"low store-days={int(low_day.sum())}")
        del long, m, out, table

    writer.close()
    ctx.corrections.extend(corr_log)
    return {
        "stats": st,
        "price_status": pd.DataFrame({
            "price_status": PRICE_STATUS, "daily_rows": status_rows, "daily_rows_with_positive_sales": status_pos_rows,
            "units_sold_raw": status_units, "series_weeks": status_series_weeks,
            "series_with_any": [series_status_any[s] for s in PRICE_STATUS]}),
        "zero_table": pd.concat(zero_tables, ignore_index=True),
        "gaps": pd.concat(gap_tables, ignore_index=True) if gap_tables else
        pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "days", "units_sold_raw"]),
        "availability": pd.DataFrame(avail_rows),
        "per_store": pd.DataFrame(per_store),
        "last_sales_wk": last_sales_wk,
    }
