"""Raw-file loading and pre-transformation audit.

Nothing in this module writes to the raw data folder. Every check is
registered (including zero-failure checks). Failing rows/cells are routed to
the exceptions list for human review instead of being dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    Config, EXPECTED_CALENDAR_COLUMNS, EXPECTED_PRICE_COLUMNS, SALES_ID_COLUMNS,
    KNOWN_EVENT_TYPES, WEEKDAY_TO_WDAY,
)
from .io_utils import CheckRegistry, read_header


class SchemaError(Exception):
    """Raised when a raw header cannot be matched to the expected schema."""


class BlockingAuditFailure(Exception):
    """Raised when a failure makes construction of the panel unsafe."""


@dataclass
class AuditContext:
    cfg: Config
    log: callable
    reg: CheckRegistry = field(default_factory=CheckRegistry)
    corrections: list = field(default_factory=list)   # documented corrections
    exceptions: list = field(default_factory=list)    # DataFrames for review
    findings: dict = field(default_factory=dict)      # measured facts for report

    def add_exceptions(self, df: pd.DataFrame, exception_type: str, source: str) -> None:
        if df is None or len(df) == 0:
            return
        df = df.copy()
        df.insert(0, "exception_type", exception_type)
        df.insert(1, "source", source)
        self.exceptions.append(df)


# ----------------------------------------------------------------------------
# Header handling
# ----------------------------------------------------------------------------
def resolve_header(observed: list[str], expected: list[str], filename: str,
                   ctx: AuditContext, check_id: str) -> list[str]:
    """Return the column names to use.

    Exact match -> use as is. Otherwise a *positional* repair is allowed ONLY
    when (a) the column count matches and (b) the concatenation of observed
    names equals the concatenation of expected names (i.e. the delimiters were
    misplaced but no characters were lost/added). The repair is logged as a
    correction; data values are untouched and must then pass type checks.
    """
    if observed == expected:
        ctx.reg.add(check_id, "provenance", f"{filename}: header equals expected schema",
                    1, 0)
        return expected
    if len(observed) == len(expected) and "".join(observed) == "".join(expected):
        ctx.reg.add(check_id, "provenance", f"{filename}: header equals expected schema",
                    1, 1, status="FAIL->REPAIRED",
                    notes=f"observed={observed}; positional repair to {expected} (logged)")
        ctx.corrections.append({
            "correction_id": f"HDR-{filename}",
            "level": "file_header (in-memory only; raw file untouched)",
            "source_file": filename,
            "record_key": "header row (line 1)",
            "field": "column names",
            "original_value": ",".join(observed),
            "revised_value": ",".join(expected),
            "values_changed": 0,
            "reason": "Header delimiters misplaced: an empty column name appears and two "
                      "names are fused, so the file cannot be parsed by name.",
            "evidence": "Same column count as expected schema; concatenated observed names "
                        "are character-identical to concatenated expected names; data "
                        "columns must subsequently pass positional type checks "
                        "(see checks R02, R03, R09 for this file).",
        })
        ctx.log(f"WARNING {filename}: malformed header {observed} -> positional repair {expected}")
        return expected
    ctx.reg.add(check_id, "provenance", f"{filename}: header equals expected schema", 1, 1,
                notes=f"observed={observed}")
    raise SchemaError(f"{filename}: header {observed} does not match {expected} and is not "
                      f"safely repairable")


def _dup_rows(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df.duplicated(cols, keep=False)


def _int_parse(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse to numbers; return (numeric, failure_mask) where failure = present but
    not an integer-valued finite number."""
    num = pd.to_numeric(s, errors="coerce")
    bad = s.notna() & (num.isna() | ~np.isfinite(num) | (num != np.round(num)))
    return num, bad


# ----------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------
def load_and_audit_calendar(path: Path, ctx: AuditContext) -> pd.DataFrame:
    reg = ctx.reg
    names = resolve_header(read_header(path), EXPECTED_CALENDAR_COLUMNS, path.name, ctx, "H01")
    raw = pd.read_csv(path, header=0, names=names, dtype=str,
                      keep_default_na=False, na_values=[""])
    n = len(raw)
    ctx.findings["calendar_rows"] = n
    st = "calendar"

    required = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                "snap_CA", "snap_TX", "snap_WI"]
    nulls = raw[required].isna()
    reg.add("C01", st, "Required calendar fields non-missing (cells)", nulls.size, int(nulls.values.sum()))

    date = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="coerce")
    reg.add("C02", st, "date parses as YYYY-MM-DD", n, int((raw["date"].notna() & date.isna()).sum()))

    int_cols = ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]
    parsed = {}
    bad_total = 0
    for c in int_cols:
        num, bad = _int_parse(raw[c])
        parsed[c] = num
        bad_total += int(bad.sum())
    reg.add("C03", st, "Integer calendar fields are whole numbers (cells)", n * len(int_cols), bad_total)

    d_ok = raw["d"].str.fullmatch(r"d_\d+").fillna(False)
    reg.add("C04", st, "d has format d_<integer>", n, int((~d_ok).sum()))
    d_num = pd.to_numeric(raw["d"].str.slice(2), errors="coerce")

    reg.add("C05", st, "Key uniqueness: one calendar row per d", n, int(_dup_rows(raw, ["d"]).sum()))
    reg.add("C06", st, "Key uniqueness: one calendar row per date", n, int(_dup_rows(raw, ["date"]).sum()))

    order = np.argsort(d_num.to_numpy())
    d_sorted = d_num.to_numpy()[order]
    reg.add("C07", st, "d sequence is contiguous d_1..d_N", n,
            int((d_sorted != np.arange(1, n + 1)).sum()))
    date_sorted = date.to_numpy()[order]
    gaps = np.diff(date_sorted).astype("timedelta64[D]").astype(int)
    reg.add("C08", st, "Dates are consecutive days in d order (no gaps/repeats)", max(n - 1, 0),
            int((gaps != 1).sum()))
    origin = date_sorted[0]
    expected_dates = origin + (d_sorted - 1).astype("timedelta64[D]")
    reg.add("C09", st, "Exact d<->date mapping: date == first_date + (d-1) days", n,
            int((date_sorted != expected_dates).sum()))

    reg.add("C10", st, "weekday matches the calendar date", n,
            int((date.dt.day_name() != raw["weekday"]).sum()))
    wday_exp = raw["weekday"].map(WEEKDAY_TO_WDAY)
    reg.add("C11", st, "wday matches weekday (Sat=1..Fri=7)", n,
            int((wday_exp != parsed["wday"]).sum()))
    reg.add("C12", st, "month/year match the calendar date (cells)", 2 * n,
            int((date.dt.month != parsed["month"]).sum() + (date.dt.year != parsed["year"]).sum()))

    wk = parsed["wm_yr_wk"].to_numpy()[order]
    wkday = raw["weekday"].to_numpy()[order]
    change = np.r_[False, wk[1:] != wk[:-1]]
    is_sat = wkday == "Saturday"
    week_rule_fail = int((change[1:] != is_sat[1:]).sum())
    reg.add("C13", st, "wm_yr_wk changes exactly on Saturdays (Sat-Fri weeks)", max(n - 1, 0),
            week_rule_fail)
    wk_sizes = pd.Series(wk).value_counts()
    reg.add("C14", st, "Each wm_yr_wk has <= 7 days", len(wk_sizes), int((wk_sizes > 7).sum()))

    for state in ["CA", "TX", "WI"]:
        v = parsed[f"snap_{state}"]
        reg.add(f"C15_{state}", st, f"snap_{state} values are 0 or 1", n, int((~v.isin([0, 1])).sum()))

    for slot in (1, 2):
        nm, tp = raw[f"event_name_{slot}"], raw[f"event_type_{slot}"]
        bad = nm.isna() != tp.isna()
        reg.add(f"C16_{slot}", st, f"event_name_{slot} present iff event_type_{slot} present", n,
                int(bad.sum()))
        if bad.any():
            ctx.add_exceptions(raw.loc[bad, ["d", "date", f"event_name_{slot}", f"event_type_{slot}"]],
                               "event_name_type_unpaired", path.name)
    reg.add("C17", st, "event slot 2 used only when slot 1 is used", n,
            int((raw["event_name_2"].notna() & raw["event_name_1"].isna()).sum()))
    pairs = pd.concat([
        raw[["event_name_1", "event_type_1"]].set_axis(["name", "type"], axis=1),
        raw[["event_name_2", "event_type_2"]].set_axis(["name", "type"], axis=1),
    ]).dropna()
    n_types = pairs.groupby("name")["type"].nunique()
    reg.add("C18", st, "Each event name maps to exactly one event type", len(n_types),
            int((n_types > 1).sum()))
    reg.add("C19", st, "Event types belong to {Sporting, Cultural, National, Religious}",
            len(pairs), int((~pairs["type"].isin(KNOWN_EVENT_TYPES)).sum()))
    reg.add("C20", st, "event_name_1 != event_name_2 on the same day", n,
            int((raw["event_name_1"].notna() & (raw["event_name_1"] == raw["event_name_2"])).sum()))
    ws = pd.concat([raw[c] for c in ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]]).dropna()
    reg.add("C21", st, "No leading/trailing whitespace in event fields", len(ws),
            int((ws != ws.str.strip()).sum()))

    cal = pd.DataFrame({
        "d": raw["d"], "d_num": d_num.astype("Int64"), "date": date,
        "wm_yr_wk": parsed["wm_yr_wk"].astype("Int64"), "weekday": raw["weekday"],
        "wday": parsed["wday"].astype("Int64"), "month": parsed["month"].astype("Int64"),
        "year": parsed["year"].astype("Int64"),
        "event_name_1": raw["event_name_1"], "event_type_1": raw["event_type_1"],
        "event_name_2": raw["event_name_2"], "event_type_2": raw["event_type_2"],
        "snap_CA": parsed["snap_CA"].astype("Int64"), "snap_TX": parsed["snap_TX"].astype("Int64"),
        "snap_WI": parsed["snap_WI"].astype("Int64"),
    }).sort_values("d_num").reset_index(drop=True)

    ctx.findings["calendar_first_date"] = str(cal["date"].min().date())
    ctx.findings["calendar_last_date"] = str(cal["date"].max().date())
    ctx.findings["calendar_d_range"] = f"{cal['d'].iloc[0]}..{cal['d'].iloc[-1]}"
    ctx.findings["calendar_event_days_slot1"] = int(cal["event_name_1"].notna().sum())
    ctx.findings["calendar_event_days_slot2"] = int(cal["event_name_2"].notna().sum())
    ctx.findings["calendar_distinct_event_names"] = int(pairs["name"].nunique())
    return cal


# ----------------------------------------------------------------------------
# Sales
# ----------------------------------------------------------------------------
def load_and_audit_sales(path: Path, version: str, ctx: AuditContext) -> tuple[pd.DataFrame, list[str]]:
    reg, st = ctx.reg, "sales"
    header = read_header(path)
    id_ok = header[:6] == SALES_ID_COLUMNS
    d_cols = header[6:]
    d_pattern_ok = all(re.fullmatch(r"d_\d+", c) for c in d_cols)
    reg.add("H02", "provenance", f"{path.name}: id columns + d_* columns as expected", 1,
            int(not (id_ok and d_pattern_ok)))
    if not (id_ok and d_pattern_ok):
        raise SchemaError(f"{path.name}: unexpected header start {header[:8]}")
    d_nums = np.array([int(c[2:]) for c in d_cols])
    reg.add("S01", st, "Sales d columns are contiguous d_1..d_N", len(d_cols),
            int((d_nums != np.arange(1, len(d_cols) + 1)).sum()))

    df = pd.read_csv(path, dtype={c: str for c in SALES_ID_COLUMNS}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    n = len(df)
    ctx.findings["sales_rows"] = n
    ctx.findings["sales_day_columns"] = len(d_cols)

    ids = df[SALES_ID_COLUMNS]
    reg.add("S02", st, "Identifier fields non-missing (cells)", ids.size, int(ids.isna().values.sum()))
    reg.add("S03", st, "Key uniqueness: one row per id", n, int(_dup_rows(df, ["id"]).sum()))
    dup_series = _dup_rows(df, ["item_id", "store_id"])
    reg.add("S04", st, "Key uniqueness: one sales row per (item_id, store_id)", n, int(dup_series.sum()))
    if dup_series.any():
        ctx.add_exceptions(df.loc[dup_series, SALES_ID_COLUMNS], "duplicate_series_key", path.name)
    exp_id = df["item_id"] + "_" + df["store_id"] + "_" + version
    bad_id = df["id"] != exp_id
    reg.add("S05", st, f"id == item_id + '_' + store_id + '_{version}'", n, int(bad_id.sum()))
    if bad_id.any():
        ctx.add_exceptions(df.loc[bad_id, SALES_ID_COLUMNS], "id_inconsistent_with_keys", path.name)

    # ---- value validity --------------------------------------------------
    dtypes = df[d_cols].dtypes
    all_int = all(pd.api.types.is_integer_dtype(t) for t in dtypes)
    cells = n * len(d_cols)
    if all_int:
        vals = df[d_cols].to_numpy(dtype=np.int64)
        n_null = n_nonnum = n_nonfinite = n_nonint = 0
        neg_mask = vals < 0
        n_neg = int(neg_mask.sum())
        bad_cells = []
        if n_neg:
            r, c = np.nonzero(neg_mask)
            bad_cells.append(pd.DataFrame({"id": df["id"].to_numpy()[r], "d": np.array(d_cols)[c],
                                           "raw_value": vals[r, c].astype(str), "problem": "negative"}))
    else:
        n_null = n_nonnum = n_nonfinite = n_neg = n_nonint = 0
        bad_cells = []
        for c in d_cols:
            s = df[c]
            num = pd.to_numeric(s, errors="coerce") if not pd.api.types.is_numeric_dtype(s) else s.astype(float)
            m_null = s.isna()
            m_nonnum = s.notna() & num.isna()
            m_nonfin = num.notna() & ~np.isfinite(num)
            m_neg = num.notna() & (num < 0)
            m_nonint = num.notna() & np.isfinite(num) & (num != np.round(num))
            for mask, label in [(m_null, "missing"), (m_nonnum, "non_numeric"), (m_nonfin, "non_finite"),
                                (m_neg, "negative"), (m_nonint, "non_integer")]:
                k = int(mask.sum())
                if k:
                    bad_cells.append(pd.DataFrame({"id": df.loc[mask, "id"], "d": c,
                                                   "raw_value": s[mask].astype(str), "problem": label}))
            n_null += int(m_null.sum()); n_nonnum += int(m_nonnum.sum()); n_nonfinite += int(m_nonfin.sum())
            n_neg += int(m_neg.sum()); n_nonint += int(m_nonint.sum())
    reg.add("S06", st, "Daily sales non-missing (cells)", cells, n_null)
    reg.add("S07", st, "Daily sales numeric (cells)", cells, n_nonnum)
    reg.add("S08", st, "Daily sales finite (cells)", cells, n_nonfinite)
    reg.add("S09", st, "Daily sales nonnegative (cells)", cells, n_neg)
    reg.add("S10", st, "Daily sales whole numbers (cells)", cells, n_nonint)
    if bad_cells:
        bc = pd.concat(bad_cells, ignore_index=True)
        ctx.findings["sales_invalid_cells_total"] = len(bc)
        ctx.add_exceptions(bc.head(ctx.cfg.max_cell_exceptions), "invalid_sales_value", path.name)

    # ---- hierarchy consistency -----------------------------------------
    dept_from_item = df["item_id"].str.rsplit("_", n=1).str[0]
    cat_from_dept = df["dept_id"].str.rsplit("_", n=1).str[0]
    state_from_store = df["store_id"].str.split("_").str[0]
    for cid, desc, bad in [
        ("S11", "dept_id consistent with item_id prefix", df["dept_id"] != dept_from_item),
        ("S12", "cat_id consistent with dept_id prefix", df["cat_id"] != cat_from_dept),
        ("S13", "state_id consistent with store_id prefix", df["state_id"] != state_from_store),
    ]:
        reg.add(cid, st, desc, n, int(bad.sum()))
        if bad.any():
            ctx.add_exceptions(df.loc[bad, SALES_ID_COLUMNS], f"hierarchy_{cid}", path.name)
    item_map = df.groupby("item_id")[["dept_id", "cat_id"]].nunique()
    reg.add("S14", st, "Each item_id maps to one dept_id and one cat_id across stores",
            len(item_map), int((item_map.max(axis=1) > 1).sum()))
    store_map = df.groupby("store_id")["state_id"].nunique()
    reg.add("S15", st, "Each store_id maps to one state_id", len(store_map), int((store_map > 1).sum()))
    n_items, n_stores = df["item_id"].nunique(), df["store_id"].nunique()
    reg.add("S16", st, "Complete item x store grid (informational)", n_items * n_stores,
            n_items * n_stores - n, severity="info",
            notes=f"{n_items} items x {n_stores} stores vs {n} rows")
    ctx.findings.update({
        "sales_unique_items": n_items, "sales_unique_stores": n_stores,
        "sales_unique_series": int(df[["item_id", "store_id"]].drop_duplicates().shape[0]),
        "sales_unique_depts": int(df["dept_id"].nunique()), "sales_unique_cats": int(df["cat_id"].nunique()),
        "sales_unique_states": int(df["state_id"].nunique()),
        "sales_stores": sorted(df["store_id"].unique().tolist()),
    })
    return df, d_cols


def compare_overlapping_sales(eval_df: pd.DataFrame, eval_d: list[str], val_path: Path,
                              ctx: AuditContext) -> None:
    """Informational provenance check: validation history should equal the
    evaluation history on the overlapping days. The files are NOT appended."""
    val = pd.read_csv(val_path, dtype={c: str for c in SALES_ID_COLUMNS}, low_memory=False)
    val_d = [c for c in val.columns if c.startswith("d_")]
    common = [c for c in val_d if c in set(eval_d)]
    a = eval_df.set_index(["item_id", "store_id"])[common]
    b = val.set_index(["item_id", "store_id"])[common]
    shared = a.index.intersection(b.index)
    diff = (a.loc[shared].to_numpy() != b.loc[shared].to_numpy())
    ctx.reg.add("S17", "sales", "validation file equals evaluation file on overlapping days "
                "(cells; files are not appended)", diff.size, int(diff.sum()), severity="warning",
                notes=f"{len(common)} overlapping days; {len(shared)} shared series; "
                      f"{len(a.index.symmetric_difference(b.index))} series in only one file")
    ctx.findings["validation_overlap_days"] = len(common)
    ctx.findings["validation_overlap_mismatched_cells"] = int(diff.sum())


# ----------------------------------------------------------------------------
# Prices
# ----------------------------------------------------------------------------
def load_and_audit_prices(path: Path, ctx: AuditContext, cal: pd.DataFrame,
                          sales: pd.DataFrame) -> pd.DataFrame:
    reg, st = ctx.reg, "prices"
    names = resolve_header(read_header(path), EXPECTED_PRICE_COLUMNS, path.name, ctx, "H03")
    raw = pd.read_csv(path, header=0, names=names, dtype=str, keep_default_na=False, na_values=[""])
    n = len(raw)
    ctx.findings["price_rows"] = n

    reg.add("R01", st, "Price fields non-missing (cells)", raw.size, int(raw.isna().values.sum()))
    wk, bad_wk = _int_parse(raw["wm_yr_wk"])
    reg.add("R02", st, "wm_yr_wk parses as integer", n, int(bad_wk.sum()))
    price = pd.to_numeric(raw["sell_price"], errors="coerce")
    nonnum = raw["sell_price"].notna() & price.isna()
    nonfin = price.notna() & ~np.isfinite(price)
    nonpos = price.notna() & np.isfinite(price) & (price <= 0)
    reg.add("R03", st, "sell_price numeric", n, int(nonnum.sum()))
    reg.add("R04", st, "sell_price finite", n, int(nonfin.sum()))
    reg.add("R05", st, "sell_price strictly positive", n, int(nonpos.sum()))
    bad_price = nonnum | nonfin | nonpos | raw["sell_price"].isna() | bad_wk
    if bad_price.any():
        ctx.add_exceptions(raw.loc[bad_price].head(ctx.cfg.max_cell_exceptions), "invalid_price_record", path.name)

    key = ["store_id", "item_id", "wm_yr_wk"]
    dup = _dup_rows(raw, key)
    exact = raw.duplicated(keep=False)
    reg.add("R06", st, "Key uniqueness: one price per (store_id, item_id, wm_yr_wk)", n, int(dup.sum()))
    reg.add("R07", st, "Exact duplicate price rows", n, int(exact.sum()))
    reg.add("R08", st, "Conflicting duplicate keys (same key, different price)", n,
            int((dup & ~exact).sum()))
    if dup.any():
        ctx.add_exceptions(raw.loc[dup], "duplicate_price_key", path.name)

    cal_weeks = set(cal["wm_yr_wk"].dropna().astype(int).tolist())
    not_in_cal = wk.notna() & ~wk.isin(cal_weeks)
    reg.add("R09", st, "wm_yr_wk exists in calendar", n, int(not_in_cal.sum()))
    sales_pairs = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
    price_pairs = pd.MultiIndex.from_frame(raw[["store_id", "item_id"]].drop_duplicates())
    orphan = ~price_pairs.isin(sales_pairs)
    reg.add("R10", st, "(store_id, item_id) in price file exists in sales file", len(price_pairs),
            int(orphan.sum()))
    reg.add("R11", st, "Sales series with no price rows at all (informational)", len(sales_pairs),
            int((~sales_pairs.isin(price_pairs)).sum()), severity="info")
    ids = pd.concat([raw["store_id"], raw["item_id"]]).dropna()
    reg.add("R12", st, "No leading/trailing whitespace in price identifiers", len(ids),
            int((ids != ids.str.strip()).sum()))
    decimals = raw["sell_price"].str.split(".").str[1].str.len().fillna(0)
    reg.add("R13", st, "sell_price has <= 2 decimal places (informational)", n,
            int((decimals > 2).sum()), severity="info")

    # Evidence required for a header repair: positional columns must type-check.
    if any(c["source_file"] == path.name for c in ctx.corrections):
        if bad_wk.sum() or not_in_cal.sum() or nonnum.sum():
            raise SchemaError(f"{path.name}: header repair not supported by data type checks")
        for c in ctx.corrections:
            if c["source_file"] == path.name:
                c["evidence"] += (f" Verified: all {n:,} values in column 3 are integers that "
                                  f"exist as calendar wm_yr_wk; all values in column 4 are numeric.")

    prices = pd.DataFrame({
        "store_id": raw["store_id"], "item_id": raw["item_id"],
        "wm_yr_wk": wk.astype("Int64"), "sell_price": price,
    })
    ctx.findings["price_unique_series"] = int(len(price_pairs))
    ctx.findings["price_week_range"] = f"{int(wk.min())}..{int(wk.max())}"
    return prices
