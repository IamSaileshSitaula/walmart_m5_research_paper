"""End-to-end runner: provenance -> audit -> panel -> features -> folds -> report.

Usage (from the repository root):
    python -m m5_cleaning.run_pipeline --data-dir data --out-dir outputs/m5_cleaning

The pipeline never writes to --data-dir. It does NOT train a model and does
NOT run any inventory simulation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .anomalies import context_label  # noqa: F401  (re-exported for notebooks)
from .audit import (AuditContext, BlockingAuditFailure, SchemaError, compare_overlapping_sales,
                    load_and_audit_calendar, load_and_audit_prices, load_and_audit_sales)
from .config import CALENDAR_FILE, PIPELINE_VERSION, PRICES_FILE, SALES_FILES, Config
from .features import calendar_event_features, event_relative_long, lag_availability_table
from .folds import build_fold_manifest, check_fold_manifest
from .io_utils import RunLogger, environment_record, file_record, sha256_file, write_json
from .panel import PRICE_STATUS, build_panel, load_corrections
from .report import checklist_rows, md_table

BLOCKING = ["S03", "S04", "S06", "S07", "S08", "S09", "S10", "C05", "C06", "C07", "C09",
            "R02", "R03", "R04", "R05", "R06", "R09"]


def locate_files(data_dir: Path) -> dict:
    found = {"calendar": data_dir / CALENDAR_FILE, "prices": data_dir / PRICES_FILE}
    for version, fname in SALES_FILES:
        found[f"sales_{version}"] = data_dir / fname
    return {k: (p if p.exists() else None) for k, p in found.items()} | {"_expected": found}


def write_missing_files_note(out_dir: Path, files: dict) -> None:
    exp = files["_expected"]
    lines = ["# Required M5 files not found", "",
             "The pipeline was created but NOT run on data. Place these files (unmodified Kaggle",
             "'M5 Forecasting - Accuracy' release) at exactly these paths, then rerun:", ""]
    lines.append(f"- `{exp['calendar']}` - required")
    lines.append(f"- `{exp['prices']}` - required")
    lines.append(f"- `{exp['sales_evaluation']}` - preferred (d_1..d_1941)")
    lines.append(f"- `{exp['sales_validation']}` - fallback only if the evaluation file is absent")
    lines += ["", "Status:"] + [f"- {k}: {'FOUND' if v else 'MISSING'}" for k, v in files.items() if k != "_expected"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MISSING_FILES.md").write_text("\n".join(lines), encoding="utf-8")


def build_day_context(cal: pd.DataFrame, window: int) -> dict:
    n = len(cal)
    events = np.array(["|".join(x for x in (a, b) if isinstance(x, str))
                       for a, b in zip(cal["event_name_1"], cal["event_name_2"])], dtype=object)
    lg = event_relative_long(cal, window)
    lg = lg.assign(abs_rel=lg["rel_day"].abs()).sort_values(["d_num", "abs_rel", "rel_day"])
    nearest = lg.drop_duplicates("d_num").set_index("d_num")
    near_name = np.full(n, None, dtype=object)
    near_rel = np.full(n, np.nan)
    idx = nearest.index.to_numpy().astype(int) - 1
    near_name[idx] = nearest["event_name"].to_numpy()
    near_rel[idx] = nearest["rel_day"].to_numpy()
    return {"events": events, "near_name": near_name, "near_rel": near_rel}


def scan_panel(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    agg = {"rows": 0, "units_raw": 0, "zero_rows": 0, "price_nulls": 0, "unlogged_changes": 0,
           "row_groups": pf.num_row_groups, "unpriced_status_rows": 0}
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["units_sold_raw", "units_sold", "units_corrected_flag",
                                          "sell_price_raw", "price_status"])
        u, uc = t["units_sold_raw"], t["units_sold"]
        agg["rows"] += t.num_rows
        agg["units_raw"] += int(pc.sum(u).as_py() or 0)
        agg["zero_rows"] += int(pc.sum(pc.equal(u, 0)).as_py() or 0)
        agg["price_nulls"] += t["sell_price_raw"].null_count
        changed = pc.and_(pc.not_equal(u, uc), pc.invert(t["units_corrected_flag"]))
        agg["unlogged_changes"] += int(pc.sum(changed).as_py() or 0)
        agg["unpriced_status_rows"] += int(pc.sum(pc.not_equal(t["price_status"], "listed")).as_py() or 0)
    agg["file_size_bytes"] = path.stat().st_size
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Folder containing the raw M5 CSV files")
    ap.add_argument("--out-dir", default="outputs/m5_cleaning")
    ap.add_argument("--corrections", default=str(Path(__file__).parent / "corrections" / "sales_corrections.csv"))
    ap.add_argument("--skip-overlap-check", action="store_true",
                    help="Skip the informational validation-vs-evaluation overlap comparison")
    args = ap.parse_args(argv)

    data_dir, out_dir = Path(args.data_dir).resolve(), Path(args.out_dir).resolve()
    if out_dir == data_dir:
        raise SystemExit("--out-dir must differ from --data-dir (raw data is never written to)")
    cfg = Config(data_dir=data_dir, out_dir=out_dir, corrections_file=Path(args.corrections))
    for sub in ["panel", "features", "review", "audit", "folds"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    log = RunLogger(out_dir / "run_log.txt")
    t0 = time.time()
    env = environment_record()
    log(f"M5 cleaning pipeline v{PIPELINE_VERSION} | data={data_dir} | out={out_dir}")

    # ------------------------------------------------------------------ files
    files = locate_files(data_dir)
    sales_choice = next(((v, files[f"sales_{v}"]) for v, _ in SALES_FILES if files[f"sales_{v}"]), None)
    if not files["calendar"] or not files["prices"] or sales_choice is None:
        write_missing_files_note(out_dir, files)
        log("Required files missing - see MISSING_FILES.md. No data was cleaned.")
        return 2
    version, sales_path = sales_choice
    ctx = AuditContext(cfg=cfg, log=log)
    ctx.reg.add("P01", "provenance", "Required files present (calendar, prices, one sales file)", 3, 0)
    ctx.reg.add("P02", "provenance", "Exactly one sales file used (evaluation preferred); files never appended",
                1, 0, notes=f"used {sales_path.name} (version={version})")

    log("Hashing raw files (read-only)...")
    prov = []
    for key in ["calendar", "prices", "sales_evaluation", "sales_validation"]:
        if files[key]:
            used = key in ("calendar", "prices") or key == f"sales_{version}"
            rec = file_record(files[key], key, used)
            if key.startswith("sales"):
                rec["sales_file_version"] = key.split("_")[1]
            prov.append(rec)
    write_json({"environment": env, "files": prov}, out_dir / "audit" / "file_provenance.json")

    # ------------------------------------------------------------------ audit
    try:
        log("Auditing calendar...")
        cal = load_and_audit_calendar(files["calendar"], ctx)
        log(f"Auditing {sales_path.name}...")
        sales, d_cols = load_and_audit_sales(sales_path, version, ctx)
        if version == "evaluation" and files["sales_validation"] and not args.skip_overlap_check:
            log("Comparing overlapping history with validation file (informational, not appended)...")
            compare_overlapping_sales(sales, d_cols, files["sales_validation"], ctx)
        log("Auditing sell_prices...")
        prices = load_and_audit_prices(files["prices"], ctx, cal, sales)
    except SchemaError as e:
        log(f"SCHEMA ERROR: {e}")
        pd.DataFrame(ctx.reg.rows).to_csv(out_dir / "audit" / "audit_checks.csv", index=False)
        return 3

    blocking = [r for r in ctx.reg.rows if r["check_id"] in BLOCKING and r["status"] == "FAIL"]
    if blocking:
        log(f"BLOCKING audit failures: {[r['check_id'] for r in blocking]} - panel NOT built.")
        pd.DataFrame(ctx.reg.rows).to_csv(out_dir / "audit" / "audit_checks.csv", index=False)
        if ctx.exceptions:
            pd.concat(ctx.exceptions, ignore_index=True).astype(str).to_csv(
                out_dir / "review" / "exceptions.csv", index=False)
        return 4

    raw_units_total = int(sales[d_cols].to_numpy(dtype=np.int64).sum())
    raw_zero_cells = int((sales[d_cols].to_numpy() == 0).sum())
    n_series, n_days = len(sales), len(d_cols)

    # ------------------------------------------------------------------ folds
    last_d = int(d_cols[-1][2:])
    folds = build_fold_manifest(cal, last_d, cfg)
    fold_chk = check_fold_manifest(folds, last_d)

    # ------------------------------------------------------------------ panel
    corrections = load_corrections(cfg.corrections_file)
    log(f"Documented sales corrections supplied: {len(corrections)}")
    day_ctx = build_day_context(cal, cfg.context_event_window)
    panel_path = out_dir / "panel" / "m5_daily_item_store_clean.parquet"
    log("Building daily item-store panel store by store...")
    res = build_panel(sales, d_cols, cal, prices, corrections, folds, day_ctx, panel_path, ctx)
    st = res["stats"]
    del sales
    log("Verifying written Parquet...")
    scan = scan_panel(panel_path)

    # ------------------------------------------------------------------ join / missingness / anomaly checks
    R = ctx.reg
    R.add("J01", "join", "Rows unchanged after calendar left-join", st["rows_before"],
          abs(st["rows_after_calendar"] - st["rows_before"]))
    R.add("J02", "join", "Units sold unchanged after calendar left-join", st["units_before"],
          abs(st["units_after_calendar"] - st["units_before"]), notes="n_checked = total units")
    R.add("J03", "join", "Rows unchanged after price left-join", st["rows_before"],
          abs(st["rows_after_price"] - st["rows_before"]))
    R.add("J05", "join", "Units sold unchanged after price left-join", st["units_before"],
          abs(st["units_after_price"] - st["units_before"]), notes="n_checked = total units")
    R.add("J04", "join", "Sales rows with no calendar match (unmatched calendar keys, sales side)",
          st["rows_before"], st["calendar_unmatched_rows"])
    persisted_fail = int(scan["rows"] != n_series * n_days) + int(scan["units_raw"] != raw_units_total)
    R.add("J06", "join", "Written Parquet: rows == series x days and units == raw total", 2, persisted_fail,
          notes=f"rows={scan['rows']:,}; expected={n_series * n_days:,}; units={scan['units_raw']:,}; raw={raw_units_total:,}")
    dup_keys = ctx.reg.get("C05")["n_failures"] + ctx.reg.get("R06")["n_failures"]
    R.add("J07", "join", "Duplicate join keys (calendar d, price store-item-week)",
          ctx.reg.get("C05")["n_checked"] + ctx.reg.get("R06")["n_checked"], dup_keys)
    R.add("J08", "join", "Calendar days with no sales rows (expected: post-sample days)", len(cal),
          ctx.findings["calendar_days_without_sales"], severity="info",
          notes=ctx.findings["calendar_days_without_sales_range"])
    R.add("J09", "join", "Price rows unmatched within the sales date span", ctx.findings["price_rows"],
          st["price_rows_unmatched_within_sales_span"])
    R.add("J10", "join", "Price rows unmatched because their week is after the last sales day", ctx.findings["price_rows"],
          st["price_rows_unmatched_after_sales_end"], severity="info",
          notes=f"last sales week wm_yr_wk={res['last_sales_wk']}")
    R.add("J11", "join", "Positive-sales days with no listed price", st["rows_before"],
          st["positive_sales_no_price_rows"], severity="warning",
          notes=f"units on those days={st['positive_sales_no_price_units']:,}")

    ps = res["price_status"]
    R.add("M01", "missingness", "Every panel row has exactly one price status (none unclassified)",
          st["rows_before"], abs(int(ps["daily_rows"].sum()) - st["rows_before"]) + int(ps.loc[ps.price_status == "calendar_join_failed", "daily_rows"].iloc[0]))
    R.add("M02", "missingness", "Raw price never filled: Parquet null prices == unpriced-status rows",
          scan["rows"], abs(scan["price_nulls"] - scan["unpriced_status_rows"]),
          notes=f"null sell_price_raw={scan['price_nulls']:,}")
    R.add("M03", "missingness", "Blank events only after successful calendar join (no join failures)",
          st["rows_before"], st["calendar_unmatched_rows"])
    R.add("M04", "missingness", "Recorded zero sales preserved (zeros in Parquet == zeros in raw file)",
          raw_zero_cells, abs(scan["zero_rows"] - raw_zero_cells))

    R.add("A01", "anomaly", "Extreme item-store-day values flagged for review (retained unchanged)",
          st["rows_before"], st["extreme_flags"], severity="info")
    R.add("A02", "anomaly", "Store-days with collapsed totals flagged for review (retained unchanged)",
          n_days * res["per_store"].shape[0], st["store_day_low"], severity="info")
    R.add("A03", "anomaly", "units_sold differs from units_sold_raw only where a logged correction exists",
          scan["rows"], scan["unlogged_changes"])
    bad_corr = sum(1 for c in ctx.corrections if not str(c.get("evidence", "")).strip() or not str(c.get("reason", "")).strip())
    R.add("A04", "anomaly", "Every correction logged with original, revised, reason, evidence",
          len(ctx.corrections), bad_corr, notes=f"{st['corrections_applied']} sales-value corrections applied")

    av = res["availability"]
    R.add("L01", "leakage", "Origin-anchored features + last-known price invariant to post-origin data "
          "(every included fold x store)", len(av), int((~av["invariant_to_future"]).sum()))
    R.add("L02", "leakage", "Negative control: target-relative lag-7 at h=8 detected as post-origin",
          len(av), int((~av["negative_control_lag7_h8_detected"].astype(bool)).sum()))
    lag_tab = lag_availability_table(cfg)
    R.add("L03", "leakage", f"All proposed target-relative lags >= horizon ({cfg.horizon})",
          len(cfg.proposed_target_lags), int(sum(l < cfg.horizon for l in cfg.proposed_target_lags)))
    R.add("L04", "leakage", "Fold chronology: train_end < forecast_start, 28-day windows within observed data, "
          "early-stopping window inside training", fold_chk["n_folds_included"],
          fold_chk["train_end_before_forecast_start_failures"] + fold_chk["forecast_window_length_failures"]
          + fold_chk["forecast_beyond_observed_failures"] + fold_chk["early_stop_outside_train_failures"])
    R.add("L05", "leakage", "No random splits; every fold records a fit-on-d<=origin rule",
          fold_chk["n_folds_included"], int(folds.loc[folds.included, "fit_rule"].isna().sum()) + int(fold_chk["random_split_used"]))

    # ------------------------------------------------------------------ raw untouched?
    log("Re-hashing raw files to confirm they are unchanged...")
    changed = sum(1 for r in prov if sha256_file(Path(r["absolute_path"])) != r["sha256"])
    R.add("P03", "provenance", "Raw files byte-identical before and after the run (SHA-256)", len(prov), changed)

    # ------------------------------------------------------------------ features (separate stage)
    log("Writing calendar/event features (feature-engineering stage)...")
    cf = calendar_event_features(cal, cfg)
    cf.to_parquet(out_dir / "features" / "calendar_event_features.parquet", index=False)
    cf.to_csv(out_dir / "features" / "calendar_event_features.csv", index=False)
    for w in (cfg.event_window_main, cfg.event_window_sensitivity):
        lg = event_relative_long(cal, w)
        lg.to_parquet(out_dir / "features" / f"event_relative_days_pm{w}.parquet", index=False)
    lag_tab.to_csv(out_dir / "features" / "lag_availability.csv", index=False)

    # ------------------------------------------------------------------ write audit / review outputs
    checks = pd.DataFrame(R.rows)
    checks.to_csv(out_dir / "audit" / "audit_checks.csv", index=False)
    folds.drop(columns=["_T"]).to_csv(out_dir / "folds" / "fold_manifest.csv", index=False)
    av.to_csv(out_dir / "folds" / "feature_availability_by_origin.csv", index=False)
    ps.to_csv(out_dir / "audit" / "price_status_summary.csv", index=False)
    res["gaps"].to_csv(out_dir / "review" / "price_gaps_within_listed_span.csv", index=False)
    res["per_store"].to_csv(out_dir / "audit" / "per_store_summary.csv", index=False)
    zt = res["zero_table"]
    zt.to_csv(out_dir / "audit" / "zero_sales_breakdown.csv", index=False)

    corr_df = pd.DataFrame(ctx.corrections, columns=["correction_id", "level", "source_file", "record_key", "field",
                                                     "original_value", "revised_value", "values_changed", "reason", "evidence"])
    corr_df.to_csv(out_dir / "review" / "corrections_log.csv", index=False)
    excl = pd.DataFrame(columns=["exclusion_id", "level", "record_key", "reason", "evidence", "rows_excluded"])
    excl.to_csv(out_dir / "review" / "exclusions_log.csv", index=False)

    exc = pd.concat(ctx.exceptions, ignore_index=True) if ctx.exceptions else pd.DataFrame(columns=["exception_type"])
    exc.insert(len(exc.columns), "review_status", "pending_human_review")
    for c in exc.columns:
        if exc[c].dtype == object:
            exc[c] = exc[c].astype("string")
    exc.to_parquet(out_dir / "review" / "exceptions.parquet", index=False)
    exc.to_csv(out_dir / "review" / "exceptions.csv", index=False)
    exc_summary = exc.groupby("exception_type").size().rename("rows").reset_index()

    # ------------------------------------------------------------------ report
    elapsed = time.time() - t0
    findings = ctx.findings | {"panel_rows": scan["rows"], "panel_units": scan["units_raw"],
                               "panel_file_bytes": scan["file_size_bytes"], "row_groups": scan["row_groups"],
                               "elapsed_seconds": round(elapsed, 1)}
    write_json({"pipeline_version": PIPELINE_VERSION, "config": cfg.to_dict(), "environment": env,
                "findings": findings, "join_stats": st, "fold_checks": fold_chk,
                "sales_file_used": sales_path.name}, out_dir / "run_manifest.json")
    write_report(out_dir, cfg, prov, version, sales_path, checks, findings, st, ps, zt, exc, exc_summary,
                 res, folds, fold_chk, av, lag_tab, corr_df, scan, cal)
    n_fail = int((checks["status"] == "FAIL").sum())
    log(f"Done in {elapsed:,.0f}s. Checks: {len(checks)} | FAIL: {n_fail}. Outputs in {out_dir}")
    log.close()
    return 0 if n_fail == 0 else 1


def _rate(df, by):
    g = df.groupby(by, observed=True)[["rows", "zero_rows", "units"]].sum().reset_index()
    g["zero_sales_rate"] = g["zero_rows"] / g["rows"]
    return g


def write_report(out_dir, cfg, prov, version, sales_path, checks, f, st, ps, zt, exc, exc_summary,
                 res, folds, fold_chk, av, lag_tab, corr_df, scan, cal):
    L = []
    a = L.append
    a("# M5 daily item-store cleaning - data-quality report\n")
    a(f"Pipeline v{PIPELINE_VERSION}. Generated by `python -m m5_cleaning.run_pipeline`. "
      "No forecasting model was trained and no inventory simulation was run.\n")
    a("> **Scope caveat.** M5 records *units sold*, not unconstrained demand, and contains no verified "
      "on-hand inventory or historical stockout labels. Recorded zeros are zero units sold, not zero "
      "demand. Any later inventory results must be labelled **simulated**.\n")

    a("## Part A - Actual findings (measured in this run)\n")
    a("### A1. Raw file provenance (read-only; never overwritten)\n")
    a(md_table(pd.DataFrame([{
        "file": r["filename"], "used": r["used_in_pipeline"], "version": r.get("sales_file_version", ""),
        "size_bytes": r["size_bytes"], "last_modified": r["last_modified_local"],
        "n_columns": r["n_columns"], "sha256": r["sha256"]} for r in prov])))
    a(f"\nSales file used: **{sales_path.name}** (version `{version}`). The other sales file is hashed for "
      "provenance only; histories overlap, so the files are never appended. Full column lists: "
      "`audit/file_provenance.json`.\n")
    hdr = [r for r in prov if r["filename"] == PRICES_FILE][0]
    a(f"\n**Header exception:** `{PRICES_FILE}` header as stored = `{','.join(hdr['column_names'])}`. "
      "See correction `HDR-sell_prices.csv` in `review/corrections_log.csv`.\n")

    a("### A2. Coverage\n")
    cov = pd.DataFrame([
        ("Sales rows (series)", f["sales_rows"]), ("Unique item_id", f["sales_unique_items"]),
        ("Unique store_id", f["sales_unique_stores"]), ("Unique item-store series", f["sales_unique_series"]),
        ("Departments / categories / states", f"{f['sales_unique_depts']} / {f['sales_unique_cats']} / {f['sales_unique_states']}"),
        ("Sales day columns", f["sales_day_columns"]),
        ("Sales date coverage", f"{cal['date'].iloc[0].date()} .. {cal['date'].iloc[f['sales_day_columns'] - 1].date()}"),
        ("Calendar rows / coverage", f"{f['calendar_rows']} / {f['calendar_first_date']} .. {f['calendar_last_date']}"),
        ("Calendar days without sales", f"{f['calendar_days_without_sales']} ({f['calendar_days_without_sales_range']})"),
        ("Price rows / series / week range", f"{f['price_rows']:,} / {f['price_unique_series']:,} / {f['price_week_range']}"),
        ("Daily panel rows", scan["rows"]), ("Total units sold (raw)", scan["units_raw"]),
        ("Panel Parquet size (bytes) / row groups", f"{scan['file_size_bytes']:,} / {scan['row_groups']}"),
    ], columns=["measure", "value"])
    a(md_table(cov))

    a("\n### A3. Every audit check (including zero-failure checks)\n")
    a(md_table(checks[["check_id", "stage", "description", "n_checked", "n_failures", "status", "notes"]]))

    a("\n### A4. Join coverage\n")
    a(md_table(pd.DataFrame([
        ("Rows before joins", st["rows_before"]), ("Rows after calendar join", st["rows_after_calendar"]),
        ("Rows after price join", st["rows_after_price"]), ("Units before joins", st["units_before"]),
        ("Units after calendar join", st["units_after_calendar"]), ("Units after price join", st["units_after_price"]),
        ("Sales rows with unmatched calendar key", st["calendar_unmatched_rows"]),
        ("Price rows matched to >=1 sales day", st["price_rows_matched"]),
        ("Price rows unmatched (total)", st["price_rows_unmatched"]),
        ("  - week after last sales day", st["price_rows_unmatched_after_sales_end"]),
        ("  - within sales span", st["price_rows_unmatched_within_sales_span"]),
        ("Panel rows with listed price", int(ps.loc[ps.price_status == "listed", "daily_rows"].iloc[0])),
        ("Positive-sales days with no listed price", st["positive_sales_no_price_rows"]),
    ], columns=["measure", "value"])))

    a("\n### A5. Missing weekly prices (descriptive categories, not launch/stockout evidence)\n")
    ps2 = ps.copy()
    ps2["share_of_daily_rows"] = ps2["daily_rows"] / ps2["daily_rows"].sum()
    a(md_table(ps2))
    g = res["gaps"]
    a(f"\nWithin-span gaps: {len(g):,} series-weeks across {g[['store_id', 'item_id']].drop_duplicates().shape[0]:,} "
      f"series; units sold in those weeks = {int(g['units_sold_raw'].sum()) if len(g) else 0:,} "
      "(list: `review/price_gaps_within_listed_span.csv`).\n")

    a("\n### A6. Zero-sales rates (zero units sold; not zero demand)\n")
    tot = zt[["rows", "zero_rows"]].sum()
    a(f"Overall: {int(tot['zero_rows']):,} of {int(tot['rows']):,} item-store-days = **{tot['zero_rows'] / tot['rows']:.2%}**.\n")
    for by in (["cat_id"], ["price_status"], ["event_status"], ["store_id"]):
        a(f"\nBy {by[0]}:\n")
        a(md_table(_rate(zt, by)))

    a("\n### A7. Anomaly review flags (no values changed)\n")
    ex = exc[exc["exception_type"] == "extreme_sales_review"]
    if len(ex):
        prim = ex["review_context"].astype(str).str.split(";").str[0].str.split("[").str[0].str.strip()
        a(f"Extreme item-store-day values flagged: **{len(ex):,}** "
          f"({int((ex['isolated_spike'].astype(str) == 'True').sum()):,} isolated spikes).\n")
        a(md_table(prim.value_counts().rename_axis("primary_context").reset_index(name="flags")))
        hol = ex[ex["nearest_event_pm7"].notna() & ex["nearest_event_pm7"].astype(str).isin(cfg.focus_holidays)]
        a("\nFlags within +/-7 days of focus holidays:\n")
        a(md_table(hol.groupby("nearest_event_pm7").size().rename("flags").reset_index()))
    sd = exc[exc["exception_type"] == "store_day_low_total"]
    a(f"\nStore-days with total units < {cfg.store_day_low_ratio:.0%} of the +/-{cfg.anomaly_local_halfwidth}-day "
      f"median: **{len(sd):,}**.\n")
    if len(sd):
        a(md_table(sd.groupby(["date", "events_on_day"], dropna=False).agg(
            stores=("store_id", "nunique"), store_ids=("store_id", lambda s: ",".join(sorted(s.astype(str))))).reset_index()))

    a("\n### A8. Corrections and exclusions\n")
    a(md_table(corr_df))
    a("\nExclusions: **none** (no rows were dropped). Exceptions by type:\n")
    a(md_table(exc_summary))

    a("\n### A9. Forecast-origin feature availability\n")
    a(f"{len(av):,} fold x store tests; future-invariant: {int(av['invariant_to_future'].sum()):,}; "
      f"negative control detected: {int(av['negative_control_lag7_h8_detected'].astype(bool).sum()):,}.\n")
    a(md_table(lag_tab))

    a("\n### A10. Chronological fold manifest\n")
    a(f"{fold_chk['n_folds_included']} included / {fold_chk['n_folds_total']} candidate folds "
      f"(full table: `folds/fold_manifest.csv`).\n")
    a(md_table(folds[["fold_id", "target_event", "event_date", "origin_date", "forecast_start_date",
                      "forecast_end_date", "event_horizon", "included", "exclusion_reason"]]))

    a("\n## Part B - Proposed rules and assumptions (not findings)\n")
    a("See `docs/decision_log.md` (rules D01-D20) and `docs/data_dictionary.md`. Key assumptions: "
      "calendar events are treated as known in advance; listed weekly prices are not assumed to be a known "
      "price plan; anomaly flags are full-history audit aids and are not model features.\n")

    a("\n## Pre-modeling checklist\n")
    cl = pd.DataFrame(checklist_rows(checks.to_dict("records")))
    a(md_table(cl))
    (out_dir / "data_quality_report.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "pre_modeling_checklist.md").write_text(
        "# Pre-modeling checklist\n\n" + md_table(cl) +
        "\nHoliday anomaly flags require human review before modeling; nothing was auto-corrected.\n",
        encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
