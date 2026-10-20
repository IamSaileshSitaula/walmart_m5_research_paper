"""Render the data-quality report and the pre-modeling checklist (Markdown)."""
from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int) or (hasattr(v, "dtype") and str(getattr(v, "dtype", "")).startswith("int")):
            return f"{int(v):,}"
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def checklist_rows(reg_rows: list[dict]) -> list[dict]:
    by = {r["check_id"]: r for r in reg_rows}

    def status_of(ids):
        sts = [by[i]["status"] for i in ids if i in by]
        missing = [i for i in ids if i not in by]
        if missing:
            return "NOT RUN", f"missing checks: {missing}"
        if any(s == "FAIL" for s in sts):
            return "FAIL", ", ".join(i for i in ids if by[i]["status"] == "FAIL")
        if any(s == "FAIL->REPAIRED" for s in sts):
            return "PASS (after logged repair)", ", ".join(i for i in ids if by[i]["status"] == "FAIL->REPAIRED")
        if any(s == "WARN" for s in sts):
            return "PASS with warnings", ", ".join(i for i in ids if by[i]["status"] == "WARN")
        return "PASS", ""

    groups = [
        ("Raw files preserved, provenance + schema recorded", ["P01", "P02", "P03", "H01", "H02", "H03"]),
        ("Key uniqueness (sales item-store, calendar d/date, price store-item-week)",
         ["S03", "S04", "C05", "C06", "R06", "R07", "R08"]),
        ("Value validity (sales whole & nonneg, price positive & finite, SNAP 0/1, hierarchy, event pairing)",
         ["S06", "S07", "S08", "S09", "S10", "R01", "R03", "R04", "R05", "C15_CA", "C15_TX", "C15_WI",
          "S11", "S12", "S13", "S14", "S15", "C16_1", "C16_2", "C17", "C18", "C19"]),
        ("Exact calendar matching (contiguous d, exact d<->date, weekday/week rules, no unmatched sales days)",
         ["C07", "C08", "C09", "C10", "C11", "C13", "S01", "J04"]),
        ("Row counts and unit totals unchanged after each join (and in the written Parquet)",
         ["J01", "J02", "J03", "J05", "J06", "J07"]),
        ("Missingness classified (price status complete, raw price never filled, event blank vs join failure)",
         ["M01", "M02", "M03", "M04"]),
        ("Holiday anomaly review (flags written, no silent edits)", ["A01", "A02", "A03", "A04"]),
        ("Feature availability at each forecast origin (no leakage)", ["L01", "L02", "L03", "L04", "L05"]),
    ]
    out = []
    for name, ids in groups:
        s, note = status_of(ids)
        if name.startswith("Holiday anomaly") and s.startswith("PASS"):
            s = "PASS (automated) - HUMAN REVIEW PENDING"
        out.append({"item": name, "status": s, "checks": ", ".join(ids), "notes": note})
    return out
