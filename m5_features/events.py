"""Event-date table (both slots), marker inspection, dev-only event selection,
and day-level same-day (B) / event-relative phase (C) features.

Day-level features depend only on the calendar, which is assumed to be known
in advance (documented assumption). They are broadcast to rows by target day.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import SPAN_PAIRS, FeatureConfig

NONE = "none"
OUTSIDE = "outside"


def slug(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def marker_kind(name: str) -> str:
    if re.search(r"start", name, re.I):
        return "start_marker"
    if re.search(r"End$|\bEnd\b", name):
        return "end_marker"
    if re.search(r"Week\d", name):
        return "period_marker"
    return "single_day"


def event_date_table(cal: pd.DataFrame, test_start: int) -> pd.DataFrame:
    parts = []
    for slot in (1, 2):
        p = cal.loc[cal[f"event_name_{slot}"].notna(), ["d_num", "date", f"event_name_{slot}", f"event_type_{slot}"]]
        parts.append(p.set_axis(["d_num", "date", "event_name", "event_type"], axis=1).assign(slot=slot))
    t = pd.concat(parts).sort_values(["d_num", "slot"]).reset_index(drop=True)
    t["d_num"] = t["d_num"].astype(int)
    t["family"] = t["event_type"]
    t["marker_kind"] = t["event_name"].map(marker_kind)
    t["year"] = t["date"].dt.year
    t["period"] = np.where(t["d_num"] < test_start, "development", "final_test_or_later")
    same = t.groupby("d_num")["event_name"].agg(list)
    t["same_day_events"] = t["d_num"].map(lambda d: "|".join(same[d]))
    t["n_events_same_day"] = t["d_num"].map(same.str.len())
    t["other_events_within_pm7"] = [
        "; ".join(f"{r.event_name}({int(r.d_num) - d:+d})" for r in t[(abs(t.d_num - d) <= 7) & (t.event_name != n)].itertuples())
        for d, n in zip(t["d_num"], t["event_name"])]
    return t


def marker_inspection(ev: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, start, end, incl in SPAN_PAIRS:
        for y in sorted(ev["year"].unique()):
            s = ev[(ev.event_name == start) & (ev.year == y)]["d_num"]
            e = ev[(ev.event_name == end) & (ev.year == y)]["d_num"]
            if len(s) == 0 and len(e) == 0:
                continue
            rows.append({"span": name, "year": y, "start_marker": start, "end_marker": end,
                         "start_d": int(s.iloc[0]) if len(s) else None, "end_d": int(e.iloc[0]) if len(e) else None,
                         "span_days": (int(e.iloc[0]) - int(s.iloc[0]) + (1 if incl else 0)) if len(s) and len(e) else None,
                         "decision": "span formed from both listed markers" if len(s) and len(e) and int(e.iloc[0]) > int(s.iloc[0])
                         else "incomplete pair in calendar - span NOT formed; markers used as anchors only"})
    for n in sorted(ev.loc[ev.marker_kind != "single_day", "event_name"].unique()):
        if any(n in (p[1], p[2]) for p in SPAN_PAIRS):
            continue
        rows.append({"span": None, "year": None, "start_marker": n if marker_kind(n) == "start_marker" else None,
                     "end_marker": n if marker_kind(n) == "end_marker" else None,
                     "decision": f"{marker_kind(n)} without a listed partner: anchored at the listed date; "
                                 "true start/duration unknown and not inferred"})
    return pd.DataFrame(rows)


def select_events(T_dev: np.ndarray, ev: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dev-only screen. T_dev = store x day totals for days 1..dev_end ONLY.
    Log-ratio of each day vs. the mean of the same weekday 4-7 weeks earlier."""
    n_days = T_dev.shape[1]
    ks = np.arange(cfg.k_min, cfg.k_max + 1)
    bw = np.array(cfg.baseline_weeks)
    rows, prof = [], []
    for name, g in ev.groupby("event_name"):
        occ = [int(d) - 1 for d in g["d_num"].unique()
               if int(d) - 1 + cfg.k_max < n_days and int(d) - 1 + cfg.k_min - 7 * bw.max() >= 0]
        eff = {p: np.nan for p, _, _ in cfg.phase_windows}
        if occ:
            lr = np.empty((len(occ), T_dev.shape[0], len(ks)))
            for i, E in enumerate(occ):
                for j, k in enumerate(ks):
                    t = E + k
                    base = T_dev[:, t - 7 * bw].mean(axis=1)
                    lr[i, :, j] = np.log((T_dev[:, t] + 1.0) / (base + 1.0))
            for p, a, b in cfg.phase_windows:
                sel = (ks >= a) & (ks <= b)
                eff[p] = float(np.median(lr[:, :, sel].mean(axis=2)))
            med_k = np.median(lr.reshape(-1, len(ks)), axis=0)
            for k, v in zip(ks, med_k):
                prof.append({"event_name": name, "k": int(k), "median_log_ratio": float(v), "n_occurrences": len(occ)})
        max_abs = np.nanmax(np.abs(list(eff.values()))) if occ else np.nan
        if name in cfg.forced_named_events:
            dec, why = "named", "prespecified thesis event"
        elif len(occ) >= cfg.min_dev_occurrences and max_abs >= cfg.min_abs_effect:
            dec, why = "named", f">= {cfg.min_dev_occurrences} dev occurrences and max |phase effect| >= {cfg.min_abs_effect}"
        else:
            dec = "pooled"
            why = (f"only {len(occ)} usable dev occurrences" if len(occ) < cfg.min_dev_occurrences
                   else f"max |phase effect| {max_abs:.3f} < {cfg.min_abs_effect}")
        rows.append({"event_name": name, "event_type": g["event_type"].iloc[0], "family": g["family"].iloc[0],
                     "marker_kind": g["marker_kind"].iloc[0], "n_dev_occurrences_used": len(occ),
                     **{f"eff_{p}": v for p, v in eff.items()}, "max_abs_phase_effect": max_abs,
                     "decision": dec, "reason": why})
    return pd.DataFrame(rows), pd.DataFrame(prof)


def _phase_of(k: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    out = np.full(k.shape, OUTSIDE, dtype=object)
    for p, a, b in cfg.phase_windows:
        out[(k >= a) & (k <= b)] = p
    return out


def _nearest(t: np.ndarray, occ: np.ndarray, cfg: FeatureConfig):
    """Signed k = t - E to the nearest occurrence; ties prefer the upcoming
    event (k < 0). Returns (k within window else NaN, count of occurrences in window)."""
    if len(occ) == 0:
        return np.full(len(t), np.nan), np.zeros(len(t), dtype=np.int16), np.zeros(len(t), dtype=np.int16)
    rel = t[:, None] - occ[None, :]
    score = np.abs(rel) * 2 + (rel > 0)
    k = rel[np.arange(len(t)), score.argmin(axis=1)].astype(float)
    inwin = (rel >= cfg.k_min) & (rel <= cfg.k_max)
    k[(k < cfg.k_min) | (k > cfg.k_max)] = np.nan
    return k, inwin.sum(axis=1).astype(np.int16), (rel == 0).sum(axis=1).astype(np.int16)


def day_level_event_features(cal: pd.DataFrame, ev: pd.DataFrame, sel: pd.DataFrame, cfg: FeatureConfig):
    t = cal["d_num"].to_numpy().astype(int)
    named = sel.loc[sel.decision == "named", "event_name"].tolist()
    pooled = sel.loc[sel.decision == "pooled"]
    families = sorted(sel["family"].unique())

    B = pd.DataFrame({"d_num": t})
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        B[f"t_{c}"] = cal[c].fillna(NONE).to_numpy()
    B["t_event_count"] = (cal["event_name_1"].notna().astype(int) + cal["event_name_2"].notna().astype(int)).astype("int8").to_numpy()
    B["t_no_event"] = (B["t_event_count"] == 0).astype("int8")

    C = pd.DataFrame({"d_num": t})
    represented = np.zeros(len(t), dtype=np.int16)
    for e in named:
        occ = np.unique(ev.loc[ev.event_name == e, "d_num"].to_numpy())
        k, _, at0 = _nearest(t, occ, cfg)
        C[f"k_{slug(e)}"] = k
        C[f"ph_{slug(e)}"] = _phase_of(np.nan_to_num(k, nan=99), cfg)
        represented += at0
    for f in families:
        names = pooled.loc[pooled.family == f, "event_name"].tolist()
        occ = np.sort(ev.loc[ev.event_name.isin(names), "d_num"].to_numpy())  # keep duplicates (same-day pooled events)
        k, n_in, at0 = _nearest(t, occ, cfg)
        C[f"k_fam_{slug(f)}"] = k
        C[f"ph_fam_{slug(f)}"] = _phase_of(np.nan_to_num(k, nan=99), cfg)
        C[f"n_fam_{slug(f)}_in_window"] = n_in
        represented += at0
    for name, start, end, incl in SPAN_PAIRS:
        col = np.zeros(len(t), dtype=np.int8)
        for y in ev["year"].unique():
            s = ev[(ev.event_name == start) & (ev.year == y)]["d_num"]
            e_ = ev[(ev.event_name == end) & (ev.year == y)]["d_num"]
            if len(s) and len(e_) and int(e_.iloc[0]) > int(s.iloc[0]):
                hi = int(e_.iloc[0]) + (0 if incl else -1)
                col[(t >= int(s.iloc[0])) & (t <= hi)] = 1
        C[f"span_{name}"] = col
    rep_check = {"days_with_events": int((B["t_event_count"] > 0).sum()),
                 "days_with_two_events": int((B["t_event_count"] == 2).sum()),
                 "listed_event_day_occurrences": int(B["t_event_count"].sum()),
                 "occurrences_represented_in_C": int(represented.sum()),
                 "days_where_C_misses_an_event": int((represented != B["t_event_count"].to_numpy()).sum())}
    return B, C, named, families, rep_check
