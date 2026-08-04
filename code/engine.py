"""engine.py — forecasting + Newsvendor decision engine.

Single source of truth for the decision logic. Both the notebooks and the
Streamlit app import from here; do not keep a second copy in a notebook cell.

Typical use:
    from engine import load_data, evaluate_request, analyze_email
    data, holidays = load_data()
    result = evaluate_request(data, holidays, "FOODS_3_586", "CA_2",
                              "Thanksgiving", "2015-11-26", requested_pct=0.5)
"""
from pathlib import Path

import numpy as np
import pandas as pd
# Defer importing Prophet (heavy, optional) until it's actually needed in
# `evaluate_request`. This lets lighter operations (like `load_data`) run
# in environments where `prophet` isn't installed.

# code/engine.py  ->  project root is one level up, data lives in RESEARCH/data
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "final_cleaned.csv"

# Events given a pre-event stocking window (-3 days) and a +1 day tail.
MAJOR_EVENTS = ["Thanksgiving", "SuperBowl", "Christmas", "Easter",
                "IndependenceDay", "LaborDay", "NewYear"]


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_data(csv_path=None):
    """Load the cleaned long-format series and build the Prophet holidays table.

    Returns (data, holidays). `data` has ds / y / snap_CA / sell_price /
    item_id / store_id. `holidays` has holiday / ds / lower_window / upper_window.
    """
    path = Path(csv_path) if csv_path else DATA_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Expected the cleaned CSV at RESEARCH/data/final_cleaned.csv"
        )

    df = pd.read_csv(path)

    # tolerate either the raw M5 names or the renamed Prophet ones
    df = df.rename(columns={"date": "ds", "sales": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    if "snap_CA" not in df.columns:
        df["snap_CA"] = 0

    holidays = build_holidays(df)
    return df, holidays


def build_holidays(df, major_events=MAJOR_EVENTS):
    """Collapse M5's two sparse event columns into a Prophet holidays table."""
    frames = []
    for col in ("event_name_1", "event_name_2"):
        if col in df.columns:
            part = df.loc[df[col].notna(), ["ds", col]].rename(columns={col: "holiday"})
            frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])

    holidays = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates(subset=["ds", "holiday"]) 
                  .sort_values("ds")
                  .reset_index(drop=True))

    holidays["lower_window"] = 0
    holidays["upper_window"] = 0
    is_major = holidays["holiday"].isin(major_events)
    holidays.loc[is_major, "lower_window"] = -3
    holidays.loc[is_major, "upper_window"] = 1
    return holidays


def resolve_event_date(event_name, holidays, on_or_after=None):
    """Map an event name to a concrete date: the next occurrence on/after a date.

    Deterministic lookup — this is calendar work, never the LLM's job.
    """
    if not event_name:
        return None
    occ = pd.to_datetime(holidays.loc[holidays["holiday"] == event_name, "ds"])
    if occ.empty:
        return None
    if on_or_after is not None:
        future = occ[occ >= pd.Timestamp(on_or_after)]
        if len(future):
            return future.min()
    return occ.max()   # fall back to the most recent occurrence in the data


def list_series(data):
    """Available (item_id, store_id) pairs, for the app's dropdowns."""
    return (data[["item_id", "store_id"]]
            .drop_duplicates()
            .sort_values(["item_id", "store_id"])
            .itertuples(index=False, name=None))


# --------------------------------------------------------------------------
# decision engine
# --------------------------------------------------------------------------

def evaluate_request(data, holidays, item_id, store_id, event_name, event_date,
                     requested_pct=None, on_hand=0,
                     margin=0.30, salvage=0.50, lead_days=28,
                     win_before=3, win_after=1,
                     eps_approve=0.02, min_uplift=0.05, seed=42):
    """Forecast the event window, price the decision, and return the verdict.

    requested_pct is a fraction (0.5 == +50%). None -> CLARIFY, which returns
    the data-supported quantity instead of judging a number the manager
    never gave.
    """
    df = data[(data["item_id"] == item_id) & (data["store_id"] == store_id)].copy()
    if df.empty:
        raise ValueError(f"No rows for {item_id} @ {store_id}")
    df["ds"] = pd.to_datetime(df["ds"]) 
    df = df.sort_values("ds")

    event_date = pd.Timestamp(event_date)
    cutoff = event_date - pd.Timedelta(days=lead_days)
    window = pd.date_range(event_date - pd.Timedelta(days=win_before),
                           event_date + pd.Timedelta(days=win_after))

    train = df[df["ds"] < cutoff]
    if len(train) < 365:
        raise ValueError(
            f"Only {len(train)} training rows before {cutoff.date()} — "
            "need at least a year of history before the decision date."
        )

    try:
        from prophet import Prophet
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ModuleNotFoundError(
            "The 'prophet' package is required to run forecasting. "
            "Install it with `pip install prophet` and try again."
        ) from exc

    model = Prophet(holidays=holidays, interval_width=0.90, changepoint_range=0.95,
                    yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False)
    model.add_regressor("snap_CA")
    model.fit(train[["ds", "y", "snap_CA"]])

    # Build the future frame from the window itself so future events work even
    # when those dates are not present in the historical data.
    future = (pd.DataFrame({"ds": window})
                .merge(df[["ds", "snap_CA"]], on="ds", how="left")
                .fillna({"snap_CA": 0}))

    np.random.seed(seed)
    samples = model.predictive_samples(future)["yhat"]
    demand_samples = np.clip(samples, 0, None).sum(axis=0)   # sum DAYS within each path

    forecast = model.predict(future)
    event_effect = float(forecast[event_name].sum()) if event_name and event_name in forecast.columns else 0.0
    baseline_total = float(forecast["yhat"].sum()) - event_effect

    price = float(df.loc[df["ds"] <= cutoff, "sell_price"].dropna().iloc[-1])
    unit_cost = (1 - margin) * price
    salvage_value = salvage * unit_cost
    underage, overage = price - unit_cost, unit_cost - salvage_value
    critical_ratio = underage / (underage + overage)
    q_star = float(np.quantile(demand_samples, critical_ratio))

    def expected_profit(qty):
        sold = np.minimum(qty, demand_samples)
        leftover = np.maximum(qty - demand_samples, 0)
        return float((price * sold - unit_cost * qty + salvage_value * leftover).mean())

    def risk_at(qty):
        return {
            "stockout_prob": float((demand_samples > qty).mean()),
            "expected_leftover": float(np.maximum(qty - demand_samples, 0).mean()),
            "expected_lost_sales": float(np.maximum(demand_samples - qty, 0).mean()),
        }

    warranted_uplift = (q_star - baseline_total) / baseline_total

    if requested_pct is None:
        verdict, requested_qty, regret, demand_pctl = "CLARIFY", None, None, None
        recommended_qty = q_star
    else:
        requested_qty = baseline_total * (1 + requested_pct)
        regret = 1 - expected_profit(requested_qty) / expected_profit(q_star)
        demand_pctl = float((demand_samples < requested_qty).mean())
        if requested_qty <= q_star or regret <= eps_approve:
            verdict, recommended_qty = "APPROVE", requested_qty
        elif warranted_uplift > min_uplift:
            verdict, recommended_qty = "PARTIAL", q_star
        else:
            verdict, recommended_qty = "REJECT", baseline_total

    # only report an actual when the whole window is in the historical data
    observed = df.loc[df["ds"].isin(window), "y"]
    actual_total = float(observed.sum()) if len(observed) == len(window) else None

    return {
        "item_id": item_id, "store_id": store_id,
        "event_name": event_name, "event_date": event_date,
        "cutoff": cutoff, "window_start": window[0], "window_end": window[-1],

        "verdict": verdict,
        "requested_pct": requested_pct,
        "requested_qty": requested_qty,
        "recommended_qty": recommended_qty,
        "order_after_on_hand": max(float(recommended_qty) - on_hand, 0.0),
        "on_hand": on_hand,

        "baseline_total": baseline_total,
        "event_effect": event_effect,
        "forecast_mean": float(demand_samples.mean()),
        "quantiles": {f"q{int(q * 100)}": float(np.quantile(demand_samples, q))
                      for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95)},

        "price": price, "unit_cost": unit_cost, "salvage_value": salvage_value,
        "critical_ratio": critical_ratio,
        "q_star": q_star, "warranted_uplift": warranted_uplift,
        "regret": regret, "demand_percentile": demand_pctl,

        "risk_at_requested": None if requested_qty is None else risk_at(requested_qty),
        "risk_at_recommended": risk_at(recommended_qty),

        "rules": [
            {"rule": f"Approve as asked if regret ≤ {eps_approve:.0%} or request ≤ Q*",
             "passed": None if regret is None else bool(requested_qty <= q_star or regret <= eps_approve),
             "detail": None if regret is None else f"regret {regret:.1%}"},
            {"rule": f"An increase is warranted if Q* exceeds baseline by > {min_uplift:.0%}",
             "passed": bool(warranted_uplift > min_uplift),
             "detail": f"data supports {warranted_uplift:+.1%}"},
        ],

        "actual_total": actual_total,
        "history": df[["ds", "y"]],
        "demand_samples": demand_samples,
    }


# --------------------------------------------------------------------------
# email -> decision, in one call
# --------------------------------------------------------------------------

def analyze_email(email_text, data, holidays, products, extract_fn,
                  sent_date=None, on_hand=0, **kwargs):
    """Extract fields from an email, resolve the date, and evaluate the request.

    Returns (fields, result). result is None when the email cannot be evaluated;
    fields["error"] then explains why.
    """
    fields = extract_fn(email_text, products)

    if not fields.get("is_request"):
        fields["error"] = "This email does not ask for an order change."
        return fields, None
    if not fields.get("item_id"):
        fields["error"] = "No product could be identified in the email."
        return fields, None
    if not fields.get("event_name"):
        fields["error"] = ("No known calendar event was named, so there is no "
                           "event window to forecast.")
        return fields, None

    event_date = resolve_event_date(fields["event_name"], holidays, on_or_after=sent_date)
    if event_date is None:
        fields["error"] = f"'{fields['event_name']}' does not appear in the event calendar."
        return fields, None
    fields["event_date"] = event_date

    store_id = fields.get("store_id")
    if not store_id or ((data["item_id"] == fields["item_id"]) & (data["store_id"] == store_id)).sum() == 0:
        match = data.loc[data["item_id"] == fields["item_id"], "store_id"]
        if match.empty:
            fields["error"] = f"{fields['item_id']} is not in the dataset."
            return fields, None
        store_id = match.iloc[0]
        fields["store_id"] = store_id

    result = evaluate_request(data, holidays, fields["item_id"], store_id,
                              fields["event_name"], event_date,
                              requested_pct=fields.get("requested_pct"),
                              on_hand=on_hand, **kwargs)
    return fields, result
