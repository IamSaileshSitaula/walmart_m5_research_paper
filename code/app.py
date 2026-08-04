"""app.py — Streamlit dashboard for the inventory approval system.

Run from the project root:
    streamlit run code/app.py
"""
import os

import altair as alt
import pandas as pd
import streamlit as st
# dotenv is optional in some environments; provide safe fallbacks if missing.
try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - fallback for missing package
    def find_dotenv(*args, **kwargs):
        return None

    def load_dotenv(*args, **kwargs):
        return False

# Defer heavy engine imports (Prophet) until needed so the app UI can
# initialize without blocking on model imports or long setup.
# Functions from `engine` are imported lazily inside the callers below.
from extractor import EXTRACTORS

load_dotenv(find_dotenv())

st.set_page_config(page_title="Inventory Request Review", layout="wide")

# Products the manager might name in an email. Keys are display names the LLM
# maps onto ids; M5 has no real product names, so these stand in for them.
PRODUCTS = {
    "Sunrise Valley tortilla chips 24oz": {"item_id": "FOODS_3_586", "store_id": "CA_2"},
}

SAMPLE_EMAIL = """From: Store Manager, CA_2
Subject: Thanksgiving stock

We ran out of the Sunrise Valley tortilla chips two days before Thanksgiving
last year and I don't want a repeat. Please increase our order by 50% for
Thanksgiving week.

Marcus"""

VERDICT_STYLE = {
    "APPROVE": ("Approve as requested", "success"),
    "PARTIAL": ("Approve in part", "warning"),
    "REJECT": ("Hold the baseline order", "error"),
    "CLARIFY": ("Needs a quantity", "info"),
}


@st.cache_resource(show_spinner=False)
def get_data():
    # Import here to avoid importing Prophet at module import time.
    from engine import load_data
    return load_data()


@st.cache_data(show_spinner=False)
def run_engine(item_id, store_id, event_name, event_date, requested_pct, on_hand,
               margin, salvage):
    data, holidays = get_data()
    # Import evaluate_request lazily to avoid heavy top-level imports.
    from engine import evaluate_request
    return evaluate_request(data, holidays, item_id, store_id, event_name, event_date,
                            requested_pct=requested_pct, on_hand=on_hand,
                            margin=margin, salvage=salvage)


def units(x):
    return "—" if x is None else f"{x:,.0f}"


# --------------------------------------------------------------------------
# sidebar: everything the dataset cannot tell us
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")
    # Allow pasting a Gemini API key in the sidebar (session-only) so users
    # who prefer Gemini can use it even if the environment doesn't have GEMINI_API_KEY.
    gemini_key = st.text_input("Gemini API key (optional)", type="password",
                              help="Paste a GEMINI_API_KEY to enable the Gemini extractor for this session.")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    # Optionally hide Anthropic/Claude if you don't want it offered.
    hide_claude = st.checkbox("Hide Claude extractor (use Gemini only)", value=True)

    available = []
    for name, fn in EXTRACTORS.items():
        if name == "gemini" and not os.getenv("GEMINI_API_KEY"):
            continue
        if hide_claude and name == "claude":
            continue
        available.append(name)
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        st.caption("No API key found — using the rule-based reader. "
                   "Add GEMINI_API_KEY to a .env file for language extraction.")
    engine_choice = st.selectbox("Email reader", available,
                                 index=available.index("gemini") if "gemini" in available else 0)

    on_hand = st.number_input("Units already in stock", min_value=0, value=0, step=10,
                              help="Not in the M5 dataset — supplied by you. "
                                   "The order is reduced by this amount.")
    st.divider()
    st.caption("Cost assumptions (M5 has price only)")
    margin = st.slider("Gross margin", 0.10, 0.50, 0.30, 0.05)
    salvage = st.slider("Salvage share of cost", 0.0, 0.9, 0.50, 0.10)

# --------------------------------------------------------------------------
# main: the request
# --------------------------------------------------------------------------
st.title("Inventory request review")
st.caption("Paste a manager's email. The system forecasts demand for the event "
           "window, computes the profit-optimal order, and explains the gap.")

email_text = st.text_area("Manager's email", value=SAMPLE_EMAIL, height=200)
sent_date = st.date_input("Date the email was sent", value=pd.Timestamp("2015-10-29"),
                          help="The forecast is fitted using only data available on this date.")

if st.button("Review request", type="primary"):
    data, holidays = get_data()
    extract_fn = EXTRACTORS[engine_choice]

    with st.spinner("Reading the email…"):
        try:
            fields = extract_fn(email_text, PRODUCTS)
        except Exception as exc:                     # noqa: BLE001
            st.error(f"The email reader failed: {exc}")
            st.stop()

    st.session_state["fields"] = fields

    if not fields.get("is_request"):
        st.info("This email does not ask for an order change, so there is nothing to review.")
        st.json(fields)
        st.stop()
    if not fields.get("item_id"):
        st.warning("No known product was named in the email.")
        st.json(fields)
        st.stop()
    if not fields.get("event_name"):
        st.warning("No calendar event was named, so there is no window to forecast. "
                   "Ask the manager which event this is for.")
        st.json(fields)
        st.stop()

    # Resolve the named event to a concrete date. Import lazily to avoid
    # importing the heavy forecasting libs at module import time.
    from engine import resolve_event_date
    event_date = resolve_event_date(fields["event_name"], holidays, on_or_after=sent_date)
    if event_date is None:
        st.warning(f"'{fields['event_name']}' is not in the event calendar.")
        st.stop()

    store_id = fields.get("store_id")
    known = data.loc[data["item_id"] == fields["item_id"], "store_id"].unique()
    if len(known) == 0:
        st.error(f"{fields['item_id']} is not in the dataset.")
        st.stop()
    if store_id not in known:
        store_id = known[0]   # email named a store we have no data for

    with st.spinner("Fitting the demand model…"):
        try:
            result = run_engine(fields["item_id"], store_id, fields["event_name"],
                                event_date, fields.get("requested_pct"), on_hand,
                                margin, salvage)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["result"] = result

# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
result = st.session_state.get("result")
fields = st.session_state.get("fields")

if result:
    headline, tone = VERDICT_STYLE[result["verdict"]]
    order = result["order_after_on_hand"]

    if result["verdict"] == "CLARIFY":
        message = (f"**{headline}** — the email does not say how much more to order. "
                   f"The data supports **{units(result['q_star'])} units**.")
    elif result["verdict"] == "APPROVE":
        message = (f"**{headline}** — order **{units(order)} units**. "
                   f"The request is within what the forecast supports.")
    elif result["verdict"] == "PARTIAL":
        message = (f"**{headline}** — order **{units(order)} units**, not "
                   f"{units(result['requested_qty'])}. An increase is justified, "
                   f"but not one this large.")
    else:
        message = (f"**{headline}** — order **{units(order)} units**. The forecast "
                   f"shows no meaningful lift for this item at this event.")

    getattr(st, tone)(message)

    # ---- evidence panels -------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Extracted from the email**")
        st.write(f"Item {result['item_id']} at {result['store_id']}")
        st.write(f"{result['event_name']} · {result['event_date']:%d %b %Y}")
        pct = result["requested_pct"]
        st.write("No quantity stated" if pct is None else f"Requested +{pct:.0%}")
        if fields:
            st.caption(f"Urgency: {fields.get('urgency', 'normal')}")

    with c2:
        st.markdown("**Demand for the window**")
        st.write(f"Normal week: {units(result['baseline_total'])}")
        st.write(f"Event effect: {result['event_effect']:+,.0f}")
        st.write(f"Forecast: {units(result['forecast_mean'])} "
                 f"(90th pct {units(result['quantiles']['q90'])})")
        st.caption(f"{result['window_start']:%d %b} – {result['window_end']:%d %b}, "
                   f"decided {result['cutoff']:%d %b}")

    with c3:
        st.markdown("**Optimal order**")
        st.write(f"Q* = {units(result['q_star'])} units")
        st.write(f"Critical ratio {result['critical_ratio']:.2f}")
        st.write(f"Data supports {result['warranted_uplift']:+.1%}")
        if result["on_hand"]:
            st.caption(f"Less {result['on_hand']} in stock → order {units(order)}")

    with c4:
        st.markdown("**Risk**")
        rec = result["risk_at_recommended"]
        st.write(f"At {units(result['recommended_qty'])}: "
                 f"{rec['expected_leftover']:.0f} left over, "
                 f"{rec['stockout_prob']:.0%} stockout risk")
        req = result["risk_at_requested"]
        if req and result["verdict"] in ("PARTIAL", "REJECT"):
            st.write(f"At {units(result['requested_qty'])}: "
                     f"{req['expected_leftover']:.0f} left over")
            st.caption(f"Granting the request costs {result['regret']:.1%} of expected profit")

    st.divider()

    # ---- rules -----------------------------------------------------------
    st.markdown("**Rules applied**")
    for rule in result["rules"]:
        if rule["passed"] is None:
            mark = "—"
        else:
            mark = "✓" if rule["passed"] else "✗"
        st.write(f"{mark}  {rule['rule']}" + (f"  ({rule['detail']})" if rule["detail"] else ""))

    # ---- charts ----------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.markdown("**Where the request sits in the demand forecast**")
        samples = pd.DataFrame({"units": result["demand_samples"]})
        hist = (alt.Chart(samples)
                .mark_bar(opacity=0.75)
                .encode(x=alt.X("units:Q", bin=alt.Bin(maxbins=40),
                                title="possible demand over the window"),
                        y=alt.Y("count()", title="simulated weeks")))
        marks = pd.DataFrame([
            {"units": result["q_star"], "label": "optimal"},
            *([{"units": result["requested_qty"], "label": "requested"}]
              if result["requested_qty"] else []),
        ])
        rules_layer = (alt.Chart(marks)
                       .mark_rule(strokeWidth=2)
                       .encode(x="units:Q",
                               color=alt.Color("label:N", legend=alt.Legend(title=None))))
        st.altair_chart(hist + rules_layer, use_container_width=True)

    with right:
        st.markdown("**Recent sales history**")
        history = result["history"]
        recent = history[history["ds"] >= result["cutoff"] - pd.Timedelta(days=365)]
        line = (alt.Chart(recent)
                .mark_line(opacity=0.5)
                .encode(x=alt.X("ds:T", title=None), y=alt.Y("y:Q", title="units sold")))
        st.altair_chart(line, use_container_width=True)

    # ---- written rationale ----------------------------------------------
    st.markdown("**Why**")
    if result["verdict"] == "REJECT":
        why = (f"Historical sales show no meaningful lift for {result['item_id']} at "
               f"{result['event_name']} — the model puts the event effect at "
               f"{result['event_effect']:+,.0f} units across the window. The optimal order "
               f"of {units(result['q_star'])} is within {abs(result['warranted_uplift']):.0%} of a "
               f"normal week, so the baseline order stands. Granting the request would "
               f"sacrifice {result['regret']:.0%} of expected profit, leaving roughly "
               f"{result['risk_at_requested']['expected_leftover']:.0f} units unsold.")
    elif result["verdict"] == "PARTIAL":
        why = (f"{result['event_name']} does lift demand for this item — the model attributes "
               f"{result['event_effect']:+,.0f} units to the event — so an increase is justified. "
               f"But the requested {units(result['requested_qty'])} units sits at the "
               f"{result['demand_percentile']:.0%} mark of simulated demand, and ordering it "
               f"would sacrifice {result['regret']:.0%} of expected profit to unsold stock. "
               f"The data supports {result['warranted_uplift']:+.1%}, giving "
               f"{units(result['q_star'])} units.")
    elif result["verdict"] == "APPROVE":
        why = (f"The requested {units(result['requested_qty'])} units is close to the optimal "
               f"{units(result['q_star'])}, costing {result['regret']:.1%} of expected profit — "
               f"immaterial. {result['event_name']} adds {result['event_effect']:+,.0f} units "
               f"to a normal week of {units(result['baseline_total'])}.")
    else:
        why = (f"The email asks for more stock without saying how much. Based on the forecast "
               f"for {result['event_name']}, the profit-optimal order is "
               f"{units(result['q_star'])} units — {result['warranted_uplift']:+.1%} against a "
               f"normal week.")
    st.write(why)

    if result["actual_total"] is not None:
        st.caption(f"Backtest: actual sales in this window were "
                   f"{units(result['actual_total'])} units.")

    with st.expander("All computed values"):
        st.json({k: v for k, v in result.items()
                 if k not in ("demand_samples", "history")}, expanded=False)