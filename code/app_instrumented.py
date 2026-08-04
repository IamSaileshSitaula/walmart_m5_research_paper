"""Instrumented copy of app.py with stdout traces to find where rendering stops.
Run as:
    streamlit run code/app_instrumented.py --server.port 8507
"""
import os
import sys
print('TRACE: starting app_instrumented import')

try:
    import altair as alt
    import pandas as pd
    import streamlit as st
    # dotenv optional
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:  # pragma: no cover
        def find_dotenv(*args, **kwargs):
            return None

        def load_dotenv(*args, **kwargs):
            return False
    print('TRACE: imported altair, pandas, streamlit, dotenv')
except Exception as e:
    print('TRACE: import error', repr(e))
    raise

try:
    # import engine and extractor as in original app
    from engine import analyze_email, evaluate_request, load_data, resolve_event_date
    from extractor import EXTRACTORS
    print('TRACE: imported engine and extractor')
except Exception as e:
    print('TRACE: engine/extractor import error', repr(e))
    raise

load_dotenv(find_dotenv())

st.set_page_config(page_title="Inventory Request Review (instrumented)", layout="wide")
print('TRACE: set_page_config')

PRODUCTS = {
    "Sunrise Valley tortilla chips 24oz": {"item_id": "FOODS_3_586", "store_id": "CA_2"},
}

SAMPLE_EMAIL = """From: Store Manager, CA_2
Subject: Thanksgiving stock

We ran out of the Sunrise Valley tortilla chips two days before Thanksgiving
last year and I don't want a repeat. Please increase our order by 50% for
Thanksgiving week.

Marcus"""

print('TRACE: constants defined')

@st.cache_resource(show_spinner=False)
def get_data():
    print('TRACE: get_data called')
    return load_data()

@st.cache_data(show_spinner=False)
def run_engine(item_id, store_id, event_name, event_date, requested_pct, on_hand,
               margin, salvage):
    print('TRACE: run_engine called')
    data, holidays = get_data()
    return evaluate_request(data, holidays, item_id, store_id, event_name, event_date,
                            requested_pct=requested_pct, on_hand=on_hand,
                            margin=margin, salvage=salvage)

print('TRACE: cached functions defined')

import streamlit as st2
print('TRACE: about to build sidebar')
with st.sidebar:
    st.subheader('Settings (instrumented)')
    print('TRACE: in sidebar context')
    available = [name for name, fn in EXTRACTORS.items() if name != 'gemini' or os.getenv('GEMINI_API_KEY')]
    st.write('TRACE: available extractors', available)

st.title('Inventory request review (instrumented)')
print('TRACE: after title')

email_text = st.text_area("Manager's email", value=SAMPLE_EMAIL, height=200)
print('TRACE: after text_area')

sent_date = st.date_input('Date the email was sent', value=pd.Timestamp('2015-10-29'))
print('TRACE: after date_input')

if st.button('Review request', type='primary'):
    print('TRACE: button pressed')
    data, holidays = get_data()
    st.write('TRACE: loaded data in button handler')

print('TRACE: script end')
