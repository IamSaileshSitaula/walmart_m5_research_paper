"""Instrumented debug copy of app.py to locate where rendering stops.
Run as:
    streamlit run code/app_debug.py --server.port 8506
"""
import os
import sys
import time

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

from engine import analyze_email, evaluate_request, load_data, resolve_event_date
from extractor import EXTRACTORS

load_dotenv(find_dotenv())

print('DEBUG: module import complete')

st.set_page_config(page_title="Inventory Request Review (debug)", layout="wide")
print('DEBUG: set_page_config')

st.write('DEBUG: before sidebar')

@st.cache_resource(show_spinner=False)
def get_data():
    print('DEBUG: get_data called')
    return load_data()

@st.cache_data(show_spinner=False)
def run_engine(item_id, store_id, event_name, event_date, requested_pct, on_hand,
               margin, salvage):
    print('DEBUG: run_engine called')
    data, holidays = get_data()
    return evaluate_request(data, holidays, item_id, store_id, event_name, event_date,
                            requested_pct=requested_pct, on_hand=on_hand,
                            margin=margin, salvage=salvage)

print('DEBUG: functions defined')

with st.sidebar:
    st.subheader('Settings')
    print('DEBUG: in sidebar')
    available = [name for name, fn in EXTRACTORS.items()
                 if name != 'gemini' or os.getenv('GEMINI_API_KEY')]
    st.write('DEBUG: available extractors', available)

st.title('Inventory request review (debug)')
st.write('DEBUG: after title')

email_text = st.text_area("Manager's email", value='DEBUG sample', height=200)
print('DEBUG: after text_area')

st.write('DEBUG: about to show button')
if st.button('Review request', type='primary'):
    print('DEBUG: button pressed')
    st.write('DEBUG: inside button handler')

st.write('DEBUG: script end')
print('DEBUG: script end')
