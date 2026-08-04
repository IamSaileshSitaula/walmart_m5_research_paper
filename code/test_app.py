import streamlit as st

st.title('Streamlit test')
st.write('If you see this, the Streamlit frontend is working.')
if st.button('Say hello'):
    st.success('Hello — frontend OK')
