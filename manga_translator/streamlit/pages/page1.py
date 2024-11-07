import logging
import streamlit as st
import src as _src

logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)
logger.debug("page1")

st.markdown("page1")

a = st.number_input("a", value=1)
b = st.number_input("b", value=2)

def on_slide_change():
    count = st.session_state.get("slide_change_count", 0)
    count += 1
    st.session_state["slide_change_count"] = count
    st.write(f"slide changed {count}")

c=st.slider("c",1,10,3, on_change=on_slide_change)

st.write(_src.cached_compute(a, c))

st.session_state