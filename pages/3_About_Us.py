import streamlit as st
from helper_functions.utility import check_password

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="About Us - Agentic AI Risk & Capability Framework Assistant"
)

# Check if the password is correct.
if not check_password():
    st.stop()
# endregion <--------- Streamlit App Configuration --------->

st.title("About Us")

st.markdown("""
Where building AI solutions meets governing AI solutions.

_Where building AI solutions  
meets governing their rise,  
we shape both code and conscience  
beneath the same widening skies._
""")