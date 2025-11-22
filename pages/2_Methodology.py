import streamlit as st
from PIL import Image
from helper_functions.utility import check_password

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Methodology - Agentic AI Risk & Capability Framework Assistant"
)

# Check if the password is correct.
if not check_password():
    st.stop()
# endregion <--------- Streamlit App Configuration --------->

st.title("Methodology")

local_image = Image.open('./data/Methodology.png')
st.image(local_image, caption="Flowchart generated using Mermaid on GitHub Wiki", use_container_width=True)