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
### Application Description
This application aims to facilitate users' understanding of the Agentic Risk & Capability Framework and why it is important,
as well as to help them apply the Framework to their agentic AI use case thus enabling them to conduct their risk assessment more efficiently.

### Application Features
- Classification of queries into Information or Application type
- Conversation history aware RAG on the [Agentic Risk & Capability Framework](https://govtech-responsibleai.github.io/agentic-risk-capability-framework/)
for Information type queries
- Linear prompt chaining using the [Agentic Risk & Capability Framework](https://govtech-responsibleai.github.io/agentic-risk-capability-framework/)
for Application type queries

_Where building AI solutions  
meets governing their rise,  
we shape both code and conscience  
beneath the same widening skies._
""")