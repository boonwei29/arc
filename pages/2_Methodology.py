import streamlit as st
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

st.markdown("""
1. Uses information from the Agentic Risk & Capability Framework developed by the Responsible AI team in GovTech Singapore's AI Practice,
which can be found [here](https://govtech-responsibleai.github.io/agentic-risk-capability-framework/)
2. Classifies query into 'Information' (information on agentic AI risks and capabilities) or 'Application' (application on agentic AI use case)
3. If 'Information' query type:
    - Uses conversation history aware RAG
    - To improve pre-retrieval process, uses HTMLHeaderTextSplitter
    - To improve retrieval process, uses Maximum Marginal Relevance (MMR)
4. If 'Application' query type:
    - Asks user for more details in a form
    - Uses linear prompt chaining, first identifying the risks for the use case, then using the identified risks and technical controls
to draft a preliminary risk assessment
""")