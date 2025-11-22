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
```mermaid
flowchart TD
    q[/Query/] --> id1[check_query_type]
    id1 -->|Information type| id2[conversation_rag_chain]
    id2 --> a[/Answer/]
    id1 -->|Application type| id3[Ask user for more details]
    id3 --> id4[identify_risks]
    id4 --> id5[generate_summary]
    id5 --> a
    id4_1[/Agentic Risk & Capability Framework XLSX/] --> id4
    id2_1[/Agentic Risk & Capability Framework HTML/] --> id2_2[create_vector_store]
    id2_2 --> id2_3[create_conversation_rag_chain]
    id2_3 -->|Initialise before invoking| id2
```
""")