# Set up and run this Streamlit App
import streamlit as st
from logics import aigov
from helper_functions.utility import check_password

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Main - Agentic AI Risk & Capability Framework Assistant"
)

# Check if the password is correct.
if not check_password():
    st.stop()
# endregion <--------- Streamlit App Configuration --------->

st.title("Agentic AI Risk & Capability Framework Assistant")

st.markdown("""
This application uses information from the Agentic Risk & Capability Framework developed by the Responsible AI team
in GovTech Singapore's AI Practice, which can be found [here](https://govtech-responsibleai.github.io/agentic-risk-capability-framework/).

You may either ask it questions about agentic AI risks and capabilities, or provide an agentic AI use case for it to help apply the Framework to.
""")

with st.expander(":red[IMPORTANT NOTICE]"):
    st.markdown("""
    This web application is developed as a proof-of-concept prototype. The information provided here is :red[NOT intended for actual usage] and
    should not be relied upon for making any decisions, especially those related to financial, legal or healthcare matters.

    :red[Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for
    how you use any generated output.]

    Always consult with qualified professionals for accurate and personalised advice.
    """)

# Initialise chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Initialise vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = aigov.create_vector_store('./data')
# Initialise conversation RAG chain
if 'conversation_rag_chain' not in st.session_state:
    st.session_state.conversation_rag_chain = aigov.create_conversation_rag_chain(st.session_state.vector_store)

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Accept user input
if prompt := st.chat_input("How may I assist you today?"):
    aigov.process_user_message(prompt)