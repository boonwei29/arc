import os
import pandas as pd
import json
import streamlit as st
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from helper_functions import llm

#### Classify the query

def check_query_type(user_message):
    delimiter = '####'

    system_message = f"""
    You will be provided with a query on the Agentic AI Risk & Capability Framework. The query will be enclosed in a pair of {delimiter}.

    Decide if the query is one of the following categories and output the category:
    - 'Application': If the user is seeking guidance on applying the Agentic AI Risk & Capability Framework, and has provided an agentic AI use case.
    - 'Information': For all other cases.

    Ensure your response contains only a one-word category, without any enclosing tags or delimiters.
    """

    messages =  [
        {'role': 'assistant',
         'content': system_message},
        {'role': 'user',
         'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response = llm.get_completion_by_messages(messages)
    return response

#### RAG for 'Information' query type

def create_vector_store(directory):
    # Validate directory path
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Load and split the files into smaller chunks
    headers_to_split_on = [
        ('h1', 'Header 1'),
        ('h2', 'Header 2'),
        ('h3', 'Header 3')
    ]
    text_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
    list_of_split_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.html'):  # Case-insensitive match
            try:
                filepath = os.path.join(directory, filename)
                data = text_splitter.split_text_from_file(filepath)
                list_of_split_files.extend(data)
                print(f"Loaded/split {filename}")

            except Exception as e:
                print(f"Error loading/splitting {filename}: {e}")

    print('Total files after splitting: ', len(list_of_split_files))

    # Create vector store
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = Chroma.from_documents(
        documents=list_of_split_files,
        embedding=embeddings_model,
        collection_name='aigov_docs',
        persist_directory='./vector_db'
    )

    return vector_store

def create_conversation_rag_chain(vector_store):
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1, seed=42)

    # Create a retriever that is aware of the conversation history
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 8, 'fetch_k': 10})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        ('user', "Given the above conversation, generate a search query to get information relevant to the conversation.")
    ])
    history_retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    # Create a RAG chain that is aware of the conversation history
    answer_prompt = ChatPromptTemplate.from_messages([
        ('assistant', """Answer the user's query based on the below context. Keep your answer as concise and as easily understood
         by laymen as possible. In your answer, try to bring across why an Agentic AI Risk & Capability Framework is needed.
         If you do not have sufficient information, inform the user that you are unable to assist with his/her query.\n\n{context}"""),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}')
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversation_rag_chain = create_retrieval_chain(history_retriever_chain, document_chain)

    return conversation_rag_chain

#### Prompt chaining for 'Application' query type

filepath = './data/Agentic Risk & Capability Framework.xlsx'
taxonomy_df = pd.read_excel(filepath, sheet_name='Taxonomy')
techctrls_df = pd.read_excel(filepath, sheet_name='Technical Controls')

def identify_risks(user_message):
    taxonomy_md = taxonomy_df.to_markdown(index=False)
    techctrls_md = techctrls_df.to_markdown(index=False)
    delimiter = '####'

    system_message = f"""
    You will be provided with an agentic AI use case to apply the Agentic AI Risk & Capability Framework to.
    The query will be enclosed in a pair of {delimiter}.

    Using the two Markdown tables below as context, expand on the agentic AI use case provided to take into account
    all possible risks associated with it.
    {taxonomy_md}
    {techctrls_md}

    If there are any relevant risks found, output the `Risk` extracted from {techctrls_md} into a Python list.
    If there are no relevant risks found, output an empty list.

    Ensure your response contains only a Python list of possible risks or an empty list, without any enclosing tags or delimiters.
    """

    messages =  [
        {'role': 'assistant',
         'content': system_message},
        {'role': 'user',
         'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response = llm.get_completion_by_messages(messages)
    return response

def generate_summary(risks_str, user_message):
    if risks_str is None or risks_str=='[]':
        risks_techctrls_df = techctrls_df[techctrls_df['Category']=='Baseline']
    else:
        risks_list = json.loads(risks_str)
        risks_techctrls_df = techctrls_df[techctrls_df['Risk'].isin(risks_list)]

    risks_techctrls_md = risks_techctrls_df.to_markdown(index=False)
    delimiter = '####'

    system_message = f"""
    You will be provided with an agentic AI use case to apply the Agentic AI Risk & Capability Framework to.
    The query will be enclosed in a pair of {delimiter}.

    Taking reference from the Markdown table {risks_techctrls_md}, which contains risks and technical controls relevant
    to the agentic AI use case provided, write a summary on the primary risks and recommended technical controls for the use case.
    The summary should include a brief description of the use case at the start.

    Keep the summary as concise and as easily understood by laymen as possible, and limit it to a maximum of two paragraphs.
    At the end of the summary, you should inform users that the assistant aims to provide only a first cut risk assessment for agentic AI use cases
    to kick-start the process, and that users should consult the appropriate personnel for a proper risk assessment as required.
    """

    messages =  [
        {'role': 'assistant',
         'content': system_message},
        {'role': 'user',
         'content': f"{delimiter}{user_message}{delimiter}"},
    ]

    response = llm.get_completion_by_messages(messages)
    response = '\n\n'.join([response, risks_techctrls_md])
    return response

#### Putting it all together

def process_user_message(user_message):
    def disp_save_chat(query, answer):
        with st.chat_message('user'):
            st.markdown(query)
        with st.chat_message('assistant'):
            st.markdown(answer)
        st.session_state.chat_history.append({'role': 'user', 'content': query})
        st.session_state.chat_history.append({'role': 'assistant', 'content': answer})

    def handle_application():
        user_message_add = '\n\n'.join([
            user_message,
            'Q: ' + st.session_state.detail_qn,
            'A: ' + st.session_state.detail,
            'Q: ' + st.session_state.unsafe_qn,
            'A: ' + st.session_state.unsafe,
            'Q: ' + st.session_state.specialised_qn,
            'A: ' + st.session_state.specialised,
            'Q: ' + st.session_state.model_qn,
            'A: ' + st.session_state.model,
            'Q: ' + st.session_state.agent_qn,
            'A: ' + st.session_state.agent,
            'Q: ' + st.session_state.sensitive_qn,
            'A: ' + st.session_state.sensitive
        ])
        risks_str = identify_risks(user_message_add)
        response = generate_summary(risks_str, user_message_add)
        disp_save_chat(user_message_add, response)

    query_type = check_query_type(user_message)

    if query_type=='Information':
        response = st.session_state.conversation_rag_chain.invoke({
            'chat_history': st.session_state.chat_history,
            'input': user_message
         })
        disp_save_chat(user_message, response['answer'])

    else:
        with st.form('add_inputs_form', enter_to_submit=False):
            st.markdown("You have provided an agentic AI use case, and would like to apply the Agentic AI Risk & Capability Framework to it.")
            st.markdown('_' + user_message + '_')

            st.session_state.detail_qn = "Please provide more details on your AI use case."
            st.text_input(st.session_state.detail_qn, key='detail')
            st.session_state.unsafe_qn = "Does your AI use case involve the analysis of unsafe material e.g. violence?"
            st.text_input(st.session_state.unsafe_qn, key='unsafe')
            st.session_state.specialised_qn = "Does your AI use case involve specialised domains e.g. medical, financial, legal?"
            st.text_input(st.session_state.specialised_qn, key='specialised')
            st.session_state.model_qn = "What is the LLM or AI model used in your solution? Does it come from a reputable source?"
            st.text_input(st.session_state.model_qn, key='model')
            st.session_state.agent_qn = "What are the agentic tools provided in your solution? What is the level of permissions granted to them?"
            st.text_input(st.session_state.agent_qn, key='agent')
            st.session_state.sensitive_qn = "Does your solution use or store sensitive user or organisational data?"
            st.text_input(st.session_state.sensitive_qn, key='sensitive')

            st.form_submit_button('Submit', on_click=handle_application)