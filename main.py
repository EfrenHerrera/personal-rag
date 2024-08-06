import streamlit as st
from common.utils.stramlit_style import hide_streamlit_style
from infrastructure.langchain_module import response

st.set_page_config(page_title="Home - Rag Shiro", page_icon=":shark:", layout="wide")
hide_streamlit_style()

st.title("Welcome to Rag Shiro's")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Escribí tu mensaje 😎"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    

if user_input != None:
    if st.session_state.messages and user_input.strip() != "":
        response = response(user_input)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})