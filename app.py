import streamlit as st
from agent_executor import settings

st.set_page_config(page_title="🌱  Sprouty", layout="centered")

st.title("🌱 Sprouty, your AI farming assistant")
"""
Hi, I am Sprouty, your farming assistant. Ask me your farming related queries!
"""
st.sidebar.info("Sprouty is a part of assignment by Sensegrass", icon="🤖")
st.sidebar.info("Built by Muhammed Musaddique. musaddique092@gmail.com")
if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = settings()

agent_executor = st.session_state.agent_executor

if agent_executor:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi, I am Sprouty, your farming assistant. Ask me your farming related queries!",
            }
        ]

    if user_input := st.chat_input("Your query", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            with st.empty():
                st.write(message["content"])

    # If the user input is not from the assistant, query the engine and display the response
    if user_input and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking...may take some time"):
                response = agent_executor.invoke(
                    {"input": user_input}, {"stop": ["Final Answer"]}
                )
                st.write(response["output"])
                print(response["output"])
                print("\n\n\n\n\n")
                print(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["output"]}
                )
