import os

# import streamlit as st
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory

load_dotenv()

st.set_page_config(page_title="CodeChat", layout="centered")

st.title("🦙 CodeChat: Talk with your GitHub Repos 💬 📚")


embeddings = HuggingFaceHubEmbeddings()


loader = PyPDFLoader("./crop-management-AR-2011-12.pdf")
pages = loader.load_and_split()

persist_directory = "local_vectorstore"
collection_name = "crop_management"
PROJECT_ROOT = "FarmerBot"  # insert your project root directory name here

vectorstore = Chroma.from_documents(
    documents=pages,
    persist_directory=os.path.join(PROJECT_ROOT, "data", persist_directory),
    collection_name=collection_name,
    embedding=embeddings,
)


# The storage layer for the parent documents
local_store = "local_docstore"
local_store = LocalFileStore(os.path.join(PROJECT_ROOT, "data", local_store))
docstore = create_kv_docstore(local_store)

# This text splitter is used to create the child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
)

tool_search = create_retriever_tool(
    retriever=retriever,
    name="search_crop_management",
    description="A guide to crop management by Indian Council of Agricultural Reasearch (ICAR). Your first place to search for crop managemt and fertilizer related queries",
)


search = GoogleSearchAPIWrapper()
google_tool = Tool(
    name="Google Search",
    func=search.run,
    description="Use for when you need to perform an internet search to find information that another tool can not provide.",
)


tools = [
    google_tool,
    tool_search,
]



# You are Sprouty, AI assistant for farmers. Your sole job is to only answer farming related queries of farmers as best you can. If you find the question ambiguous or missing some details, respond by asking the user for neccessary specific details. Also provide the VERBATIM citations for the sources refered in the end of the response. Stop if you arrive at the final answer. Remember, if the question below is not related to farming, simply respond by asking the user to ask only farming related questions. If you don't find any relevant answer, simply say I don't know. You don't need to provide citations to the answers you don't find and also to the questions that are unrelated to farming. If you find an answer, respond in a detailed manner.
# Have a conversation with a human, answering the following questions as best you can. You have access to the following tools
# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times until you arrive at the final answer.)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# The process can stop now!

# Begin!

# {chat_history}

# Question: {input}
# Thought:{agent_scratchpad}
template = """
You are Sprouty, a friendly, helpful, and an expert farming AI assistant for farmers. 
Your sole job is to only answer queries of farmers related to your identity and farming as best you can. 
Answer only to the point. 
If you find the question ambiguous or missing some details, respond by asking the user for neccessary specific details. 
Also provide the VERBATIM citations for the sources refered in the end of the response. Stop if you arrive at the final answer. 
Remember, if the question below is not related to farming, simply respond by asking the user to ask only farming related questions. 
If you don't find any relevant answer, simply say I don't know. If the user asks about your identity, simply say it. 
You don't need to provide citations to the answers you don't find and also to the questions that are unrelated to farming. If you find an answer, respond in a detailed manner.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, if answer is not found, itshould be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. (youcan )

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
prompt_template = PromptTemplate.from_template(template)


memory = ConversationBufferMemory(memory_key="chat_history")


agent = create_react_agent(
    llm,
    tools,
    prompt_template,
    # stop_sequence=["Final Answer"],
    # stop_sequence=["Observation"],
)
# ========================================experimental===========================================================

# from langchain import hub
# from langchain.agents import AgentExecutor, load_tools
# from langchain.agents.format_scratchpad import format_log_to_str
# from langchain.agents.output_parsers import (
#     ReActJsonSingleInputOutputParser,
# )
# from langchain.tools.render import render_text_description

# # prompt = hub.pull("hwchase17/react-json")
# prompt = prompt_template.partial(
#     tools=render_text_description(tools),
#     tool_names=", ".join([t.name for t in tools]),
# )

# chat_model_with_stop = llm.bind(stop=["\nObservation"])
# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
#     }
#     | prompt
#     | chat_model_with_stop
#     | ReActJsonSingleInputOutputParser()
# )
# =================================================================================================================

agent_executor = AgentExecutor.from_agent_and_tools(
    # retrieval_qa_with_sources_chain,
    agent=agent,
    memory=memory,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)


if __name__ == "__main__":
    while True:
        question = input("Enter your question: ")
        # input1 = "Translate this sentence from English to French: I love programming."
        # demo_ephemeral_chat_history.add_user_message(question)
        if question.lower() != "quit":
            output = agent_executor.invoke(
                {
                    "input": question,
                },
            )

            # answer.info("`Answer:`\n\n" + output)
            # st.info("`Sources:`\n\n" + output)
            print(output["output"])
        else:
            exit()
            SYSTEM_PROMPT = """You are a helpful AI Assistant and an expert in marketing. Before answering any question, use the "list_overall_products" tool to get a list of all products in the database. If the user question has nothing to do with the products, respond that you don't know and that the user should only ask questions about our products. If the user question is about the product, use the "get_detailed_product" tool to get more details about the product."""
