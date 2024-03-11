__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from openai import OpenAI
# import chromadb
# from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# The UI Part
st.title("👨‍💻 Wazzup!!!! Let's Chat with the Hansard Senate Estimates for Employment Department (DEWR) - 14 Feb 2024")
prompt = st.text_area("Please enter what you want to know from the hearing for the Employment Department.")

# Load VectorDB
if st.sidebar.button("Load Hansard into Vector DB if loading the page for the first time.", type="primary"): 
    loader = TextLoader("hansardFeb2024.txt")
    docs = loader.load()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
       # Set chunk size, just to show.
        chunk_size=750,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "Search_Hansard",
        "Searches and returns Hansard data.",
    )
    retriever_tool = [tool]
     # retriever_tool = create_retriever_tool(
        # retriever,
        # "handsard_search",
        # "Search for information about Handsard. For any questions about Handsard, you must use this tool!",
     # )
    tools = [retriever_tool]
    st.write("Vector DB Created and Retriever Tool Created.")

if st.button("Submit to DJ Arvee", type="primary"):
     # Get the prompt to use - you can modify this!
     prompt_template = hub.pull("hwchase17/openai-tools-agent")
     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=st.secrets["api_key"])
     agent = create_openai_tools_agent(llm, tools, prompt_template)
     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
     st.write(agent_executor.invoke({"input": prompt}))
     st.write("after agent execute")


