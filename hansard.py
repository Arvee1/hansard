import streamlit as st
import ollama
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
# from chromadb.utils import embedding_functions
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection
'''
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
st.write("after ChromaDB client create")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )
'''
st.write("after embedding function create")
