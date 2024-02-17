__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import ollama
import chromadb
from chromadb.utils import embedding_functions
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
st.write("after ChromaDB client create")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

st.write("after embedding function create")

with open("hansard-utf8.txt") as f:
    hansard = f.read()

text_splitter = RecursiveCharacterTextSplitter(
#     Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([hansard])
st.write(texts[0])
# print(texts[1])
# print(texts[2])
# print(texts[3])

documents = text_splitter.split_text(hansard)[:len(texts)]
st.write(documents)
