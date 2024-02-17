__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
# import ollama
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
# from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

client_AI = OpenAI(api_key="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")

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
st.write("after ChromaDB client create")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

st.write("after embedding function create")

# collection = client.create_collection(
collection = client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=embedding_func,
     metadata={"hnsw:space": "cosine"},
 )

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

collection.add(
     documents=documents,
     ids=[f"id{i}" for i in range(len(documents))],
#     metadatas=[{"genre": g} for g in genres]
)

# number of rows
st.write(len(collection.get()['documents']))

prompt = ("What were the key points provided by Ms Jenkins relating to underpayment in the department?")

query_results = collection.query(
     query_texts=[prompt],
     # include=["documents", "embeddings"],
     include=["documents"],
     n_results=100,
 )

# print(query_results["embeddings"])
# st.write(query_results["documents"])

augment_query = str(query_results["documents"])
st.write(augment_query)

response = client_AI.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You are a friendly assistant."
    },
    {
      "role": "user",
      "content": augment_query + " Prompt: " + prompt

    }
  ],
  temperature=0.1,
  max_tokens=64,
  top_p=1
)
st.write(response.choices[0].message.content)

'''
response = ollama.chat(
    model='llama2',
    messages=[
        {
            "role": "system",
            "content": "You are a friendly assistant."
        },
        {
            "role": "user",
            "content": augment_query + " Prompt: " + prompt
        },
    ],
)

st.write(response['message']['content'])
'''
