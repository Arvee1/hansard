__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

collection = client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=embedding_func,
     metadata={"hnsw:space": "cosine"},
 )

with open("hansard-utf8.txt") as f:
    hansard = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([hansard])

documents = text_splitter.split_text(hansard)[:len(texts)]
st.write(documents)

collection.add(
     documents=documents,
     ids=[f"id{i}" for i in range(len(documents))],
)

# number of rows
st.write(len(collection.get()['documents']))

# prompt = ("What are the key questions that Senator Cash asks? What were on notice?")

# The UI Part
st.title("👨‍💻 Chat with the Hansard Estimates - 2023")
st.write("Please enter enter your API Key.")
apikey = st.text_area("Enter API Key")

st.write("Please enter what you want to know from the hearing for the Employment Department.")
prompt = st.text_area("What do you want to know?")

if st.button("Submit to AI", type="primary"):
     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=100,
     )
     augment_query = str(query_results["documents"])
    
     client_AI = OpenAI(api_key=apikey)
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
       max_tokens=1000,
       top_p=1
     )
     
     st.write(response.choices[0].message.content)

