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

# The UI Part
st.title("👨‍💻 Wazzup!!!! Let's Chat with the Hansard Senate Estimates for Employment Department (DEWR) - 14 Feb 2024")
# apikey = st.sidebar.text_area("Please enter enter your API Key.")
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

     # with open("hansardFeb2024.txt") as f:
         # hansard = f.read()
         # text_splitter = RecursiveCharacterTextSplitter(
             # chunk_size=750,
             # chunk_overlap=50,
             # length_function=len,
             # is_separator_regex=False,
         # )
    
     # texts = text_splitter.create_documents([hansard])
     # documents = text_splitter.split_text(hansard)[:len(texts)]
    
     # collection.add(
          # documents=documents,
          # ids=[f"id{i}" for i in range(len(documents))],
     # )
  
     # number of rows
     # st.write(len(collection.get()['documents']))
     # st.sidebar.write("Hansard Vector DB created. With " + len(collection.get()['documents']) + " rows." )

if st.button("Submit to DJ Arvee", type="primary"):
     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=75,
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


