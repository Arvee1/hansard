__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import replicate

openai_api_key = st.secrets["api_key"]

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

# The UI Part
st.title("👨‍💻 Wazzup!!!! Let's Chat with the Hansard Senate Estimates for Employment Department (DEWR) - 3 June 2024")
# apikey = st.sidebar.text_area("Please enter enter your API Key.")
apikey = openai_api_key
prompt = st.text_area("Please enter what you want to know from the hearing for the Employment Department.")


# Load VectorDB
# if st.sidebar.button("Load OFSC Facsheets into Vector DB if loading the page for the first time.", type="primary"):
@st.cache_resource
def create_vector():
      with open("Education and Employment Legislation Committee_2024_06_03.txt") as f:
          hansard = f.read()
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=500,
              chunk_overlap=20,
              length_function=len,
              is_separator_regex=False,
          )
           
      texts = text_splitter.create_documents([hansard])
      documents = text_splitter.split_text(hansard)[:len(texts)]
     
      collection.add(
           documents=documents,
           ids=[f"id{i}" for i in range(len(documents))],
      )
      f.close()

create_vector()

if st.button("Submit to DJ Arvee", type="primary"):
     query_results = collection.query(
          query_texts=[prompt],
          # include=["documents", "embeddings"],
          include=["documents"],
          n_results=20,
          # n_results=75,
     )
     augment_query = str(query_results["documents"])

     result_ai = ""
     # The meta/llama-2-7b-chat model can stream output as it's running.
     # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
     for event in replicate.stream(
          "meta/meta-llama-3-70b-instruct",
          input={
               "top_k": 50,
               "top_p": 0.9,
               "prompt": "Prompt: " + prompt + " " + augment_query,
               "max_tokens": 512,
               "min_tokens": 0,
               "temperature": 0.6,
               "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
               "presence_penalty": 1.15,
               "frequency_penalty": 0.2
          },
     ):
          result_ai = result_ai + (str(event))
     st.write(result_ai)

     # client_AI = OpenAI(api_key=apikey)
     # response = client_AI.chat.completions.create(
       # model="gpt-3.5-turbo",
       # messages=[
         # {
           # "role": "system",
           # "content": "You are a friendly assistant."
         # },
         # {
           # "role": "user",
           # "content": augment_query + " Prompt: " + prompt
         # }
       # ],
       # temperature=0.1,
       # max_tokens=1000,
       # top_p=1
     # )
     
     # st.write(response.choices[0].message.content)

