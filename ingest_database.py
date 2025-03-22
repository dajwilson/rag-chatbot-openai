import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from mistralai import Mistral
import requests
import numpy as np
import faiss
from getpass import getpassl

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
LLM = ""
    
# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)

try:
    raw_documents = loader.load()
except Exception as e:
    print(f"Error loading document: {e}")
    raw_documents = []
    
if not raw_documents:
    print("No documents loaded; aborting further processing.")
    exit(1)
else:
    loaded_files = set([doc.metadata['source'] for doc in raw_documents])
    print(f"{len(loaded_files)} PDF files fully loaded.")

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

if LLM == "Mistral":
    def get_text_embedding(input):
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
      )
    return embeddings_batch_response.data[0].embedding
    text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
    
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model= model,
        messages = [
            {
                "role": "user",
                "content": "What is the best French cheese?",
            },
        ]
    )
    print(chat_response.choices[0].message.content)
else:
    # adding chunks to vector store
    try:  
        vector_store.add_documents(documents=chunks, ids=uuids)
    except Exception as e:
        print(f"Error calling API: {e}")
    