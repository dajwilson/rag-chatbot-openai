import os
import time
import requests
from uuid import uuid4
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.embedding_manager import EmbeddingManager
from utils.database_utils import initialize_chroma, get_existing_ids, load_and_split_new_documents, add_new_chunks_to_chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found. Check your .env file.")

# Configuration
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 30
BATCH_SIZE = 4
client = Mistral(api_key=api_key)

# Main Execution
if __name__ == "__main__":
    embedding_manager = EmbeddingManager(client, batch_size=BATCH_SIZE)
    vector_store = initialize_chroma(embedding_manager, CHROMA_PATH)
    existing_ids = get_existing_ids(vector_store)
    chunks = load_and_split_new_documents(existing_ids, CHUNK_SIZE, CHUNK_OVERLAP)

    if chunks:
        add_new_chunks_to_chroma(vector_store, chunks)
    
    print("ChromaDB is ready for queries.")