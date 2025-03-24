import requests
from uuid import uuid4
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize ChromaDB
def initialize_chroma(embedding_manager, CHROMA_PATH):
    return Chroma(
        collection_name="mistral_collection",
        embedding_function=embedding_manager,
        persist_directory=CHROMA_PATH,
    )
    
def load_json_from_url(url):
    """Fetch the JSON from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON from {url}: {e}")
        return None

# Function to get existing document IDs
def get_existing_ids(vector_store):
    try:
        existing_ids = set(vector_store.get()['ids'])
        print(f"Found {len(existing_ids)} existing document IDs in ChromaDB.")
        return existing_ids
    except Exception as e:
        print(f"Error retrieving existing IDs: {e}")
        return set()

# Function to load and split new documents
def load_and_split_new_documents(existing_ids, CHUNK_SIZE, CHUNK_OVERLAP):
    print("Loading local PDF documents...")
    loader = PyPDFDirectoryLoader("data")
    try:
        raw_documents = loader.load()
        print(f"Loaded {len(raw_documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
    
    print("Loading online PDF documents...")
    try:
        hey = "hey"
    except Exception as e:
        print(f"Error loading documents: {e}")
    
    # Filter out already ingested documents
    new_documents = [doc for doc in raw_documents if doc.metadata['source'] not in existing_ids]
    if not new_documents:
        print("No new documents to process.")
        return []
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(new_documents)
    
    if not chunks:
        print("❌ No chunks were generated. Check PDF processing.")
        exit(1)
    print(f"Total new chunks created: {len(chunks)}")
    
    return chunks

# Function to add new chunks to ChromaDB (with batch handling)
def add_new_chunks_to_chroma(vector_store, chunks, batch_size=5000):
    """Add new document chunks to ChromaDB in smaller batches to prevent exceeding limits."""
    print("Adding new chunks to ChromaDB in batches...")

    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]

            vector_store.add_documents(documents=batch, ids=uuids)
            print(f"✅ Added batch {i // batch_size + 1}/{-(-len(chunks) // batch_size)} ({len(batch)} chunks)")

        print("🎉 All chunks successfully added to ChromaDB!")

    except Exception as e:
        print(f"❌ Error adding new chunks to ChromaDB: {e}")