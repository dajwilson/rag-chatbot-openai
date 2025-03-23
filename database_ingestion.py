import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found. Check your .env file.")

client = Mistral(api_key=api_key)

# Configuration
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 30
BATCH_SIZE = 4

class EmbeddingManager:
    def __init__(self, client, min_delay=0.15, max_delay=1.0, batch_size=BATCH_SIZE, stability_threshold=5):
        self.client = client
        self.request_delay = 0.2
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.batch_size = batch_size
        self.stability_count = 0
        self.stability_threshold = stability_threshold

    def process_batch(self, batch):
        while True:
            time.sleep(self.request_delay)
            try:
                response = self.client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                if self.stability_count < self.stability_threshold:
                    self.request_delay = max(self.min_delay, self.request_delay - 0.01)
                    self.stability_count += 1
                return [item.embedding for item in response.data]
            except Exception as e:
                if "429" in str(e):
                    self.request_delay = min(self.max_delay, self.request_delay + 0.1)
                    self.stability_count = 0
                time.sleep(self.request_delay)

    def get_batch_embeddings(self, texts):
        if not texts or all(text.strip() == "" for text in texts):
            return []
        embeddings = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self.process_batch, texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Embedding Batches", unit="batch"):
                pass
        for future in futures:
            embeddings.extend(future.result())
        return embeddings

class MistralEmbeddingFunction:
    def __init__(self, embedding_manager):
        self.embedding_manager = embedding_manager

    def embed_documents(self, texts):
        return self.embedding_manager.get_batch_embeddings(texts)
    
    def embed_query(self, text):
        return self.embedding_manager.get_batch_embeddings([text])[0]

# Initialize ChromaDB
def initialize_chroma(embedding_manager):
    return Chroma(
        collection_name="mistral_collection",
        embedding_function=MistralEmbeddingFunction(embedding_manager),
        persist_directory=CHROMA_PATH,
    )

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
def load_and_split_new_documents(existing_ids):
    print("Loading PDF documents...")
    loader = PyPDFDirectoryLoader("data")
    try:
        raw_documents = loader.load()
        print(f"Loaded {len(raw_documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
    
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

# Query Function
def query_mistral(vector_store, query, k=5):
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)  # Return raw knowledge

# Main Execution
if __name__ == "__main__":
    embedding_manager = EmbeddingManager(client)
    vector_store = initialize_chroma(embedding_manager)
    existing_ids = get_existing_ids(vector_store)
    chunks = load_and_split_new_documents(existing_ids)
    
    if chunks:
        add_new_chunks_to_chroma(vector_store, chunks)
    
    print("ChromaDB is ready for queries.")