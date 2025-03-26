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
    
# Function to download and extract text and metadata from online PDFs
def download_and_process_online_pdf(pdf_url, metadata):
    try:
        # Download the PDF
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_data = response.content
        
        # Save to a local file
        pdf_name = pdf_url.split("/")[-1]
        with open(pdf_name, "wb") as f:
            f.write(pdf_data)

        # Extract text and metadata from the PDF
        doc = fitz.open(pdf_name)
        chunks = []
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            if text.strip():
                page_metadata = {
                    "title": metadata.get("title", "Unknown Title"),
                    "year": metadata.get("year", "Unknown Year"),
                    "subject": metadata.get("subject", "Unknown Subject"),
                    "page": page_num + 1,  # Page numbers are 1-based
                    "url": pdf_url,  # Include URL for citation
                }
                chunks.append((text, page_metadata))
        return chunks
    except requests.exceptions.RequestException as e:
        print(f"Error downloading or processing PDF from {pdf_url}: {e}")
        return []

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
def load_and_split_new_documents(existing_ids, json_metadata, CHUNK_SIZE, CHUNK_OVERLAP):
    print("Loading local PDF documents...")
    loader = PyPDFDirectoryLoader("data")
    try:
        raw_documents = loader.load()
        print(f"Loaded {len(raw_documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []
    
    print("Loading online PDF documents...")
    new_documents = []
    for entry in json_metadata["pdfs"]:
        pdf_url = entry["url"]
        chunks = download_and_process_online_pdf(pdf_url, entry)
        if chunks:
            new_documents.extend(chunks)
    
    # Filter out already ingested documents (based on URL in metadata)
    new_documents = [doc for doc in new_documents if doc[1]["url"] not in existing_ids]
    
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