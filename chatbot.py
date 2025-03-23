from langchain_chroma import Chroma
from database_ingestion import MistralEmbeddingFunction, EmbeddingManager
import gradio as gr
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("MISTRAL_API_KEY not found. Exiting...")
    exit(1)

# Initialize Mistral client and embedding manager
client = Mistral(api_key=api_key)
embedding_manager = EmbeddingManager(client)  # Pass client to the embedding manager
mistral_embedding = MistralEmbeddingFunction()
model = "mistral-small-latest"

# ChromaDB Configuration
CHROMA_PATH = "chroma_db"

print("Loading existing ChromaDB...")
vector_store = Chroma(
    collection_name="mistral_collection",
    persist_directory=CHROMA_PATH,
    embedding_function=MistralEmbeddingFunction(embedding_manager),  # Pass embedding_manager
)
print("ChromaDB loaded successfully!")

num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})
# call this function for every message added to the chatbot
def stream_response(message, history):

    # Retrieve relevant chunks based on the question
    docs = retriever.invoke(message)

    # Compile retrieved knowledge
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    # Construct prompt
    rag_prompt = f"""
    You are an assistant which answers questions based on provided knowledge.
    While answering, you don't use your internal knowledge, 
    but solely the information in the "The knowledge" section.
    You don't mention anything to the user about the provided knowledge.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """
    messages = [{"role": "user", "content": rag_prompt}]
    partial_message = ""  # To accumulate the response
    
    # Call Mistral API 
    chat_response = client.chat.complete(
        model = model,  
        messages=messages
    )

    # Extract response safely
    try:
        response_text = chat_response.choices[0].message.content
    except AttributeError as e:
        print("Error extracting response:", e)
        response_text = "Error: Unexpected response format."

    yield response_text  # Send response to Gradio

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# Launch the Gradio app
print("Launching Gradio chatbot")
chatbot.launch()