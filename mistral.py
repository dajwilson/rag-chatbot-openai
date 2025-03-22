from mistralai import Mistral
import requests
import numpy as np
import faiss
import os

api_key= os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

print('starting get request')
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
print('finished get request')
text = response.text

f = open('essay.txt', 'w')
f.write(text)
f.close()

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
print(len(chunks))

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings.create(
          model="mistral-embed",
          inputs=input
      )
    return embeddings_batch_response.data[0].embedding
text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

question = "What were the two main things the author worked on before college?"
question_embeddings = np.array([get_text_embedding(question)])

D, I = index.search(question_embeddings, k=2) # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

prompt = f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-small-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

print(run_mistral(prompt))
print("-- end of script--")