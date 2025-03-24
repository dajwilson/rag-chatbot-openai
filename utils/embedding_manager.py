import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, client, min_delay=0.15, max_delay=1.0, batch_size=None, stability_threshold=5):
        self.client = client
        self.request_delay = 0.2
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.batch_size = batch_size if batch_size is not None else 5 
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
    
    def embed_documents(self, texts):
        return self.get_batch_embeddings(texts)
    
    def embed_query(self, text):
        return self.get_batch_embeddings([text])[0]