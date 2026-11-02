"""
Workload #3: Build a RAG pipeline with ChromaDB
This uses 8 gb ram for embeddings + vector DB
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import time
import psutil
from tqdm import tqdm

print(f"RAM available: {psutil.virtual_memory().available / 1e9:.2f} GB")

#synthetic research papers (1000 document)
print("Generating 1000 synthetic papers...")
papers = []
for i in range(1000):
    papers.append({
        'id': f'paper-{i}',
        'title': f'Research Paper {i}',
        'abstract': f'This paper explores machine learning techniques for {" ".join(np.random.choice(["NLP", "CV", "RL", "LLM", "RAG"], 10))}. ' * 5,
        'content': ' '.join(['Artificial intelligence'] * 100)
    })

# Initialize embedding model (use 1GB RAM)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create ChromaDB (persistent storage)
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection(
    name = "papers",
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = 'all-MiniLM-L6-v2',
    )
)

# add documents to vector DB
print("Inserting into vector DB...")
start = time.time()

for paper in tqdm(papers):
    collection.add(
        documents=[paper['content']],
        metadata=[{'title': paper['title']}],
        ids=[paper['id']]
    )

print(f"Added {len(papers)} papers in {time.time() - start:.2f} seconds")

# Query
print("\n👀 Querying...")
query = "machine learning applications in healthcare"
results = collection.query(
    query_text=[query],
    n_results=5
)

print(f"Top 5 results for '{query}':")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadata'][0])):
    print(f"{i+1}. {meta['title']} - {doc[:100]}...")

print(f"\nRAM used: {psutil.virtual_memory().used / 1e9:.2f} GB")