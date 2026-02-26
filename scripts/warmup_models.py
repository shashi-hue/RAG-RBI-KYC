from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

print("Downloading bge-large-en-v1.5...")
SentenceTransformer("BAAI/bge-large-en-v1.5")

print("Downloading MiniLM reranker...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

print("Downloading BM25 sparse model...")
list(SparseTextEmbedding("Qdrant/bm25").embed(["warmup"]))

print("All models cached.")
