"""Vector store creation and management.

Builds and persists FAISS indices for efficient semantic search over document embeddings.
"""

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


def create_and_save_vector_store(chunks, index_path="jpmorgan.faiss", meta_path="chunks.json"):
    """Create and persist a FAISS vector store from document chunks.
    
    Generates embeddings using BGE (BAAI/bge-small-en-v1.5) and builds an IndexFlatIP
    index for efficient cosine similarity search.
    
    Args:
        chunks: List of document text chunks to index.
        index_path: Output path for FAISS index file.
        meta_path: Output path for chunk metadata JSON.
        
    Returns:
        Tuple of (faiss_index, embedding_model).
    """
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    # Normalize embeddings for cosine similarity via inner product
    embeddings = model.encode(chunks, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)

    # Persist index and metadata
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    return index, model


if __name__ == "__main__":
    from chunking_pipeline import extract_and_aggregate
    chunks = extract_and_aggregate("raw_data\\jpmc-10k-2025.pdf")
    create_and_save_vector_store(chunks)