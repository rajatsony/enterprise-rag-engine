"""Semantic search and ranking pipeline.

Provides efficient document retrieval using bi-encoder embedding search (FAISS)
followed by cross-encoder reranking for improved result relevance.
"""

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_system():
    """Load embedding models and vector store.
    
    Returns:
        Tuple of (bi_encoder, cross_encoder, faiss_index, text_chunks).
    """
    bi_encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    index = faiss.read_index("jpmorgan.faiss")
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    return bi_encoder, cross_encoder, index, chunks
    
def search_and_rerank(query, bi_encoder, cross_encoder, index, chunks, top_k=15):
    """Retrieve and rerank documents using two-stage ranking.
    
    First stage uses bi-encoder embeddings with FAISS for fast retrieval.
    Second stage applies cross-encoder scoring to improve result relevance.
    
    Args:
        query: Search query text.
        bi_encoder: Bi-encoder model for initial retrieval.
        cross_encoder: Cross-encoder model for result reranking.
        index: FAISS vector index.
        chunks: List of document text chunks.
        top_k: Number of candidates to retrieve before reranking.
        
    Returns:
        Tuple of (initial_retrieval_results, reranked_results).
    """
    # Stage 1: Retrieve candidates using bi-encoder
    query_vector = bi_encoder.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = index.search(query_vector, top_k)
    retrieved_chunks = [chunks[i] for i in faiss_indices[0]]
    
    # Stage 2: Rerank using cross-encoder
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    cross_scores = cross_encoder.predict(pairs)
    chunk_scores = list(zip(retrieved_chunks, cross_scores))
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    reranked_chunks = [chunk for chunk, score in chunk_scores]
    
    return retrieved_chunks, reranked_chunks[:3]

def calculate_mrr(target_substring, ranked_chunks):
    """Calculate Mean Reciprocal Rank for evaluation.
    
    Args:
        target_substring: Text to search for in ranked results.
        ranked_chunks: List of results in rank order.
        
    Returns:
        Reciprocal rank (1.0 / rank_position) if found, else 0.0.
    """
    for index, chunk in enumerate(ranked_chunks):
        if target_substring in chunk:
            return 1.0 / (index + 1)
    return 0.0

def run_evaluation(query, target_substring, bi_enc, cross_enc, faiss_idx, chunk_data):
    """Evaluate ranking quality for a given query.
    
    Args:
        query: Test query.
        target_substring: Text that should be retrieved.
        bi_enc: Bi-encoder model.
        cross_enc: Cross-encoder model.
        faiss_idx: FAISS index.
        chunk_data: Document chunks.
    """
    faiss_results, cross_results = search_and_rerank(
        query, bi_enc, cross_enc, faiss_idx, chunk_data, top_k=5
    )
    
    faiss_mrr = calculate_mrr(target_substring, faiss_results)
    cross_mrr = calculate_mrr(target_substring, cross_results)
    
    print(f"\nQuery: '{query}'")
    print(f"Target: '{target_substring}'")
    print(f"Bi-Encoder MRR:  {faiss_mrr:.2f}")
    print(f"Cross-Encoder MRR: {cross_mrr:.2f}")

# Test Execution
if __name__ == "__main__":
    bi_enc, cross_enc, faiss_idx, chunk_data = load_system()
    
    # Let's ask a hyper-specific financial question based on the text we saw earlier
    test_query = "decrease in deposit is because of which factor?"
    target_substring = """Cash and due from banks and deposits with banks decreased driven by Markets activities in CIB, higher investment securities, higher loans and cash deployment in Treasury and ClO"""

    faiss_results, cross_results = search_and_rerank(
        test_query, bi_enc, cross_enc, faiss_idx, chunk_data
    )
    run_evaluation(test_query, target_substring, bi_enc, cross_enc, faiss_idx, chunk_data)

    # print("FAISS Top 5 Results:")
    # for i, res in enumerate(faiss_results[:5]):
    #     print(f"{i+1}. {res[:200]}...")
    # print("\nCross-Encoder Top 5 Results:")
    # for i, res in enumerate(cross_results):
    #     print(f"{i+1}. {res[:200]}...")