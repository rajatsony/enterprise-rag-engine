"""Enterprise RAG Application.

A secure retrieval-augmented generation system for enterprise financial documents.
Integrates semantic firewalls, vector retrieval, and strict prompt engineering to prevent
model misuse while providing accurate document-based answers.
"""

from guardrail import SemanticFirewall
from search_pipeline import load_system, search_and_rerank


def build_llm_prompt(user_query, retrieved_chunks):
    """Construct a constrained LLM prompt with retrieval-based context.
    
    Args:
        user_query: The user's input question.
        retrieved_chunks: List of relevant document chunks from vector retrieval.
        
    Returns:
        A formatted system prompt with strict instruction boundaries.
    """
    context_block = "\n\n".join([f"--- Chunk {i+1} ---\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    system_prompt = f"""You are a highly secure Financial AI Assistant for a major bank.
You will be provided with context extracted from an official corporate document.

STRICT RULES:
1. You must ONLY answer the user's question using the provided Context.
2. If the answer is not contained within the Context, you must reply EXACTLY with: "I'm sorry, but that information is not available in the provided document."
3. You must NEVER give personal financial advice or recommendations.
4. You must NEVER write code, tell jokes, or perform off-topic tasks.

====================
CONTEXT:
{context_block}
====================

USER QUESTION: {user_query}

ANSWER:"""
    return system_prompt

# Initialize security and retrieval components
firewall = SemanticFirewall()
bi_enc, cross_enc, faiss_idx, chunk_data = load_system()

def run_enterprise_rag(user_query):
    """Execute the complete RAG pipeline with security and ranking.
    
    Args:
        user_query: User's natural language question.
        
    Returns:
        A formatted LLM prompt with context and instructions, or a security alert.
    """
    # Validate query against semantic firewall
    if not firewall.check_query(user_query):
        return "SECURITY ALERT: Request blocked by Semantic Firewall.\n"
    
    # Retrieve and re-rank relevant document chunks
    _, best_chunks = search_and_rerank(user_query, bi_enc, cross_enc, faiss_idx, chunk_data, top_k=3)
    
    # Build constrained LLM prompt with retrieved context
    final_prompt = build_llm_prompt(user_query, best_chunks)
    
    return final_prompt

if __name__ == "__main__":
    # Test case: Blocked query (financial advice request)
    print(run_enterprise_rag("Should I put my 401k into JPM stock?"))
    
    print("-" * 50)
    
    # Test case: Valid document query
    print(run_enterprise_rag("What was the compensation expense?"))