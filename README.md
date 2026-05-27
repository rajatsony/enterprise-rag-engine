# Enterprise Financial RAG Engine

A secure, multi-stage Retrieval-Augmented Generation (RAG) pipeline designed for dense 10-K financial reports. Built entirely locally using open-source models.

## Architecture Highlights
* **Ingestion:** Uses YOLOX Vision Models (`unstructured.io`) to bypass standard PDF parsing errors, preserving multi-column layouts and extracting clean HTML tables.
* **Retrieval:** Implements an asymmetric search using BGE-small Bi-Encoders (FAISS) and re-ranks context using an MS-MARCO Cross-Encoder to optimize Mean Reciprocal Rank (MRR).
* **Security:** Features a zero-latency Semantic Firewall (`np.dot` matrix math) to intercept prompt injections and enforce deterministic guardrails before LLM generation.

## How to Run
1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
2. Run `chunking_pipeline.py` to extract text from the PDF.
3. Run `vector_store.py` to generate the FAISS index.
4. Run `app.py` to launch the secure end-to-end engine.