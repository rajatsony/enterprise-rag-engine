"""PDF document extraction and chunking pipeline.

Extracts text and tables from PDF documents, processes them through a vision model,
and aggregates them into coherent text chunks for vectorization.
"""

from unstructured.partition.pdf import partition_pdf
import re


def extract_and_aggregate(pdf_path):
    """Extract and chunk text from a PDF document.
    
    Uses high-resolution PDF parsing to preserve document structure.
    Separates tables and text, removes headers/footers, and aggregates
    fragmented text into coherent chunks.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        List of text chunks ready for embedding.
    """
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res", 
        infer_table_structure=True,
    )
    
    raw_extractions = []
    
    # Extract and classify elements (text vs. tables, exclude headers/footers)
    for element in elements:
        element_type = str(type(element))
        
        if "Header" in element_type or "Footer" in element_type:
            continue
            
        elif "Table" in element_type:
            html_table = element.metadata.text_as_html
            if html_table:
                raw_extractions.append({
                    "type": "table",
                    "content": f"\n[START_TABLE]\n{html_table}\n[END_TABLE]\n"
                })
        else:
            text_content = element.text.replace(
                "JPMorgan Chase & Co./2025 Form 10-K", ""
            ).strip()
            if text_content.isdigit() or not text_content:
                continue
            raw_extractions.append({"type": "text", "content": text_content})
    
    # Aggregate fragmented text while preserving table boundaries
    final_chunks = []
    current_chunk_text = ""
    
    for item in raw_extractions:
        if item["type"] == "table":
            # Save current text chunk before inserting table
            if current_chunk_text:
                final_chunks.append(current_chunk_text.strip())
                current_chunk_text = ""
            final_chunks.append(item["content"])
        elif item["type"] == "text":
            # Continue sentence if not complete, else start new paragraph
            if current_chunk_text and not current_chunk_text.endswith("."):
                current_chunk_text += " " + item["content"]
            else:
                current_chunk_text += ("\n\n" + item["content"] 
                                     if current_chunk_text else item["content"])
    
    if current_chunk_text:
        final_chunks.append(current_chunk_text.strip())
    
    return final_chunks


if __name__ == "__main__":
    file_path = "raw_data/practice_doc.pdf" 
    final_documents = extract_and_aggregate(file_path)
    
    for i, doc in enumerate(final_documents[:10]): 
        print(f"Chunk {i + 1}")
        print(doc)
        print()