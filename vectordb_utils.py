"""
Vector DB Utility Functions for Azure OCR Data

This module provides helper functions to:
1. Insert chunks into vector databases (Pinecone, Chroma, Qdrant examples)
2. Query and retrieve data with rich context
3. Format responses with page/table/line information
"""

import json
from typing import Optional


def load_vectordb_chunks(file_path: str = "VECTORDB_READY.json") -> dict:
    """Load the processed vector DB chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# PINECONE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_for_pinecone(chunks: list, doc_id: str = "invoice_001") -> list:
    """
    Prepare chunks for Pinecone upsert.
    
    Pinecone format: {"id": str, "values": [embeddings], "metadata": dict}
    """
    pinecone_records = []
    
    for chunk in chunks:
        # Clean metadata - Pinecone doesn't support nested dicts or null values
        metadata = {
            "doc_id": doc_id,
            "chunk_id": chunk["metadata"]["chunk_id"],
            "chunk_type": chunk["metadata"]["chunk_type"],
            "text": chunk["text"],  # Store text in metadata for retrieval
        }
        
        # Add optional fields if they exist and are not None
        if chunk["metadata"].get("page_number"):
            metadata["page_number"] = chunk["metadata"]["page_number"]
        if chunk["metadata"].get("role"):
            metadata["role"] = chunk["metadata"]["role"]
        if chunk["metadata"].get("table_index") is not None:
            metadata["table_index"] = chunk["metadata"]["table_index"]
            metadata["table_row"] = chunk["metadata"].get("table_row")
            metadata["table_column"] = chunk["metadata"].get("table_column")
            metadata["cell_kind"] = chunk["metadata"].get("cell_kind")
        if chunk["metadata"].get("span_offset") is not None:
            metadata["span_offset"] = chunk["metadata"]["span_offset"]
            metadata["span_length"] = chunk["metadata"]["span_length"]
        
        pinecone_records.append({
            "id": f"{doc_id}_{chunk['metadata']['chunk_id']}",
            "values": [],  # Replace with actual embeddings
            "metadata": metadata
        })
    
    return pinecone_records


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMADB EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_for_chroma(chunks: list, doc_id: str = "invoice_001") -> dict:
    """
    Prepare chunks for ChromaDB collection.add().
    
    ChromaDB format: 
    {
        "ids": list[str], 
        "documents": list[str], 
        "metadatas": list[dict]
    }
    """
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(f"{doc_id}_{chunk['metadata']['chunk_id']}")
        documents.append(chunk["text"])
        
        metadata = {
            "doc_id": doc_id,
            "chunk_id": chunk["metadata"]["chunk_id"],
            "chunk_type": chunk["metadata"]["chunk_type"],
        }
        
        # Add optional fields (Chroma supports None values as empty strings)
        metadata["page_number"] = chunk["metadata"].get("page_number") or 0
        metadata["role"] = chunk["metadata"].get("role") or ""
        metadata["table_index"] = chunk["metadata"].get("table_index") if chunk["metadata"].get("table_index") is not None else -1
        metadata["table_row"] = chunk["metadata"].get("table_row") if chunk["metadata"].get("table_row") is not None else -1
        metadata["table_column"] = chunk["metadata"].get("table_column") if chunk["metadata"].get("table_column") is not None else -1
        metadata["cell_kind"] = chunk["metadata"].get("cell_kind") or ""
        
        metadatas.append(metadata)
    
    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_retrieval_response(results: list) -> str:
    """
    Format vector DB retrieval results with rich context information.
    
    This function takes raw retrieval results and formats them to show
    exactly where the data came from (page, table cell, etc.)
    
    Args:
        results: List of dicts with 'text' and 'metadata' keys
    
    Returns:
        Formatted string with source information
    """
    formatted_parts = []
    
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        
        # Build source info string
        source_parts = []
        
        # Page info
        if metadata.get("page_number"):
            source_parts.append(f"Page {metadata['page_number']}")
        
        # Role info (title, pageNumber, sectionHeading, etc.)
        if metadata.get("role"):
            source_parts.append(f"Type: {metadata['role']}")
        
        # Table info
        if metadata.get("table_index") is not None and metadata.get("table_index") >= 0:
            table_info = f"Table {metadata['table_index'] + 1}"
            if metadata.get("table_row") is not None:
                table_info += f", Row {metadata['table_row'] + 1}"
            if metadata.get("table_column") is not None:
                table_info += f", Col {metadata['table_column'] + 1}"
            if metadata.get("cell_kind"):
                table_info += f" ({metadata['cell_kind']})"
            source_parts.append(table_info)
        
        # Character position (for precise reference back to original)
        if metadata.get("span_offset") is not None:
            source_parts.append(f"Chars {metadata['span_offset']}-{metadata['span_offset'] + metadata.get('span_length', 0)}")
        
        source_str = " | ".join(source_parts) if source_parts else "Unknown source"
        
        formatted_parts.append(
            f"[Result {i}]\n"
            f"📍 Source: {source_str}\n"
            f"📄 Content: {text}\n"
        )
    
    return "\n".join(formatted_parts)


def build_query_filter(
    page_number: Optional[int] = None,
    chunk_type: Optional[str] = None,
    table_index: Optional[int] = None,
    role: Optional[str] = None,
) -> dict:
    """
    Build a filter dict for vector DB queries.
    
    Args:
        page_number: Filter by page (1-indexed)
        chunk_type: "paragraph", "table_cell", or "table"
        table_index: Filter by table number (0-indexed)
        role: Filter by role ("title", "pageNumber", "sectionHeading", etc.)
    
    Returns:
        Filter dict compatible with most vector DBs
    """
    filters = {}
    
    if page_number is not None:
        filters["page_number"] = page_number
    if chunk_type:
        filters["chunk_type"] = chunk_type
    if table_index is not None:
        filters["table_index"] = table_index
    if role:
        filters["role"] = role
    
    return filters


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO: SIMULATE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

def demo_retrieval():
    """
    Demonstrate how retrieval responses would look with the metadata.
    """
    data = load_vectordb_chunks()
    chunks = data["chunks"]
    
    print("="*70)
    print("🔍 DEMO: Simulated Vector DB Retrieval Results")
    print("="*70)
    
    # Simulate a query: "What is the total amount?"
    print("\n📝 Query: 'What is the total amount due?'")
    print("-"*50)
    
    # Manually select relevant chunks (in real scenario, this comes from vector search)
    relevant_keywords = ["TOTAL", "865", "BALANCE"]
    matching_chunks = [
        c for c in chunks 
        if any(kw.lower() in c["text"].lower() for kw in relevant_keywords)
    ][:5]
    
    formatted = format_retrieval_response([
        {"text": c["text"], "metadata": c["metadata"]}
        for c in matching_chunks
    ])
    print(formatted)
    
    # Simulate another query: "What are the charges?"
    print("\n" + "="*70)
    print("\n📝 Query: 'What are the charges breakdown?'")
    print("-"*50)
    
    relevant_keywords = ["CARTAGE", "HANDLING", "CHARGES"]
    matching_chunks = [
        c for c in chunks 
        if any(kw.lower() in c["text"].lower() for kw in relevant_keywords)
    ][:5]
    
    formatted = format_retrieval_response([
        {"text": c["text"], "metadata": c["metadata"]}
        for c in matching_chunks
    ])
    print(formatted)
    
    # Show table-specific query
    print("\n" + "="*70)
    print("\n📝 Query: 'Show me data from tables on page 1'")
    print("-"*50)
    
    table_chunks = [
        c for c in chunks 
        if c["metadata"].get("chunk_type") == "table" 
        and c["metadata"].get("page_number") == 1
    ][:3]
    
    formatted = format_retrieval_response([
        {"text": c["text"], "metadata": c["metadata"]}
        for c in table_chunks
    ])
    print(formatted)


if __name__ == "__main__":
    demo_retrieval()

