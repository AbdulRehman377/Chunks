"""
ChromaDB Storage for Azure OCR Chunks

This script:
1. Loads processed OCR chunks from VECTORDB_READY.json
2. Creates embeddings using Azure OpenAI or Chroma's default
3. Stores everything in ChromaDB with full metadata
4. Provides query functions with source attribution

Requirements (from your requirements.txt):
- chromadb
- langchain-chroma  
- langchain-openai
- python-dotenv
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

# ChromaDB
import chromadb
from chromadb.config import Settings

# LangChain + OpenAI Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VECTORDB_READY_PATH = "VECTORDB_READY.json"
CHROMA_PERSIST_DIR = "./chroma_db"  # Where ChromaDB will store data
COLLECTION_NAME = "ocr_documents"

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_vectordb_chunks(file_path: str = VECTORDB_READY_PATH) -> dict:
    """Load the processed vector DB chunks."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_chunks_for_chroma(chunks: list, doc_id: str = "TAX_INVOICE") -> tuple:
    """
    Prepare chunks for ChromaDB insertion.
    
    Returns:
        tuple: (ids, texts, metadatas)
    """
    ids = []
    texts = []
    metadatas = []
    
    for chunk in chunks:
        chunk_id = f"{doc_id}_{chunk['metadata']['chunk_id']}"
        ids.append(chunk_id)
        texts.append(chunk["text"])
        
        # Build metadata - ChromaDB requires primitive types (str, int, float, bool)
        metadata = {
            "doc_id": doc_id,
            "chunk_id": chunk["metadata"]["chunk_id"],
            "chunk_type": chunk["metadata"]["chunk_type"],
            "page_number": chunk["metadata"].get("page_number") or 0,
            "role": chunk["metadata"].get("role") or "",
            "span_offset": chunk["metadata"].get("span_offset") if chunk["metadata"].get("span_offset") is not None else -1,
            "span_length": chunk["metadata"].get("span_length") if chunk["metadata"].get("span_length") is not None else -1,
        }
        
        # Add table-specific metadata
        if chunk["metadata"].get("table_index") is not None:
            metadata["table_index"] = chunk["metadata"]["table_index"]
            metadata["table_row"] = chunk["metadata"].get("table_row") if chunk["metadata"].get("table_row") is not None else -1
            metadata["table_column"] = chunk["metadata"].get("table_column") if chunk["metadata"].get("table_column") is not None else -1
            metadata["cell_kind"] = chunk["metadata"].get("cell_kind") or ""
        else:
            metadata["table_index"] = -1
            metadata["table_row"] = -1
            metadata["table_column"] = -1
            metadata["cell_kind"] = ""
        
        # Convert bounding_box to string (ChromaDB doesn't support lists in metadata)
        if chunk["metadata"].get("bounding_box"):
            metadata["bounding_box"] = json.dumps(chunk["metadata"]["bounding_box"])
        else:
            metadata["bounding_box"] = ""
        
        metadatas.append(metadata)
    
    return ids, texts, metadatas


# ═══════════════════════════════════════════════════════════════════════════════
# CHROMADB OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_embeddings():
    """
    Get embeddings instance.
    Priority: Azure OpenAI → OpenAI → Chroma default (sentence-transformers)
    """
    # Try Azure OpenAI first
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_resource = os.getenv("OPENAI_RESOURCE")
    azure_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    
    # Check for embedding model deployment name (you may need to set this)
    azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    if azure_api_key and azure_resource:
        print(f"   Using Azure OpenAI Embeddings")
        print(f"   Resource: {azure_resource}")
        print(f"   Deployment: {azure_embedding_deployment}")
        
        azure_endpoint = f"https://{azure_resource}.openai.azure.com/"
        
        return AzureOpenAIEmbeddings(
            azure_deployment=azure_embedding_deployment,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key.strip(),
            api_version=azure_api_version,
        )
    
    # Try regular OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("   Using OpenAI Embeddings")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
    
    # Fall back to Chroma's default embeddings (sentence-transformers)
    print("   ⚠️  No OpenAI/Azure keys found, using Chroma's default embeddings")
    print("   (sentence-transformers/all-MiniLM-L6-v2 - runs locally)")
    return None  # ChromaDB will use default embedding function


def create_chroma_collection(persist_directory: str = CHROMA_PERSIST_DIR, embeddings=None):
    """Create or get existing ChromaDB collection with embeddings."""
    if embeddings is None:
        embeddings = get_embeddings()
    
    if embeddings is not None:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    else:
        # Use ChromaDB's default embedding function
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            persist_directory=persist_directory,
        )
    
    return vectorstore


def store_chunks_in_chroma(
    chunks: list,
    doc_id: str = "TAX_INVOICE",
    persist_directory: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """
    Store chunks in ChromaDB with embeddings.
    
    Args:
        chunks: List of chunk dicts from VECTORDB_READY.json
        doc_id: Document identifier
        persist_directory: Where to persist ChromaDB
    
    Returns:
        Chroma vectorstore instance
    """
    print("\n" + "="*60)
    print("🗄️  CHROMADB STORAGE")
    print("="*60)
    
    # Prepare data
    ids, texts, metadatas = prepare_chunks_for_chroma(chunks, doc_id)
    
    print(f"📄 Document ID: {doc_id}")
    print(f"📝 Total chunks to store: {len(texts)}")
    print(f"💾 Persist directory: {persist_directory}")
    
    # Get embeddings
    print("\n🔗 Initializing Embeddings...")
    embeddings = get_embeddings()
    
    # Create/connect to ChromaDB
    print("\n🗃️  Connecting to ChromaDB...")
    
    if embeddings is not None:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    else:
        # Use ChromaDB with default embeddings
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_directory,
        )
    
    # Check if documents already exist
    try:
        existing_docs = vectorstore.get(where={"doc_id": doc_id})
        if existing_docs and existing_docs['ids']:
            print(f"⚠️  Found {len(existing_docs['ids'])} existing chunks for '{doc_id}'")
            print("🗑️  Deleting existing chunks...")
            vectorstore.delete(ids=existing_docs['ids'])
    except Exception as e:
        print(f"   Note: {e}")
    
    # Add documents
    print(f"\n📤 Adding {len(texts)} chunks to ChromaDB...")
    print("   (Creating embeddings...)")
    
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    print("\n✅ Successfully stored all chunks!")
    
    # Verify
    try:
        collection_count = vectorstore._collection.count()
        print(f"📊 Total documents in collection: {collection_count}")
    except:
        print("📊 Documents stored successfully")
    
    return vectorstore


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global embeddings cache to avoid re-initializing
_embeddings_cache = None

def query_chroma(
    query: str,
    k: int = 5,
    filter_page: Optional[int] = None,
    filter_type: Optional[str] = None,
    filter_table: Optional[int] = None,
    persist_directory: str = CHROMA_PERSIST_DIR
) -> list:
    """
    Query ChromaDB with optional filters.
    
    Args:
        query: Search query
        k: Number of results
        filter_page: Filter by page number
        filter_type: Filter by chunk type (paragraph, table_cell, table)
        filter_table: Filter by table index
        persist_directory: ChromaDB directory
    
    Returns:
        List of results with text and metadata
    """
    global _embeddings_cache
    
    # Get or create embeddings
    if _embeddings_cache is None:
        _embeddings_cache = get_embeddings()
    
    vectorstore = create_chroma_collection(persist_directory, _embeddings_cache)
    
    # Build filter
    where_filter = None
    if filter_page or filter_type or filter_table is not None:
        conditions = []
        if filter_page:
            conditions.append({"page_number": filter_page})
        if filter_type:
            conditions.append({"chunk_type": filter_type})
        if filter_table is not None:
            conditions.append({"table_index": filter_table})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}
    
    # Search
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=where_filter
    )
    
    return results


def format_query_results(results: list) -> str:
    """Format query results with source attribution."""
    formatted_parts = []
    
    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        
        # Build source info
        source_parts = []
        
        if metadata.get("page_number") and metadata["page_number"] > 0:
            source_parts.append(f"Page {metadata['page_number']}")
        
        if metadata.get("role"):
            source_parts.append(f"Type: {metadata['role']}")
        
        if metadata.get("table_index", -1) >= 0:
            table_info = f"Table {metadata['table_index'] + 1}"
            if metadata.get("table_row", -1) >= 0:
                table_info += f", Row {metadata['table_row'] + 1}"
            if metadata.get("table_column", -1) >= 0:
                table_info += f", Col {metadata['table_column'] + 1}"
            if metadata.get("cell_kind"):
                table_info += f" ({metadata['cell_kind']})"
            source_parts.append(table_info)
        
        source_str = " | ".join(source_parts) if source_parts else "Document"
        
        formatted_parts.append(
            f"[Result {i}] (Score: {score:.4f})\n"
            f"📍 Source: {source_str}\n"
            f"📄 Content: {doc.page_content}\n"
        )
    
    return "\n".join(formatted_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main function to store and query OCR data in ChromaDB."""
    
    # 1. Load processed chunks
    print("📂 Loading VECTORDB_READY.json...")
    data = load_vectordb_chunks()
    chunks = data["chunks"]
    doc_metadata = data["document_metadata"]
    
    print(f"   Loaded {len(chunks)} chunks from {doc_metadata['total_pages']} pages")
    
    # 2. Store in ChromaDB
    vectorstore = store_chunks_in_chroma(chunks)
    
    # 3. Demo queries
    print("\n" + "="*60)
    print("🔍 DEMO QUERIES")
    print("="*60)
    
    # Query 1: General search
    print("\n📝 Query 1: 'What is the total amount due?'")
    print("-"*50)
    results = query_chroma("What is the total amount due?", k=5)
    print(format_query_results(results))
    
    # Query 2: With page filter
    print("\n📝 Query 2: 'shipping details' (Page 1 only)")
    print("-"*50)
    results = query_chroma("shipping details", k=5, filter_page=1)
    print(format_query_results(results))
    
    # Query 3: Table-specific
    print("\n📝 Query 3: 'charges' (Table chunks only)")
    print("-"*50)
    results = query_chroma("charges breakdown", k=5, filter_type="table")
    print(format_query_results(results))
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"\n💾 ChromaDB persisted at: {CHROMA_PERSIST_DIR}")
    print(f"📄 VECTORDB_READY.json preserved at: {VECTORDB_READY_PATH}")
    print("\nYou can now query your document using:")
    print("  from chromadb_store import query_chroma, format_query_results")
    print("  results = query_chroma('your question')")
    print("  print(format_query_results(results))")


if __name__ == "__main__":
    main()

