"""
Store Enhanced Chunks in ChromaDB

Uses the enhanced chunker output and stores in ChromaDB
with Azure OpenAI embeddings.
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Configuration
ENHANCED_CHUNKS_PATH = "ENHANCED_CHUNKS.json"
CHROMA_PERSIST_DIR = "./chroma_db_enhanced"
COLLECTION_NAME = "enhanced_ocr_documents"


def get_embeddings():
    """Get Azure OpenAI embeddings."""
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_resource = os.getenv("OPENAI_RESOURCE")
    azure_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    if azure_api_key and azure_resource:
        azure_endpoint = f"https://{azure_resource}.openai.azure.com/"
        return AzureOpenAIEmbeddings(
            azure_deployment=azure_embedding_deployment,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key.strip(),
            api_version=azure_api_version,
        )
    raise ValueError("Azure OpenAI credentials not found")


def load_enhanced_chunks(file_path: str) -> list:
    """Load enhanced chunks from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def store_chunks(chunks: list, doc_id: str = "TAX_INVOICE"):
    """Store chunks in ChromaDB."""
    print(f"\n{'='*60}")
    print("🗄️  STORING ENHANCED CHUNKS IN CHROMADB")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    
    # Prepare data
    ids = []
    texts = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_id}_chunk_{i}")
        texts.append(chunk["text"])
        
        metadata = {
            "doc_id": doc_id,
            "chunk_index": i,
            "content_type": chunk["metadata"].get("content_type", "text"),
            "page_number": chunk["metadata"].get("page_number") or 0,
            "section": chunk["metadata"].get("section") or "",
        }
        
        # Add table-specific metadata
        if "table_index" in chunk["metadata"]:
            metadata["table_index"] = chunk["metadata"]["table_index"]
        if "row_index" in chunk["metadata"]:
            metadata["row_index"] = chunk["metadata"]["row_index"]
        if "headers" in chunk["metadata"]:
            metadata["headers"] = ", ".join(chunk["metadata"]["headers"])
        
        metadatas.append(metadata)
    
    # Get embeddings
    print("\n🔗 Initializing Azure OpenAI Embeddings...")
    embeddings = get_embeddings()
    
    # Create ChromaDB
    print("🗃️  Creating ChromaDB collection...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    # Clear existing
    try:
        existing = vectorstore.get(where={"doc_id": doc_id})
        if existing and existing['ids']:
            print(f"🗑️  Deleting {len(existing['ids'])} existing chunks...")
            vectorstore.delete(ids=existing['ids'])
    except:
        pass
    
    # Add chunks
    print(f"📤 Adding {len(texts)} chunks...")
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    print(f"\n✅ Successfully stored {len(texts)} chunks!")
    return vectorstore


def query_and_display(vectorstore, query: str, k: int = 5):
    """Query and display results."""
    print(f"\n{'='*60}")
    print(f"🔍 Query: '{query}'")
    print(f"{'='*60}")
    
    results = vectorstore.similarity_search_with_score(query=query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        print(f"\n[Result {i}] (Score: {score:.4f})")
        print(f"📍 Type: {metadata.get('content_type')} | Page: {metadata.get('page_number')} | Section: {metadata.get('section')}")
        
        # Show content preview
        content = doc.page_content
        # Remove header for cleaner display
        if content.startswith("[Source:"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"📄 Content: {preview}")


def main():
    # Load chunks
    print("📂 Loading enhanced chunks...")
    chunks = load_enhanced_chunks(ENHANCED_CHUNKS_PATH)
    print(f"   Loaded {len(chunks)} chunks")
    
    # Store in ChromaDB
    vectorstore = store_chunks(chunks)
    
    # Test queries
    print(f"\n{'='*60}")
    print("🧪 TESTING QUERIES")
    print(f"{'='*60}")
    
    test_queries = [
        "Who is the shipper?",
        "What is the total amount due?",
        "What are the charges?",
        "shipping details",
        "customs broker",
    ]
    
    for query in test_queries:
        query_and_display(vectorstore, query, k=3)


if __name__ == "__main__":
    main()

