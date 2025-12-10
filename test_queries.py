"""
Test queries against the enhanced ChromaDB
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_db_enhanced"
COLLECTION_NAME = "enhanced_ocr_documents"


def get_embeddings():
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_resource = os.getenv("OPENAI_RESOURCE")
    azure_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    
    azure_endpoint = f"https://{azure_resource}.openai.azure.com/"
    return AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key.strip(),
        api_version=azure_api_version,
    )


def query(query_text: str, k: int = 5):
    """Run a query and display results."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    print(f"\n{'='*70}")
    print(f"🔍 Query: '{query_text}'")
    print(f"{'='*70}")
    
    results = vectorstore.similarity_search_with_score(query=query_text, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        print(f"\n[Result {i}] (Score: {score:.4f})")
        print(f"📍 Type: {metadata.get('content_type')} | Page: {metadata.get('page_number')} | Section: {metadata.get('section')}")
        
        content = doc.page_content
        if content.startswith("[Source:"):
            content = content.split("]\n\n", 1)[-1] if "]\n\n" in content else content
        
        print(f"📄 Content:\n{content}")
    
    return results


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*70)
    print("🧪 ENHANCED CHUNKS QUERY TESTER")
    print("="*70)
    print("Type your query and press Enter. Type 'exit' to quit.\n")
    
    while True:
        user_query = input("\n🔎 Enter query: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if user_query:
            query(user_query, k=5)

