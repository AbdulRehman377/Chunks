# query_test.py
from chromadb_store import query_chroma, format_query_results
results = query_chroma("Who is the shipper?", k=10)
print(format_query_results(results))