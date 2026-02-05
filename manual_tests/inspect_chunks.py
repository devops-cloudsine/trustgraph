#!/usr/bin/env python3
"""
Script to inspect actual chunks stored in Qdrant
"""
import sys
from qdrant_client import QdrantClient
import tiktoken

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Get collection name from user
if len(sys.argv) < 3:
    print("Usage: python inspect_chunks.py <user_id> <collection_id>")
    print("Example: python inspect_chunks.py user_897359257447481fb160ec5c6163b3e7 50e1ecbc-2fd2-4804-9b40-056c8437fe24")
    sys.exit(1)

user = sys.argv[1]
collection_id = sys.argv[2]
collection_name = f"d_{user}_{collection_id}"

print(f"Inspecting collection: {collection_name}")
print("=" * 80)

# Check if collection exists
if not client.collection_exists(collection_name):
    print(f"Collection '{collection_name}' does not exist!")
    sys.exit(1)

# Get collection info
info = client.get_collection(collection_name)
print(f"Total points: {info.points_count}")
print(f"Vector size: {info.config.params.vectors.size}")
print("=" * 80)

# Get some sample chunks
result = client.scroll(
    collection_name=collection_name,
    limit=10,
    with_payload=True,
    with_vectors=False
)

points = result[0]

# Initialize tokenizer (cl100k_base is used by the chunker)
enc = tiktoken.get_encoding("cl100k_base")

print(f"\nFound {len(points)} sample chunks:\n")

for i, point in enumerate(points, 1):
    chunk_text = point.payload.get("doc", "")
    
    # Count tokens
    tokens = enc.encode(chunk_text)
    token_count = len(tokens)
    
    print(f"Chunk {i} (ID: {point.id}):")
    print(f"  Token count: {token_count}")
    print(f"  Character count: {len(chunk_text)}")
    print(f"  Preview: {chunk_text[:200]}...")
    print("-" * 80)

print(f"\nTo see more chunks, modify the 'limit' parameter in the script")
