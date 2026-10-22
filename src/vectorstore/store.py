import chromadb


client = chromadb.PersistentClient(path="./chroma_db")


# ChromaDB
# │
# └── Collection : dogs
#     │
#     ├── Chunk 1 → Embedding 1
#     ├── Chunk 2 → Embedding 2
#     ├── Chunk 3 → Embedding 3
#     └── ...

# distance cosinus(A,B) = 1 - simelariteCosinus(A,B)
collection = client.get_or_create_collection(
    name="dogs",
    configuration={
        "hnsw": {"space": "cosine"}   
    } 
)


def store_embeddings(chunks, embeddings) : 
    
    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": "dogs.txt"}
            for _ in chunks
        ]
    )

    print(f"Stored {len(chunks)} chunks")