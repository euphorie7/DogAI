from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from embeddings.embedder import generate_embeddings


def main() : 

    # 1. Load documents
    documents = load_documents("data/dogs.txt")

    # 2. Chunking
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    # 3. Generate embeddings
    embeddings = generate_embeddings(chunks)


    # 4. Store embeddings
    store_embeddings(chunks, embeddings)

    print(f"Documents separeted with --*--: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {embeddings.shape}")

