from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import generate_embeddings
from src.vectorstore.store import store_embeddings



def main() : 

    # 1. Load documents
    print("1. Chargement des documents...", flush=True)
    documents = load_documents("data/dogs.txt")
    print(f"Documents séparés : {len(documents)}", flush=True)


    # 2. Chunking
    print("2. Chunking...", flush=True)
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )
    print(f"Chunks : {len(chunks)}", flush=True)
    # 3. Generate embeddings
    print("3. Génération des embeddings...", flush=True)
    embeddings = generate_embeddings(chunks)
    print(f"Embeddings : {embeddings.shape}", flush=True)


    # 4. Store embeddings
    print("4. Storing !", flush=True)
    store_embeddings(chunks, embeddings)
    print("5. Terminé !", flush=True)
  

if __name__ == "__main__":
    main()


