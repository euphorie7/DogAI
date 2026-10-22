import chromadb

from sentence_transformers import SentenceTransformer



# Charger le même modèle que pour les documents
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cpu"
)

client = chromadb.PersistentClient(path="./chroma_db")

# ChromaDB (./chroma_db)
# │
# ├── Collection "dogs"
# │   ├── Chunk 1 → Texte + Embedding + Métadonnées
# │   ├── Chunk 2 → Texte + Embedding + Métadonnées
# │   └── Chunk 3 → Texte + Embedding + Métadonnées
# │
# └── Collection "cats"
#     ├── Chunk 1 → Texte + Embedding + Métadonnées
#     └── Chunk 2 → Texte + Embedding + Métadonnées

collection = client.get_collection(name="dogs")


def retrieve_documents(question: str, top_k: int = 3):

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    return results