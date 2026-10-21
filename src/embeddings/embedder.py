from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"



model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

def generate_embeddings(chunks : list[str]):
    embeddings = model.encode(
        chunks, 
        batch_size=8, 
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings