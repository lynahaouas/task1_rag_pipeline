"""
embeddings.py
Wraps the multilingual embedding model so ingestion and querying
use the exact same embedding logic - critical for accurate similarity search.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"

_model = None  # loaded once, reused across calls


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Converts a list of texts into embeddings.

    The multilingual-e5 model expects a prefix depending on whether
    you're embedding a document chunk or a search query - this is part
    of how the model was trained, and skipping it hurts retrieval accuracy.
    """
    model = get_model()

    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + text for text in texts]

    embeddings = model.encode(prefixed_texts, normalize_embeddings=True)
    return embeddings.tolist()


if __name__ == "__main__":
    sample_texts = ["Ez egy teszt mondat.", "This is a test sentence."]
    vectors = embed_texts(sample_texts, is_query=False)
    print(f"Number of embeddings: {len(vectors)}")
    print(f"Embedding dimension: {len(vectors[0])}")