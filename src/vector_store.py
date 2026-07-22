"""
vector_store.py
Handles all interaction with ChromaDB: creating/loading the persistent
collection, adding document chunks, and querying for the closest matches.
"""

import chromadb

DB_PATH = "db"
COLLECTION_NAME = "telekom_tc"


def get_collection():
    """
    Returns a persistent ChromaDB collection, creating it if it doesn't exist.
    Using PersistentClient means data survives between runs (saved to disk),
    satisfying the requirement that the pipeline 'updates the database'.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def add_chunks(chunks: list[str], embeddings: list[list[float]], metadatas: list[dict], ids: list[str]):
    """
    Adds document chunks, their embeddings, and metadata (e.g., source filename)
    to the vector database.
    """
    collection = get_collection()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_chunks(query_embedding: list[float], top_k: int = 3) -> dict:
    """
    Retrieves the top_k closest chunks to the given query embedding.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


if __name__ == "__main__":
    from embeddings import embed_texts

    test_chunks = ["Ez egy teszt mondat a szerződésről.", "Ez egy másik teszt mondat a díjakról."]
    test_embeddings = embed_texts(test_chunks, is_query=False)
    test_metadatas = [{"source": "test.pdf"} for _ in test_chunks]
    test_ids = ["test_chunk_0", "test_chunk_1"]

    add_chunks(test_chunks, test_embeddings, test_metadatas, test_ids)
    print("Chunks added successfully.")

    query_embedding = embed_texts(["Mi a helyzet a díjakkal?"], is_query=True)[0]
    results = query_chunks(query_embedding, top_k=2)
    print("Query results:", results)