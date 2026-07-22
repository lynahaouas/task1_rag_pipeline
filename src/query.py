"""
query.py
Takes a user question, retrieves the top-X most relevant chunks from
ChromaDB, and uses an LLM (via Ollama) to generate a final natural-language
answer grounded in those retrieved chunks.
"""

from openai import OpenAI
from src.embeddings import embed_texts
from src.vector_store import query_chunks

SYSTEM_PROMPT = (
    
    "You are an assistant answering questions about Magyar Telekom's residential "
    "terms and conditions and pricing appendices. The context provided below is in "
    "Hungarian, since that is the original language of the source documents. "
    "Answer ONLY using the information contained in the provided context. If the "
    "answer is not contained in the context, say clearly that you don't have enough "
    "information to answer, instead of guessing.\n\n"
    "Always respond in clear, fluent English, regardless of the language of the "
    "question or the context, since this produces more reliable and accurate output."
)


def retrieve_context(question: str, top_k: int = 3) -> list[dict]:
    """
    Embeds the question and retrieves the top_k closest chunks from the database.
    Returns a list of dicts with the chunk text and its source metadata.
    """
    query_embedding = embed_texts([question], is_query=True)[0]
    results = query_chunks(query_embedding, top_k=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        context_chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "distance": dist,
        })
    return context_chunks


def generate_answer(question: str, context_chunks: list[dict], client: OpenAI, model: str = "qwen2.5") -> str:
    """
    Sends the question + retrieved context to the LLM to generate a grounded answer.
    """
    context_text = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk in context_chunks
    )

    user_message = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def run_query(question: str, top_k: int = 3) -> str:
    """
    Full RAG flow: retrieve relevant chunks, then generate a final answer.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    context_chunks = retrieve_context(question, top_k=top_k)

    print(f"\n--- Retrieved top {top_k} chunks ---")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"{i}. [{chunk['source']}] (distance: {chunk['distance']:.4f})")
        print(f"   {chunk['text'][:150]}...")

    answer = generate_answer(question, context_chunks, client)
    return answer


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    final_answer = run_query(question, top_k=top_k)
    print("\n--- FINAL ANSWER ---")
    print(final_answer)