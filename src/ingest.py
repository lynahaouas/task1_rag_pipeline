"""
ingest.py
Reads all PDFs from data/raw_pdfs/, extracts text, splits into chunks,
embeds them, and stores them in the persistent ChromaDB vector database.
Run this whenever documents are added or updated.
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings import embed_texts
from src.vector_store import add_chunks

RAW_PDF_DIR = "data/raw_pdfs"

# Chunk size in characters, with overlap so context isn't lost at chunk boundaries
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def extract_text_from_pdf(filepath: str) -> str:
    """Extracts all text from a single PDF file."""
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def chunk_text(text: str) -> list[str]:
    """Splits a long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def run_ingestion():
    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {RAW_PDF_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting ingestion...")

    for filename in pdf_files:
        filepath = os.path.join(RAW_PDF_DIR, filename)
        print(f"Processing: {filename}")

        text = extract_text_from_pdf(filepath)
        if not text.strip():
            print(f"  Warning: no extractable text found in {filename}, skipping.")
            continue

        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks.")

        embeddings = embed_texts(chunks, is_query=False)

        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        add_chunks(chunks, embeddings, metadatas, ids)
        print(f"  Added {len(chunks)} chunks to the database.")

    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()