# RAG Pipeline - User Guide

Answers questions about Magyar Telekom's residential Terms and Conditions
and pricing appendices using a local RAG (Retrieval-Augmented Generation) pipeline.

## Requirements

- Python 3.10+
- Ollama installed and running locally, with the `qwen2.5` model pulled

## Setup

1. Place the source PDFs (T&Cs + appendices) in `data/raw_pdfs/`.
2. Run ingestion once (or whenever documents change):
```bash
   ./run.sh ingest
```
   This extracts, chunks, embeds, and stores all documents in a local vector database (`db/`).

## Asking Questions

```bash
./run.sh query "your question here" [top_k]
```

- `top_k` (optional, default 3): how many of the closest matching document
  excerpts to use when generating the answer.

## Example

```bash
./run.sh query "Milyen kötbér jár késedelmes fizetés esetén?" 5
```

Output includes:
1. The top matching excerpts retrieved from the source documents (with source filename).
2. A final answer in English, generated only from those excerpts.

## Notes

- Source documents are in Hungarian. The system retrieves and displays
  matching excerpts in their original Hungarian, but generates the final
  answer in English for reliability (see developer docs for details).
- If the retrieved excerpts don't contain enough information to answer a
  question, the system will say so rather than guessing.