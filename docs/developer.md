# RAG Pipeline - Developer Documentation

## Architecture Overview
PDFs (data/raw_pdfs/)
│
▼
ingest.py
├─▶ pypdf: extract raw text from each PDF
├─▶ RecursiveCharacterTextSplitter: split into ~1000-char overlapping chunks
├─▶ embeddings.py: embed each chunk (multilingual-e5-large, "passage:" prefix)
└─▶ vector_store.py: store chunks + embeddings + metadata in ChromaDB (persistent, local)

User question
│
▼
query.py
├─▶ embeddings.py: embed the question ("query:" prefix)
├─▶ vector_store.py: retrieve top_k closest chunks
└─▶ Ollama (qwen2.5): generate a final answer grounded in retrieved chunks

## Key Design Decisions

- **Multilingual embedding model (`intfloat/multilingual-e5-large`)**: Source
  documents are Hungarian. An English-centric embedding model would represent
  Hungarian semantics poorly, hurting retrieval accuracy. This model was
  trained across many languages, including Hungarian.

- **`"query:"` / `"passage:"` prefixes**: Required by the e5 model family's
  training scheme to align query and document embedding spaces correctly.
  Skipping this measurably reduces retrieval quality.

- **Chunking (1000 chars, 150 overlap)**: Balances retrieval precision
  (smaller chunks match specific questions better) against context
  preservation (overlap avoids losing meaning at chunk boundaries).

- **ChromaDB (persistent, local)**: Satisfies the local-execution requirement
  with zero external dependencies; data survives between runs.

- **Generation in English regardless of question/context language**: Testing
  showed the local model (Qwen2.5, run via Ollama) has strong Hungarian
  *comprehension* but noticeably weaker Hungarian *generation* fluency
  (occasional invented words). Forcing English output produces more reliable,
  coherent answers while retrieval (proven accurate) still happens directly
  on the original Hungarian text. A larger commercial model (e.g., GPT-4o)
  would likely generate fluent Hungarian directly, if that's a hard
  requirement in production.

- **Anti-hallucination via system prompt**: The LLM is explicitly instructed
  to answer only from retrieved context and to state when it doesn't have
  enough information, rather than guessing. This was verified during testing:
  when the top-k retrieved chunks didn't fully cover a question, the system
  correctly reported it lacked sufficient information instead of fabricating
  an answer.

## Known Limitations / Possible Improvements

- Retrieval quality depends on how closely a question's phrasing matches the
  document's own terminology (e.g., using the contract's own term "rendes
  felmondás" retrieves better results than a generic paraphrase).
- Document metadata could be extended to flag "active" vs. "discontinued"
  pricing appendices (based on filename conventions `ertekesitheto` vs.
  `lezart`) to avoid retrieving outdated pricing information for
  "current price" questions.
- `top_k` is currently a manual, per-query parameter; a production system
  might auto-tune this based on result distance/confidence scores.

## Extending

To add new documents, drop additional PDFs into `data/raw_pdfs/` and re-run
`./run.sh ingest` — existing chunks are not duplicated if the same filenames
are reprocessed (ChromaDB updates by ID).