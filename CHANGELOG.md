# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-06-04

### Added
- **Offline HTML reports** — Replaced Plotly CDN with pure CSS/HTML charts. Reports work without internet.
- **Real faithfulness scoring** — NLI model support (`--nli-model`) for verifying generated answers against retrieved chunks. Falls back to heuristic overlap check.
- **Answer generation** — `--generate-answers` flag wires LiteLLM to populate `RetrievalResult.generated_answer` for faithfulness evaluation.
- **Scaled auto-QA** — Increased from 3 to 50 synthetic test questions with perplexity-based filtering to remove trivial questions.
- **Chunk visualizer integration** — Merged standalone `chunk_visualizer.py` into main report as dedicated "Chunk Analysis" section with histogram and expandable previews.
- **RAGAS re-added** — Optional extra `pip install ragcheck[ragas]` with proper version pin (`>=0.4.0,<0.5.0`).
- **Windows compatibility** — UTF-8 encoding fixes in `config_loader.py`, removed Unicode checkmark causing `cp1252` encoding errors.
- **Local model support** — Zero-cost operation via Ollama (`--answer-model ollama/phi3:mini`).

### Fixed
- Histogram bin calculation for tiny datasets (no more backwards ranges like `276–275`)
- Faithfulness showing `0%` instead of `N/A` when `--generate-answers` is not used
- `FutureWarning` from `sentence-transformers` embedding dimension method

## [0.1.0] - 2026-06-02

### Added
- Typer CLI with `init`, `run`, `report` commands
- 6 chunking strategies: fixed, semantic, recursive, markdown, agentic, late
- Chunk visualization with Plotly histograms
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- ChromaDB vector store
- Auto-QA generation via LiteLLM
- Dense retriever with latency/cost tracking
- Failure classification: 4 modes (retrieval miss, hallucination, overload, boundary error)
- Recommendation engine with decision tree
- Beautiful HTML reports (single file, no server)
- CI/CD mode with GitHub Actions
- PDF/PNG export via Playwright