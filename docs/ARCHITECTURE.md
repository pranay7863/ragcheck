# ragcheck Architecture

## Overview

```
ragcheck CLI (Typer + Rich)
    |
    +-- Document Loader (Text, Markdown)
    |
    +-- Chunk Analyzer (6 strategies)
    |       +-- Fixed-size, Semantic, Recursive
    |       +-- Markdown-aware, Agentic, Late
    |
    +-- Embedding Manager (sentence-transformers)
    |
    +-- Vector Store (ChromaDB)
    |
    +-- Retriever Tester (DenseRetriever)
    |       +-- Auto-QA Generation (LiteLLM)
    |       +-- Latency/Cost Tracking
    |
    +-- Failure Classifier (4 modes)
    |       +-- Retrieval Miss
    |       +-- Context Overload
    |       +-- Hallucination
    |       +-- Chunk Boundary Error
    |
    +-- Recommendation Engine (Decision Tree)
    |
    +-- Report Engine (Jinja2 + Plotly)
            +-- HTML Report (single file)
            +-- PDF/PNG Export (Playwright)
```

## Design Principles

1. **Zero-infrastructure**: `pip install ragcheck` works out of the box
2. **Single-file output**: HTML report is one file, no server needed
3. **Framework agnostic**: No LangChain or LlamaIndex dependency in core
4. **Offline-first**: Core metrics use local models; LLM calls are optional
