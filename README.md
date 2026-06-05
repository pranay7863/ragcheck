# ragcheck - Lighthouse for RAG Systems

[![PyPI version](https://badge.fury.io/py/ragcheck-cli.svg)](https://badge.fury.io/py/ragcheck-cli)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> One command to diagnose your RAG pipeline and get actionable fixes.

```bash
pip install ragcheck-cli
ragcheck init
ragcheck run --docs ./data --query "What is Article 370?"
```

## What is ragcheck?

**ragcheck** is a lightweight, one-command diagnostic CLI that generates a beautiful, shareable HTML report analyzing why your RAG system fails and how to fix it.

Think of it as **Lighthouse for RAG systems** — just like Lighthouse audits web pages, ragcheck audits your retrieval pipeline.

## Features

- **Auto-Generated Test Suite** - 50 synthetic questions from your documents
- **Chunk Visualizer** - See exactly where your chunking breaks
- **Retrieval Heatmap** - Identify dead chunks and dominant chunks
- **Failure Classification** - Know WHY your RAG fails, not just THAT it fails
- **Actionable Recommendations** - Specific fixes with predicted impact
- **CI/CD Integration** - Fail builds when RAG quality regresses

## Quick Start

### Installation

```bash
pip install ragcheck-cli
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv tool install ragcheck-cli
```

### Initialize

```bash
ragcheck init
```

Creates a `ragcheck.yaml` config file in your project.

### Run Analysis

```bash
ragcheck run --docs ./data --query "Your test query"
```

Generates `ragcheck_report.html` with:
- Scorecards (retrieval accuracy, faithfulness)
- Chunk boundary visualization
- Retrieval heatmap
- Failure mode classification
- Before/after score predictions

### CI Mode

```bash
ragcheck run --docs ./data --ci --min-score 0.80
```

Returns exit code 0/1. Use in GitHub Actions to fail builds on quality regression.

## Example Report

![ragcheck report](https://raw.githubusercontent.com/pranay7863/ragcheck/main/docs/report-screenshot.png)

## Architecture

```
ragcheck CLI
    ├── Chunk Analyzer (6 strategies + benchmark)
    ├── Retriever Tester (auto-QA + dense retrieval)
    ├── Failure Classifier (4 failure modes)
    ├── Recommendation Engine (decision tree)
    └── Report Engine (Jinja2 + CSS/HTML)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| CLI | Typer + Rich |
| Config | Pydantic |
| Embeddings | sentence-transformers |
| Vector DB | ChromaDB |
| LLM Interface | LiteLLM |
| Reports | Jinja2 + CSS/HTML |

## Configuration

`ragcheck.yaml`:

```yaml
project_name: ragcheck
docs_path: ./data
chunking:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 128
llm:
  provider: openai
  model: gpt-3.5-turbo
retrieval:
  top_k: 5
  similarity_threshold: 0.7
report:
  format: html
  include_heatmap: true
```

## Development

```bash
git clone https://github.com/pranay7863/ragcheck.git
cd ragcheck
uv sync
uv run pytest
uv run ruff check .
uv run mypy ragcheck/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE)

## Roadmap

- [x] v0.2.0 — Offline reports, NLI faithfulness, scaled auto-QA, chunk viz
- [ ] v0.3.0 — More vector DBs (Pinecone, Weaviate)
- [ ] v0.3.0 — SaaS API for teams
- [ ] v0.4.0 — Enterprise features (SSO, audit logs)

## Support

- [GitHub](https://github.com/pranay7863/ragcheck)
- Twitter: [@mane_pranay](https://twitter.com/mane_pranay)

---

**Built with discipline.** Read the [blueprint](docs/ARCHITECTURE.md) that started it all.
