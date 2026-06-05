"""Demo script for all 6 chunking strategies."""

from pathlib import Path

from ragcheck.analyzers.chunkers import ChunkerFactory, benchmark_chunking
from ragcheck.reports.chunk_visualizer import generate_chunk_viz


def main():
    sample_text = """# Introduction to RAG

RAG (Retrieval-Augmented Generation) is a technique that combines
retrieval systems with generative AI. It works by retrieving relevant documents
from a knowledge base and then using a large language model to generate answers.

## Key Components

The key components are: a document store, an embedding model,
a vector database, and a language model. Chunking strategy is critical because
poor chunking can split important context across boundaries.

## Chunking Strategies

Common strategies include fixed-size, semantic, recursive, markdown-aware,
agentic (LLM-based), and late chunking (contextual embeddings)."""

    # Benchmark all 6 strategies
    strategies = ["fixed", "semantic", "recursive", "markdown", "agentic", "late"]
    results = benchmark_chunking(sample_text, "sample.md", strategies)

    print("Chunking Benchmark Results — All 6 Strategies")
    print("=" * 60)
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper():12} | Chunks: {metrics['num_chunks']:3} | "
              f"Avg: {metrics['avg_length']:6.1f} | Loss: {metrics['context_loss_score']:.2%}")

    # Generate HTML visualization for markdown chunker (most interesting for this doc)
    md_chunks = results["markdown"]["chunks"]
    html = generate_chunk_viz(md_chunks, "sample.md", "markdown", sample_text)

    output_path = Path("chunk_visualization.html")
    output_path.write_text(html, encoding="utf-8")
    print(f"\nVisualization saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
