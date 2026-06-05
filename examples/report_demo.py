"""Demo: Generate a full HTML report from synthetic data."""

from pathlib import Path

from ragcheck.analyzers.chunkers import Chunk
from ragcheck.reports.generator import ReportGenerator
from ragcheck.testers.retrieval_tester import RetrievalResult


def main():
    # Simulate test results
    results = [
        RetrievalResult(
            question="What is RAG?",
            expected_answer="Retrieval-Augmented Generation",
            retrieved_chunks=[Chunk("RAG is Retrieval-Augmented Generation", 0, 35, "doc.txt", "fixed")],
            scores=[0.1],
            latency_ms=12.0,
            cost_estimate=0.001,
        ),
        RetrievalResult(
            question="How does chunking work?",
            expected_answer="Splits text into chunks",
            retrieved_chunks=[],
            scores=[],
            latency_ms=8.0,
            cost_estimate=0.001,
        ),
        RetrievalResult(
            question="What are embeddings?",
            expected_answer="Vector representations",
            retrieved_chunks=[
                Chunk("Embeddings are vectors", 0, 22, "doc.txt", "fixed"),
                Chunk("Vectors represent text", 23, 45, "doc.txt", "fixed"),
            ],
            scores=[0.2, 0.3],
            latency_ms=15.0,
            cost_estimate=0.001,
        ),
    ]

    # Add source_chunks attribute to results that need it
    for r in results:
        r.source_chunks = [r.expected_answer]

    chunking_results = {
        "strategy": "recursive",
        "chunks": [
            Chunk("RAG is Retrieval-Augmented Generation", 0, 35, "doc.txt", "recursive"),
            Chunk("Embeddings are vectors", 0, 22, "doc.txt", "recursive"),
            Chunk("Vectors represent text", 23, 45, "doc.txt", "recursive"),
        ],
    }

    generator = ReportGenerator()
    html = generator.generate(results, chunking_results, project_name="ragcheck-demo")

    output_path = Path("ragcheck_report.html")
    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to: {output_path.absolute()}")
    print("Open it in your browser to see the full report.")


if __name__ == "__main__":
    main()
