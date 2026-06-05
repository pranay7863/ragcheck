"""End-to-end demo of ragcheck."""

from pathlib import Path

from ragcheck.analyzers.chunkers import ChunkerFactory, benchmark_chunking
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore
from ragcheck.reports.generator import ReportGenerator
from ragcheck.testers.auto_qa import TestQuestion, generate_questions
from ragcheck.testers.retrieval_tester import DenseRetriever, run_retrieval_tests


def main() -> None:
    """Run full ragcheck pipeline on sample data."""
    # Sample text
    text = """RAG (Retrieval-Augmented Generation) is a technique that combines
    retrieval systems with generative AI models. It was introduced to address
    the limitations of pure parametric knowledge in large language models.

    The retrieval component fetches relevant documents from a knowledge base.
    The generation component then produces an answer conditioned on both
    the query and the retrieved documents.

    Key challenges in RAG include chunking strategy selection, embedding
    model choice, and retrieval accuracy. Poor chunking can split important
    context across boundaries. Wrong embedding models can fail to capture
    semantic similarity."""

    print("=== ragcheck Demo ===\n")

    # 1. Chunk
    print("1. Chunking with 6 strategies...")
    benchmark = benchmark_chunking(text, "demo.txt", [
        "fixed", "semantic", "recursive", "markdown", "agentic", "late"
    ])
    for strategy, metrics in benchmark.items():
        print(f"   {strategy:10s}: {metrics['num_chunks']} chunks, "
              f"avg {metrics['avg_length']:.0f} chars, "
              f"loss {metrics['context_loss_score']:.1%}")

    # 2. Embed & Store
    print("\n2. Embedding & storing chunks...")
    chunker = ChunkerFactory.create("recursive")
    chunks = chunker.chunk(text, "demo.txt")
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.embed([c.text for c in chunks])
    store = ChromaVectorStore()
    store.add_chunks(chunks, embeddings)
    print(f"   Embedded {len(chunks)} chunks into {embedder.dimension}D vectors")

    # 3. Generate questions
    print("\n3. Generating test questions...")
    questions = [
        TestQuestion(
            question="What is RAG?",
            expected_answer="Retrieval-Augmented Generation",
            source_chunks=[chunks[0].text],
            difficulty="easy",
        ),
        TestQuestion(
            question="What does the retrieval component do?",
            expected_answer="fetches relevant documents",
            source_chunks=[chunks[1].text],
            difficulty="medium",
        ),
    ]
    print(f"   Created {len(questions)} test questions")

    # 4. Test retrieval
    print("\n4. Testing retrieval...")
    retriever = DenseRetriever(embedder, store)
    results, metrics = run_retrieval_tests(questions, retriever, top_k=3)
    print(f"   Tests: {sum(1 for r in results if r.hit_at_k)}/{len(results)} passed")
    print(f"   Avg latency: {metrics.avg_latency_ms:.1f}ms")

    # 5. Generate report
    print("\n5. Generating report...")
    generator = ReportGenerator()
    report_html = generator.generate(
        test_results=results,
        chunking_results={"strategy": "recursive", "chunks": chunks},
        project_name="ragcheck-demo",
    )
    output_path = Path("ragcheck_demo_report.html")
    output_path.write_text(report_html, encoding="utf-8")
    print(f"   Report saved to: {output_path.absolute()}")
    print("\nOpen the HTML file in your browser to see the full report.")


if __name__ == "__main__":
    main()
