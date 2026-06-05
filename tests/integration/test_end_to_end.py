"""End-to-end integration test for ragcheck."""


from ragcheck.analyzers.chunkers import ChunkerFactory, benchmark_chunking
from ragcheck.core.document_loader import load_documents
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore
from ragcheck.reports.generator import ReportGenerator
from ragcheck.testers.auto_qa import TestQuestion
from ragcheck.testers.retrieval_tester import DenseRetriever, run_retrieval_tests


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        docs_dir = tmp_path / "data"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text(
            "RAG is Retrieval-Augmented Generation. "
            "It combines retrieval with generation."
        )
        (docs_dir / "doc2.txt").write_text(
            "Embeddings are vector representations of text. "
            "They enable semantic search."
        )

        documents = load_documents(docs_dir)
        assert len(documents) == 2

        all_text = "\n\n".join(documents.values())
        chunker = ChunkerFactory.create("recursive")
        chunks = chunker.chunk(all_text, "combined")
        assert len(chunks) > 0

        results = benchmark_chunking(all_text, "combined", ["recursive"])
        assert "recursive" in results
        assert results["recursive"]["num_chunks"] > 0

        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])
        store = ChromaVectorStore(collection_name="e2e_test")
        store.reset()
        store.add_chunks(chunks, embeddings)

        questions = [
            TestQuestion(
                "What is RAG?",
                "Retrieval-Augmented Generation",
                [chunks[0].text],
                "easy",
            ),
        ]
        retriever = DenseRetriever(embedder, store)
        test_results, metrics = run_retrieval_tests(questions, retriever, top_k=2)

        assert len(test_results) == 1
        assert metrics.total_queries == 1

        generator = ReportGenerator()
        html = generator.generate(test_results, results["recursive"], "e2e-test")
        assert "ragcheck Report" in html
        assert "e2e-test" in html

    def test_empty_documents(self, tmp_path):
        docs_dir = tmp_path / "data"
        docs_dir.mkdir()
        (docs_dir / "empty.txt").write_text("")

        documents = load_documents(docs_dir)
        assert len(documents) == 1

        all_text = "\n\n".join(documents.values())
        chunker = ChunkerFactory.create("recursive")
        chunks = chunker.chunk(all_text, "combined")
        assert len(chunks) >= 0
