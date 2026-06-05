"""Tests for retrieval tester."""


from ragcheck.analyzers.chunkers import Chunk
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore
from ragcheck.testers.auto_qa import TestQuestion
from ragcheck.testers.retrieval_tester import (
    CostMetrics,
    DenseRetriever,
    RetrievalResult,
    run_retrieval_tests,
)


class TestRetrievalResult:
    def test_hit_at_k(self):
        result = RetrievalResult(
            question="Q",
            expected_answer="A",
            retrieved_chunks=[Chunk("test", 0, 4, "a.txt", "fixed")],
            scores=[0.5],
            latency_ms=10.0,
            cost_estimate=0.001,
        )
        assert result.hit_at_k is True

    def test_mrr(self):
        result = RetrievalResult(
            question="Q",
            expected_answer="A",
            retrieved_chunks=[
                Chunk("a", 0, 1, "a.txt", "fixed"),
                Chunk("b", 0, 1, "a.txt", "fixed"),
            ],
            scores=[0.3, 0.8],
            latency_ms=10.0,
            cost_estimate=0.001,
        )
        assert result.mrr == 0.5  # max at index 1 -> 1/2


class TestCostMetrics:
    def test_defaults(self):
        m = CostMetrics()
        assert m.avg_latency_ms == 0.0
        assert m.estimated_cost_usd == 0.0

    def test_avg_latency(self):
        m = CostMetrics(total_queries=2, total_latency_ms=100.0)
        assert m.avg_latency_ms == 50.0


class TestDenseRetriever:
    def test_retrieve_returns_results(self):
        store = ChromaVectorStore(collection_name="test_dense")
        store.reset()

        chunks = [
            Chunk(text="RAG retrieval", start=0, end=13, source="doc.txt", strategy="fixed"),
            Chunk(text="LLM generation", start=14, end=28, source="doc.txt", strategy="fixed"),
        ]
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        retriever = DenseRetriever(embedder, store)
        results = retriever.retrieve("What is RAG?", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r[0], Chunk) for r in results)


class TestRunRetrievalTests:
    def test_runs_all_questions(self):
        store = ChromaVectorStore(collection_name="test_run")
        store.reset()

        chunks = [
            Chunk(text="RAG is great", start=0, end=12, source="doc.txt", strategy="fixed"),
            Chunk(text="LLMs are powerful", start=13, end=30, source="doc.txt", strategy="fixed"),
        ]
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        questions = [
            TestQuestion("What is RAG?", "A retrieval technique", ["RAG is great"], "easy"),
            TestQuestion("What are LLMs?", "Large language models", ["LLMs are powerful"], "easy"),
        ]

        retriever = DenseRetriever(embedder, store)
        results, metrics = run_retrieval_tests(questions, retriever, top_k=2)

        assert len(results) == 2
        assert metrics.total_queries == 2
        assert metrics.total_latency_ms > 0
