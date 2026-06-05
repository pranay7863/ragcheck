"""Tests for vector store module."""


from ragcheck.analyzers.chunkers import Chunk
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    def test_add_and_search(self):
        store = ChromaVectorStore(collection_name="test_add_search")
        store.reset()

        chunks = [
            Chunk(text="RAG is great", start=0, end=12, source="doc1.txt", strategy="fixed"),
            Chunk(text="LLMs are powerful", start=13, end=30, source="doc1.txt", strategy="fixed"),
            Chunk(text="Python is nice", start=0, end=14, source="doc2.txt", strategy="semantic"),
        ]
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])

        store.add_chunks(chunks, embeddings)
        query_emb = embedder.embed(["What is RAG?"])[0]
        results = store.search(query_emb, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r[0], Chunk) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_reset_clears_data(self):
        store = ChromaVectorStore(collection_name="test_reset")
        store.reset()

        chunks = [Chunk(text="test", start=0, end=4, source="a.txt", strategy="fixed")]
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        store.reset()
        query_emb = embedder.embed(["test"])[0]
        results = store.search(query_emb, top_k=5)
        assert len(results) == 0

    def test_search_returns_scores(self):
        store = ChromaVectorStore(collection_name="test_scores")
        store.reset()

        chunks = [
            Chunk(text="apple", start=0, end=5, source="fruit.txt", strategy="fixed"),
            Chunk(text="banana", start=10, end=16, source="fruit.txt", strategy="fixed"),
            Chunk(text="car", start=20, end=23, source="vehicle.txt", strategy="fixed"),
        ]
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(chunks, embeddings)

        query_emb = embedder.embed(["fruit"])[0]
        results = store.search(query_emb, top_k=3)

        # Scores should be non-negative (Chroma uses distances)
        assert all(score >= 0 for _, score in results)
