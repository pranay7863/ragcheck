"""Tests for chunking strategies."""

import pytest

from ragcheck.analyzers.chunkers import (
    AgenticChunker,
    ChunkerFactory,
    FixedSizeChunker,
    LateChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    benchmark_chunking,
)


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        text = "This is a test document. " * 50
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) > 1
        assert all(len(c.text) <= 100 for c in chunks)
        assert chunks[0].strategy == "fixed"

    def test_short_text(self):
        text = "Short text."
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_overlap_clamped(self):
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=50)
        assert chunker.chunk_overlap < chunker.chunk_size


class TestSemanticChunker:
    def test_sentence_boundaries(self):
        text = "First sentence. Second sentence. Third sentence."
        chunker = SemanticChunker(max_chunk_size=50)
        chunks = chunker.chunk(text, "test.txt")
        assert all("." in c.text for c in chunks)

    def test_single_long_sentence(self):
        text = "This is one very long sentence that exceeds the chunk size limit."
        chunker = SemanticChunker(max_chunk_size=20)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) >= 1


class TestRecursiveChunker:
    def test_hierarchical_split(self):
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunker = RecursiveChunker(chunk_size=30)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) >= 2
        assert all(c.strategy == "recursive" for c in chunks)
        assert all(c.start >= 0 for c in chunks)
        assert all(c.end >= c.start for c in chunks)

    def test_no_negative_indices(self):
        text = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=50)
        chunks = chunker.chunk(text, "test.txt")
        assert all(c.start >= 0 for c in chunks)
        assert all(c.end >= c.start for c in chunks)

    def test_force_split_fallback(self):
        text = "a" * 200
        chunker = RecursiveChunker(chunk_size=50)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) > 1
        assert all(c.strategy == "recursive" for c in chunks)


class TestMarkdownChunker:
    def test_header_split(self):
        text = "# Header 1\nContent 1\n\n# Header 2\nContent 2"
        chunker = MarkdownChunker(max_chunk_size=512)
        chunks = chunker.chunk(text, "test.md")
        # With max_chunk_size=512, the entire text fits in one chunk
        # because it's only ~42 chars. The chunker respects headers when
        # text exceeds max_chunk_size or when forced to split.
        assert len(chunks) >= 1
        assert all(c.strategy == "markdown" for c in chunks)
        # The chunk should contain both headers
        full_text = " ".join(c.text for c in chunks)
        assert "Header 1" in full_text
        assert "Header 2" in full_text

    def test_header_split_with_small_size(self):
        text = "# Header 1\nContent 1\n\n# Header 2\nContent 2"
        chunker = MarkdownChunker(max_chunk_size=20)
        chunks = chunker.chunk(text, "test.md")
        # With small size, should split and fallback to recursive
        assert len(chunks) > 1
        assert all(c.strategy == "markdown" for c in chunks)

    def test_fallback_to_recursive(self):
        text = "# Header\n" + "word " * 500
        chunker = MarkdownChunker(max_chunk_size=100)
        chunks = chunker.chunk(text, "test.md")
        assert len(chunks) > 1
        assert all(c.strategy == "markdown" for c in chunks)


class TestAgenticChunker:
    def test_produces_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunker = AgenticChunker(max_chunk_size=50)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) >= 1
        assert all(c.strategy == "agentic" for c in chunks)


class TestLateChunker:
    def test_produces_chunks(self):
        text = "First sentence. Second sentence. Third sentence."
        chunker = LateChunker(max_chunk_size=50)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) >= 1
        assert all(c.strategy == "late" for c in chunks)


class TestBenchmarkChunking:
    def test_all_six_strategies(self):
        text = "This is a test. " * 100
        strategies = ["fixed", "semantic", "recursive", "markdown", "agentic", "late"]
        results = benchmark_chunking(text, "test.txt", strategies)

        for s in strategies:
            assert s in results
            assert "avg_length" in results[s]
            assert "context_loss_score" in results[s]
            assert "chunks" in results[s]


class TestChunkerFactory:
    def test_all_strategies(self):
        assert isinstance(ChunkerFactory.create("fixed"), FixedSizeChunker)
        assert isinstance(ChunkerFactory.create("semantic"), SemanticChunker)
        assert isinstance(ChunkerFactory.create("recursive"), RecursiveChunker)
        assert isinstance(ChunkerFactory.create("markdown"), MarkdownChunker)
        assert isinstance(ChunkerFactory.create("agentic"), AgenticChunker)
        assert isinstance(ChunkerFactory.create("late"), LateChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkerFactory.create("invalid")
