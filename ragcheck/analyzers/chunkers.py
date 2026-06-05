"""Chunking strategies for ragcheck."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Chunk:
    """Represents a text chunk."""

    text: str
    start: int
    end: int
    source: str
    strategy: str


class Chunker(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, text: str, source: str) -> list[Chunk]:
        """Split text into chunks."""
        ...


class FixedSizeChunker:
    """Fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size - 1) if chunk_size > 1 else 0

    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                while end > start and text[end] not in " \n\t":
                    end -= 1

            chunks.append(Chunk(
                text=text[start:end].strip(),
                start=start,
                end=end,
                source=source,
                strategy="fixed",
            ))
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks


class SemanticChunker:
    """Semantic chunking using sentence boundaries."""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size
        try:
            import nltk  # type: ignore[import-untyped]
            nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize  # type: ignore[import-untyped]
            self.sent_tokenize = sent_tokenize
        except ImportError:
            self.sent_tokenize = lambda text: text.split(". ")

    def chunk(self, text: str, source: str) -> list[Chunk]:
        sentences = self.sent_tokenize(text)
        chunks: list[Chunk] = []
        current_chunk: list[str] = []
        current_size = 0
        start_idx = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start=start_idx,
                    end=start_idx + len(chunk_text),
                    source=source,
                    strategy="semantic",
                ))
                start_idx += len(chunk_text)
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start=start_idx,
                end=start_idx + len(chunk_text),
                source=source,
                strategy="semantic",
            ))

        return chunks


class RecursiveChunker:
    """Hierarchical recursive chunking."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size - 1) if chunk_size > 1 else 0
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._split(text, 0, source, chunks)
        return chunks

    def _split(self, text: str, start: int, source: str, chunks: list[Chunk]) -> None:
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(Chunk(
                    text=text.strip(),
                    start=max(0, start),
                    end=max(0, start) + len(text),
                    source=source,
                    strategy="recursive",
                ))
            return

        for separator in self.separators:
            if separator not in text:
                continue
            parts = text.split(separator)
            if len(parts) <= 1:
                continue

            current_chunk = ""
            current_start = start
            emitted = False

            for _i, part in enumerate(parts):
                candidate = current_chunk + (separator if current_chunk else "") + part
                if len(candidate) > self.chunk_size and current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        start=max(0, current_start),
                        end=max(0, current_start) + len(current_chunk),
                        source=source,
                        strategy="recursive",
                    ))
                    current_start += len(current_chunk) - self.chunk_overlap
                    current_chunk = part
                    emitted = True
                else:
                    current_chunk = candidate

            if current_chunk:
                if emitted:
                    self._split(current_chunk, max(0, current_start), source, chunks)
                else:
                    self._force_split(text, start, source, chunks)
            return

        self._force_split(text, start, source, chunks)

    def _force_split(self, text: str, start: int, source: str, chunks: list[Chunk]) -> None:
        end = min(self.chunk_size, len(text))
        chunks.append(Chunk(
            text=text[:end].strip(),
            start=max(0, start),
            end=max(0, start) + end,
            source=source,
            strategy="recursive",
        ))
        if len(text) > end:
            next_start = max(0, start + end - self.chunk_overlap)
            self._split(text[end - self.chunk_overlap:], next_start, source, chunks)


class MarkdownChunker:
    """Chunking that respects markdown structure."""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, source: str) -> list[Chunk]:
        import re
        header_pattern = r'^(#{1,6}\s+.+)$'
        sections = re.split(f'(?={header_pattern})', text, flags=re.MULTILINE)

        chunks: list[Chunk] = []
        pos = 0
        for section in sections:
            if not section.strip():
                continue
            if len(section) > self.max_chunk_size:
                recursive = RecursiveChunker(self.max_chunk_size)
                sub_chunks = recursive.chunk(section, source)
                for sc in sub_chunks:
                    sc.strategy = "markdown"
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    text=section.strip(),
                    start=pos,
                    end=pos + len(section),
                    source=source,
                    strategy="markdown",
                ))
            pos += len(section)

        return chunks


class AgenticChunker:
    """Agentic chunking — skeleton, falls back to semantic."""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size
        self._fallback = SemanticChunker(max_chunk_size)

    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks = self._fallback.chunk(text, source)
        for c in chunks:
            c.strategy = "agentic"
        return chunks


class LateChunker:
    """Late chunking — skeleton, falls back to semantic."""

    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size
        self._fallback = SemanticChunker(max_chunk_size)

    def chunk(self, text: str, source: str) -> list[Chunk]:
        chunks = self._fallback.chunk(text, source)
        for c in chunks:
            c.strategy = "late"
        return chunks


class ChunkerFactory:
    """Factory for creating chunkers."""

    @staticmethod
    def create(strategy: str, **kwargs: Any) -> Chunker:
        match strategy:
            case "fixed":
                return FixedSizeChunker(**kwargs)
            case "semantic":
                return SemanticChunker(**kwargs)
            case "recursive":
                return RecursiveChunker(**kwargs)
            case "markdown":
                return MarkdownChunker(**kwargs)
            case "agentic":
                return AgenticChunker(**kwargs)
            case "late":
                return LateChunker(**kwargs)
            case _:
                raise ValueError(f"Unknown chunking strategy: {strategy}")


def benchmark_chunking(text: str, source: str, strategies: list[str]) -> dict[str, Any]:
    """Benchmark multiple chunking strategies."""
    results: dict[str, Any] = {}
    for strategy in strategies:
        chunker = ChunkerFactory.create(strategy)
        chunks = chunker.chunk(text, source)

        avg_length = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0
        min_length = min(len(c.text) for c in chunks) if chunks else 0
        max_length = max(len(c.text) for c in chunks) if chunks else 0
        num_chunks = len(chunks)

        context_loss = sum(1 for chunk in chunks if chunk.text and chunk.text[-1] not in ".!?:")
        context_loss_score = context_loss / num_chunks if num_chunks else 0

        results[strategy] = {
            "num_chunks": num_chunks,
            "avg_length": avg_length,
            "min_length": min_length,
            "max_length": max_length,
            "context_loss_score": context_loss_score,
            "chunks": chunks,
        }

    return results
