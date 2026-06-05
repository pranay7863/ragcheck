"""Vector store abstraction for ragcheck."""

from typing import Any, Protocol, cast

import numpy as np

from ragcheck.analyzers.chunkers import Chunk


class VectorStore(Protocol):
    """Protocol for vector stores."""

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks with embeddings to the store."""
        ...

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        ...

    def reset(self) -> None:
        """Clear all data."""
        ...


class ChromaVectorStore:
    """ChromaDB-based vector store."""

    def __init__(self, collection_name: str = "ragcheck") -> None:
        import chromadb
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        ids = [f"{c.source}_{c.start}_{i}" for i, c in enumerate(chunks)]
        texts = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = [
            {"source": c.source, "start": c.start, "end": c.end} for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        raw_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        # Cast ChromaDB's nested return types
        results: dict[str, Any] = cast(dict[str, Any], raw_results)
        ids_list: list[list[str]] = cast(list[list[str]], results.get("ids") or [[]])
        docs_list: list[list[str]] = cast(list[list[str]], results.get("documents") or [[]])
        meta_list: list[list[dict[str, Any]]] = cast(
            list[list[dict[str, Any]]], results.get("metadatas") or [[]]
        )
        dist_list: list[list[float]] = cast(list[list[float]], results.get("distances") or [[]])

        chunks_with_scores: list[tuple[Chunk, float]] = []
        count = len(ids_list[0]) if ids_list and ids_list[0] else 0
        for i in range(count):
            metadata = meta_list[0][i]
            chunk = Chunk(
                text=docs_list[0][i],
                start=int(metadata["start"]),
                end=int(metadata["end"]),
                source=str(metadata["source"]),
                strategy="stored",
            )
            score = dist_list[0][i]
            chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)
