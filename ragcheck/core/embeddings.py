"""Embedding management for ragcheck."""

from typing import Any, Protocol, cast

from numpy import ndarray


class Embedder(Protocol):
    """Protocol for embedding models."""

    def embed(self, texts: list[str]) -> ndarray[Any, Any]:
        """Embed a list of texts."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        try:
            dim = self.model.get_embedding_dimension()
        except AttributeError:
            dim = self.model.get_sentence_embedding_dimension()
        self._dimension = cast(int, dim)

    def embed(self, texts: list[str]) -> ndarray[Any, Any]:
        return cast(ndarray[Any, Any], self.model.encode(texts, show_progress_bar=False))

    @property
    def dimension(self) -> int:
        return self._dimension