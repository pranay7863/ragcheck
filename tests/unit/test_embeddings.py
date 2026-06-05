"""Tests for embedding module."""

import numpy as np

from ragcheck.core.embeddings import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    def test_embed_returns_numpy(self):
        embedder = SentenceTransformerEmbedder()
        texts = ["Hello world", "Another sentence"]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.dimension

    def test_dimension_positive(self):
        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension > 0

    def test_single_text(self):
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed(["Single text"])
        assert embeddings.shape == (1, embedder.dimension)

    def test_empty_list(self):
        embedder = SentenceTransformerEmbedder()
        embeddings = embedder.embed([])
        assert embeddings.shape[0] == 0
