"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from ragcheck.core.config import ChunkingConfig, RagcheckConfig


class TestChunkingConfig:
    def test_default_values(self):
        config = ChunkingConfig()
        assert config.strategy == "recursive"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 128

    def test_invalid_chunk_size(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_size=32)  # Below minimum

    def test_invalid_strategy(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(strategy="invalid")  # Not in Literal


class TestRagcheckConfig:
    def test_default_instantiation(self):
        config = RagcheckConfig()
        assert config.project_name == "my-rag-project"
        assert config.chunking.chunk_size == 512

    def test_nested_config_access(self):
        config = RagcheckConfig(chunking=ChunkingConfig(chunk_size=1024))
        assert config.chunking.chunk_size == 1024
