"""Pydantic configuration schema for ragcheck."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""

    strategy: Literal[
        "fixed",
        "semantic",
        "recursive",
        "markdown",
        "agentic",
        "late",
    ] = Field(default="recursive", description="Chunking strategy to use")
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=128, ge=0, le=1024)
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for semantic chunking",
    )


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, ollama, etc.)",
    )
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    api_key: str | None = Field(
        default=None, description="API key (or use env var)"
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class RetrievalConfig(BaseModel):
    """Retriever configuration."""

    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    vector_db: Literal["chromadb", "pinecone", "weaviate"] = Field(
        default="chromadb"
    )
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(default="BAAI/bge-reranker-base")


class ReportConfig(BaseModel):
    """Report generation configuration."""

    format: Literal["html", "json"] = Field(default="html")
    include_heatmap: bool = Field(default=True)
    include_chunk_viz: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)
    theme: Literal["light", "dark"] = Field(default="light")


class RagcheckConfig(BaseModel):
    """Root configuration for ragcheck."""

    project_name: str = Field(default="my-rag-project")
    docs_path: Path = Field(default=Path("./data"))
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    max_test_questions: int = Field(default=50, ge=1, le=200)
    ci_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
