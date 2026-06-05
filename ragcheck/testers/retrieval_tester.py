"""Test retrieval quality against generated questions."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from litellm import completion

from ragcheck.analyzers.chunkers import Chunk
from ragcheck.core.embeddings import Embedder
from ragcheck.core.vector_store import VectorStore
from ragcheck.testers.auto_qa import TestQuestion


@dataclass
class RetrievalResult:
    """Result of a single retrieval test."""

    question: str
    expected_answer: str
    retrieved_chunks: list[Chunk]
    scores: list[float]
    latency_ms: float
    cost_estimate: float
    source_chunks: list[str] = field(default_factory=list)
    generated_answer: str = ""  # Set by the evaluation pipeline after LLM generation

    @property
    def hit_at_k(self) -> bool:
        """Check if any relevant chunk was retrieved."""
        if not self.source_chunks:
            # Weak fallback when auto-QA didn't track sources
            return len(self.retrieved_chunks) > 0

        retrieved_texts = [c.text for c in self.retrieved_chunks]
        return any(
            any(src in rt or rt in src for rt in retrieved_texts)
            for src in self.source_chunks
        )

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank."""
        if not self.scores:
            return 0.0
        best_idx = self.scores.index(max(self.scores))
        return 1.0 / (best_idx + 1)


@dataclass
class CostMetrics:
    """Track costs across the evaluation."""

    total_queries: int = 0
    total_embedding_tokens: int = 0
    total_llm_tokens: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_queries if self.total_queries else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        embedding_cost = self.total_embedding_tokens * 0.0001 / 1000
        llm_cost = self.total_llm_tokens * 0.002 / 1000
        return embedding_cost + llm_cost


class Retriever(Protocol):
    """Protocol for retrievers."""

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Retrieve chunks for a query."""
        ...


class DenseRetriever:
    """Dense retrieval using embeddings."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        import time
        start = time.time()

        query_embedding = self.embedder.embed([query])[0]
        results = self.vector_store.search(query_embedding, top_k)

        _latency = (time.time() - start) * 1000  # noqa: F841
        return results


def _detect_provider(model: str) -> str:
    """Detect provider prefix from model string."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


def _generate_answer(
    question: str,
    retrieved_chunks: list[Chunk],
    model: str = "ollama/phi3:mini",
    api_key: str | None = None,
) -> str:
    """Generate an answer from retrieved chunks using LiteLLM."""
    context = "\n\n".join(c.text for c in retrieved_chunks)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say 'I do not know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    if api_key:
        kwargs["api_key"] = api_key

    # Set base_url for providers that need it (Groq, OpenRouter, etc.)
    provider = _detect_provider(model)
    if provider == "groq":
        kwargs["base_url"] = "https://api.groq.com/openai/v1"
    elif provider == "openrouter":
        kwargs["base_url"] = "https://openrouter.ai/api/v1"

    try:
        response = completion(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        import warnings
        warnings.warn(f"Answer generation failed for model {model}: {e}")
        return ""


def run_retrieval_tests(
    questions: list[TestQuestion],
    retriever: Retriever,
    top_k: int = 5,
    generate_answers: bool = False,
    answer_model: str = "ollama/phi3:mini",
    answer_api_key: str | None = None,
) -> tuple[list[RetrievalResult], CostMetrics]:
    """Run retrieval tests against a set of questions."""
    import time

    results: list[RetrievalResult] = []
    metrics = CostMetrics()

    for q in questions:
        start = time.time()
        retrieved = retriever.retrieve(q.question, top_k)
        latency = (time.time() - start) * 1000

        chunks = [c for c, _ in retrieved]
        scores = [s for _, s in retrieved]

        cost = 0.002 * len(q.question) / 1000

        result = RetrievalResult(
            question=q.question,
            expected_answer=q.expected_answer,
            retrieved_chunks=chunks,
            scores=scores,
            latency_ms=latency,
            cost_estimate=cost,
            source_chunks=getattr(q, "source_chunks", []),
        )

        if generate_answers:
            result.generated_answer = _generate_answer(
                q.question, chunks, model=answer_model, api_key=answer_api_key
            )

        results.append(result)

        metrics.total_queries += 1
        metrics.total_latency_ms += latency
        metrics.total_embedding_tokens += len(q.question)

    return results, metrics