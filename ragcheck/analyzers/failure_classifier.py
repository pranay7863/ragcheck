"""Classify RAG failures into actionable categories."""

from dataclasses import dataclass
from enum import Enum

from ragcheck.analyzers.chunkers import Chunk


class FailureMode(Enum):
    """Types of RAG failures."""

    RETRIEVAL_MISS = "retrieval_miss"
    CONTEXT_OVERLOAD = "context_overload"
    HALLUCINATION = "hallucination"
    CHUNK_BOUNDARY_ERROR = "chunk_boundary_error"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis of a single failure."""

    failure_mode: FailureMode
    confidence: float
    explanation: str
    recommendation: str
    expected_improvement: float


class FailureClassifier:
    """Rule-based failure classification with optional NLI faithfulness."""

    def __init__(self, nli_model: str | None = None) -> None:
        """Initialise classifier.

        Args:
            nli_model: Hugging Face model id for zero-shot NLI (e.g.
                ``microsoft/deberta-xlarge-mnli``). If *None*, falls back
                to the cheap heuristic overlap check.
        """
        self.nli_model = nli_model
        self._nli_pipe = None
        if nli_model:
            try:
                from transformers import pipeline
                self._nli_pipe = pipeline(
                    "zero-shot-classification",
                    model=nli_model,
                    device=-1,  # CPU; override with CUDA if desired
                )
            except Exception:
                self._nli_pipe = None

    def classify(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        retrieved_chunks: list[Chunk],
        source_chunks: list[str],
    ) -> FailureAnalysis:
        """Classify why a RAG query failed."""
        retrieved_texts = [c.text for c in retrieved_chunks]

        correct_chunks_retrieved = any(
            any(src in rt or rt in src for rt in retrieved_texts)
            for src in source_chunks
        ) if source_chunks else len(retrieved_chunks) > 0

        if not correct_chunks_retrieved:
            return FailureAnalysis(
                failure_mode=FailureMode.RETRIEVAL_MISS,
                confidence=0.9,
                explanation="The correct chunk was not in the top-k retrieved results.",
                recommendation="Increase top-k, lower similarity threshold, or add a reranker.",
                expected_improvement=0.15,
            )

        if len(retrieved_chunks) > 5:
            return FailureAnalysis(
                failure_mode=FailureMode.CONTEXT_OVERLOAD,
                confidence=0.7,
                explanation="Too many chunks retrieved; LLM may have ignored relevant information.",
                recommendation="Reduce top-k to 3-5, use semantic compression, or add a reranker.",
                expected_improvement=0.08,
            )

        if generated_answer and not self._answer_supported(
            generated_answer, retrieved_texts
        ):
            return FailureAnalysis(
                failure_mode=FailureMode.HALLUCINATION,
                confidence=0.85,
                explanation=(
                    "The generated answer contains information "
                    "not present in retrieved chunks."
                ),
                recommendation="Improve prompt grounding or add NLI verification layer.",
                expected_improvement=0.12,
            )

        if self._is_boundary_error(expected_answer, retrieved_texts):
            return FailureAnalysis(
                failure_mode=FailureMode.CHUNK_BOUNDARY_ERROR,
                confidence=0.8,
                explanation="The answer spans multiple chunks but neither was retrieved fully.",
                recommendation="Increase chunk overlap to 20% or switch to semantic chunking.",
                expected_improvement=0.12,
            )

        return FailureAnalysis(
            failure_mode=FailureMode.UNKNOWN,
            confidence=0.5,
            explanation="Unable to determine exact failure mode.",
            recommendation="Review retrieval pipeline and embedding model selection.",
            expected_improvement=0.05,
        )

    def _answer_supported(self, answer: str, chunks: list[str]) -> bool:
        """Check if answer is supported by chunks (NLI or heuristic)."""
        if not answer or answer.strip().lower() in ("i do not know", "i don't know", "unknown"):
            return False  # Empty or "I don't know" = not supported

        # Prefer NLI when available
        if self._nli_pipe is not None:
            premise = " ".join(chunks)
            try:
                result = self._nli_pipe(
                    answer,
                    candidate_labels=["entailment", "contradiction", "neutral"],
                    hypothesis_template="This statement is {} by the context.",
                )
                # entailment must be the top label and score above 0.6
                if result["labels"][0] == "entailment" and result["scores"][0] > 0.6:
                    return True
                return False
            except Exception:
                pass  # fall through to heuristic

        # Improved heuristic: check for substantial word overlap OR key phrase containment
        answer_lower = answer.lower().strip()
        chunk_text = " ".join(chunks).lower()
        chunk_words = set(chunk_text.split())
        answer_words = set(answer_lower.split())

        if not answer_words:
            return False

        # Word overlap ratio
        overlap = len(answer_words & chunk_words)
        overlap_ratio = overlap / len(answer_words)

        # Also check if any significant phrases from answer appear in chunks
        # (handles paraphrased answers where individual words differ but key phrases match)
        answer_phrases = [answer_lower[i:i+20] for i in range(0, len(answer_lower)-19, 10)]
        phrase_matches = sum(1 for p in answer_phrases if p in chunk_text)
        phrase_ratio = phrase_matches / len(answer_phrases) if answer_phrases else 0

        # Supported if either: >15% word overlap OR >20% phrase containment
        # This is much more lenient than the old 50% threshold, accommodating
        # paraphrased answers from small local models like phi3:mini
        return overlap_ratio > 0.15 or phrase_ratio > 0.20

    def _is_boundary_error(self, expected: str, chunks: list[str]) -> bool:
        """Detect if answer is split across chunk boundaries."""
        chunk_positions = []
        for chunk in chunks:
            idx = expected.find(chunk[:50]) if chunk else -1
            if idx >= 0:
                chunk_positions.append(idx)

        if len(chunk_positions) >= 2:
            return max(chunk_positions) - min(chunk_positions) > 100
        return False
