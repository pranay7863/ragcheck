"""Generate actionable recommendations based on failure patterns."""

from dataclasses import dataclass

from ragcheck.analyzers.failure_classifier import FailureAnalysis, FailureMode


@dataclass
class Recommendation:
    """A single recommendation."""

    title: str
    description: str
    expected_improvement: float
    tradeoffs: str
    implementation_difficulty: str  # easy, medium, hard
    code_example: str | None


class RecommendationEngine:
    """Generate recommendations based on failure pattern distribution."""

    def __init__(self) -> None:
        self.recommendation_map: dict[FailureMode, list[Recommendation]] = {
            FailureMode.RETRIEVAL_MISS: [
                Recommendation(
                    title="Increase top-k",
                    description="Retrieve more chunks to improve recall.",
                    expected_improvement=0.05,
                    tradeoffs="+15% latency, +10% cost",
                    implementation_difficulty="easy",
                    code_example=(
                        "from ragcheck.rerankers import BGEReranker\n"
                        "reranker = BGEReranker()"
                    ),
                ),
                Recommendation(
                    title="Add a reranker",
                    description="Use BGE-Reranker to re-rank retrieved chunks.",
                    expected_improvement=0.08,
                    tradeoffs="+50ms latency per query",
                    implementation_difficulty="medium",
                    code_example=None,
                ),
                Recommendation(
                    title="Use hybrid search",
                    description="Combine BM25 keyword search with dense retrieval.",
                    expected_improvement=0.12,
                    tradeoffs="Requires indexing pipeline, +20% storage",
                    implementation_difficulty="hard",
                    code_example=None,
                ),
            ],
            FailureMode.CONTEXT_OVERLOAD: [
                Recommendation(
                    title="Reduce top-k",
                    description="Send fewer chunks to the LLM to reduce noise.",
                    expected_improvement=0.08,
                    tradeoffs="-5% recall, -10% cost",
                    implementation_difficulty="easy",
                    code_example="top_k = 3  # was 10",
                ),
                Recommendation(
                    title="Add semantic compression",
                    description="Compress chunks to key sentences before sending to LLM.",
                    expected_improvement=0.10,
                    tradeoffs="+30ms latency",
                    implementation_difficulty="medium",
                    code_example=None,
                ),
            ],
            FailureMode.HALLUCINATION: [
                Recommendation(
                    title="Improve prompt grounding",
                    description="Add explicit instruction to only use retrieved context.",
                    expected_improvement=0.12,
                    tradeoffs="None",
                    implementation_difficulty="easy",
                    code_example=(
                        'prompt = "Answer ONLY using the provided context. '
                        'If unsure, say I do not know."'
                    ),
                ),
                Recommendation(
                    title="Add NLI verification",
                    description="Use a local NLI model to verify answer faithfulness.",
                    expected_improvement=0.15,
                    tradeoffs="+100ms latency",
                    implementation_difficulty="medium",
                    code_example=None,
                ),
            ],
            FailureMode.CHUNK_BOUNDARY_ERROR: [
                Recommendation(
                    title="Increase chunk overlap",
                    description="Increase overlap to 20% of chunk size.",
                    expected_improvement=0.12,
                    tradeoffs="+15% storage, +5% embedding cost",
                    implementation_difficulty="easy",
                    code_example="chunk_overlap = 102  # 20% of 512",
                ),
                Recommendation(
                    title="Switch to semantic chunking",
                    description="Use sentence boundaries instead of fixed sizes.",
                    expected_improvement=0.15,
                    tradeoffs="Variable chunk sizes, may need tuning",
                    implementation_difficulty="medium",
                    code_example='strategy = "semantic"',
                ),
            ],
        }

    def generate_recommendations(
        self,
        failures: list[FailureAnalysis],
    ) -> list[Recommendation]:
        """Generate prioritized recommendations from failure patterns.

        Args:
            failures: List of failure analyses

        Returns:
            Prioritized list of unique recommendations
        """
        from collections import Counter

        mode_counts = Counter(f.failure_mode for f in failures)
        total = len(failures)

        recommendations: list[Recommendation] = []
        seen_titles: set[str] = set()

        for mode, count in mode_counts.most_common():
            percentage = count / total
            if percentage < 0.1:
                continue

            for rec in self.recommendation_map.get(mode, []):
                if rec.title not in seen_titles:
                    adjusted_rec = Recommendation(
                        title=rec.title,
                        description=rec.description,
                        expected_improvement=rec.expected_improvement * percentage,
                        tradeoffs=rec.tradeoffs,
                        implementation_difficulty=rec.implementation_difficulty,
                        code_example=rec.code_example,
                    )
                    recommendations.append(adjusted_rec)
                    seen_titles.add(rec.title)

        recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)
        return recommendations


def predict_scores(
    current_score: float,
    recommendations: list[Recommendation],
) -> dict[str, float]:
    """Predict scores after applying recommendations.

    Args:
        current_score: Current RAG score (0-1)
        recommendations: List of recommendations

    Returns:
        Dictionary with predicted scores
    """
    cumulative_improvement = sum(r.expected_improvement for r in recommendations[:3])
    predicted_score = min(1.0, current_score + cumulative_improvement)

    return {
        "current_score": current_score,
        "predicted_score": predicted_score,
        "improvement": cumulative_improvement,
        "recommendations_applied": len(recommendations[:3]),
    }
