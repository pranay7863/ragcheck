"""Main report generation orchestrator."""

from typing import Any

from ragcheck.analyzers.failure_classifier import FailureClassifier, FailureMode
from ragcheck.analyzers.recommender import Recommendation, RecommendationEngine
from ragcheck.reports.html_report import generate_report


def _build_chunk_histogram(chunk_lengths: list[int]) -> list[dict[str, Any]]:
    """Bin chunk lengths into a CSS histogram.

    Adapts bin count to the data range so tiny datasets don't produce
    backwards or empty bins.
    """
    if not chunk_lengths:
        return []
    min_len = min(chunk_lengths)
    max_len = max(chunk_lengths)
    if max_len == min_len:
        return [{"range": f"{min_len}", "count": len(chunk_lengths), "pct": 100}]

    num_bins = min(10, max_len - min_len + 1)
    step = max(1, (max_len - min_len) // num_bins)

    bins: list[dict[str, Any]] = []
    current = min_len
    while current <= max_len:
        nxt = min(current + step, max_len)
        count = sum(1 for l in chunk_lengths if current <= l <= nxt)
        bins.append({
            "range": f"{current}–{nxt}",
            "count": count,
            "pct": int(count / len(chunk_lengths) * 100) if chunk_lengths else 0,
        })
        if nxt == max_len:
            break
        current = nxt + 1
    return bins


def _build_chunk_details(chunks: list[Any]) -> list[dict[str, Any]]:
    """Prepare chunk detail dicts for the HTML report."""
    details = []
    for i, c in enumerate(chunks):
        text = getattr(c, "text", str(c))
        start = getattr(c, "start", i)
        end = getattr(c, "end", i + 1)
        details.append({
            "start": start,
            "end": end,
            "length": len(text),
            "preview": text[:200] + ("..." if len(text) > 200 else ""),
        })
    return details


class ReportGenerator:
    """Orchestrate report generation from test results."""

    def __init__(self, nli_model: str | None = None) -> None:
        self.failure_classifier = FailureClassifier(nli_model=nli_model)
        self.recommendation_engine = RecommendationEngine()

    def generate(
        self,
        test_results: list[Any],
        chunking_results: dict[str, Any],
        project_name: str = "my-rag-project",
    ) -> str:
        """Generate complete HTML report from test results."""
        # Flatten nested result lists
        flat_results: list[Any] = []
        for item in test_results:
            if isinstance(item, list):
                flat_results.extend(item)
            else:
                flat_results.append(item)

        total = len(flat_results)
        passed = sum(1 for r in flat_results if getattr(r, "hit_at_k", False))
        overall_score = passed / total if total else None

        # Faithfulness: fraction of generated answers supported by retrieved chunks
        has_answers = any(
            getattr(r, "generated_answer", "") for r in flat_results
        )
        if has_answers:
            faithful_count = 0
            for result in flat_results:
                gen = getattr(result, "generated_answer", "")
                if gen:
                    chunks_texts = [c.text for c in result.retrieved_chunks]
                    if self.failure_classifier._answer_supported(gen, chunks_texts):
                        faithful_count += 1
            faithfulness_score = faithful_count / total if total else 0.0
        else:
            faithfulness_score = None  # type: ignore[assignment]

        # Classify ALL results so we always have insights to show
        failure_analyses = []
        failures = []
        for result in flat_results:
            analysis = self.failure_classifier.classify(
                question=result.question,
                expected_answer=result.expected_answer,
                generated_answer=getattr(result, "generated_answer", ""),
                retrieved_chunks=result.retrieved_chunks,
                source_chunks=getattr(result, "source_chunks", []),
            )

            is_failure = not getattr(result, "hit_at_k", False)
            is_meaningful = analysis.failure_mode != FailureMode.UNKNOWN

            if is_failure or is_meaningful:
                failure_analyses.append(analysis)
                failures.append({
                    "mode": analysis.failure_mode.value,
                    "confidence": int(analysis.confidence * 100),
                    "explanation": analysis.explanation,
                    "question": result.question,
                })

        # Placeholder when everything passes — report still needs content
        if not failures:
            failures.append({
                "mode": "none",
                "confidence": 100,
                "explanation": ("No critical failures detected. All test queries "
                                 "retrieved relevant chunks."),
                "question": "All queries",
            })

        recommendations = self.recommendation_engine.generate_recommendations(failure_analyses)

        # Fallback so the Recommendations section is never empty
        if not recommendations:
            recommendations = [
                Recommendation(
                    title="Monitor chunk overlap",
                    description=("Keep chunk overlap at 10–20% of chunk size "
                                 "to prevent boundary errors."),
                    expected_improvement=0.05,
                    tradeoffs="Minimal storage increase",
                    implementation_difficulty="easy",
                    code_example="chunk_overlap = 102  # 20% of 512",
                ),
                Recommendation(
                    title="Add a reranker",
                    description="Use BGE-Reranker to improve precision of top-k results.",
                    expected_improvement=0.08,
                    tradeoffs="+50ms latency per query",
                    implementation_difficulty="medium",
                    code_example=None,
                ),
            ]

        chunks = chunking_results.get("chunks", [])
        heatmap_data = self._generate_heatmap_data(flat_results, chunks)

        strategy = chunking_results.get("strategy", "unknown")
        chunk_lengths = [len(c.text) for c in chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        context_loss = sum(1 for c in chunks if c.text and c.text[-1] not in ".!?:")
        context_loss_score = context_loss / len(chunks) if chunks else 0

        top3_improvement = sum(r.expected_improvement for r in recommendations[:3])
        predicted_score = min(1.0, overall_score + top3_improvement) if overall_score is not None else None

        return generate_report(
            project_name=project_name,
            overall_score=overall_score,
            retrieval_score=overall_score,
            faithfulness_score=faithfulness_score,
            tests_passed=passed,
            tests_total=total,
            failures=failures[:10],
            recommendations=[
                {
                    "title": r.title,
                    "description": r.description,
                    "expected_improvement": int(r.expected_improvement * 100),
                    "tradeoffs": r.tradeoffs,
                    "difficulty": r.implementation_difficulty,
                    "code_example": r.code_example,
                }
                for r in recommendations[:5]
            ],
            current_score=overall_score,
            predicted_score=predicted_score,
            chunk_strategy=strategy,
            num_chunks=len(chunks),
            chunk_histogram=_build_chunk_histogram(chunk_lengths),
            chunk_details=_build_chunk_details(chunks),
            avg_chunk_length=avg_chunk_length,
            context_loss_score=context_loss_score,
            heatmap_data=heatmap_data,
        )

    def _generate_heatmap_data(
        self,
        test_results: list[Any],
        all_chunks: list[Any],
    ) -> list[dict[str, Any]]:
        """Generate Plotly heatmap data: rows = chunks, cols = queries."""
        if not all_chunks or not test_results:
            return [{
                "z": [[0]],
                "type": "heatmap",
                "colorscale": "RdYlGn",
                "showscale": True,
            }]

        num_chunks = len(all_chunks)
        num_queries = len(test_results)

        # Build retrieval frequency matrix
        z = [[0 for _ in range(num_queries)] for _ in range(num_chunks)]

        for q_idx, result in enumerate(test_results):
            for chunk in result.retrieved_chunks:
                for c_idx, c in enumerate(all_chunks):
                    if getattr(chunk, "text", "") == getattr(c, "text", ""):
                        z[c_idx][q_idx] = 1
                        break

        return [{
            "z": z,
            "x": [f"Q{i + 1}" for i in range(num_queries)],
            "y": [f"C{i + 1}" for i in range(num_chunks)],
            "type": "heatmap",
            "colorscale": "RdYlGn",
            "showscale": True,
            "hoverongaps": False,
        }]