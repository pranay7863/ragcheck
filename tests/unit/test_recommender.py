"""Tests for recommendation engine."""

from ragcheck.analyzers.failure_classifier import FailureAnalysis, FailureMode
from ragcheck.analyzers.recommender import (
    Recommendation,
    RecommendationEngine,
    predict_scores,
)


class TestRecommendationEngine:
    def test_generates_recommendations(self):
        engine = RecommendationEngine()
        failures = [
            FailureAnalysis(FailureMode.RETRIEVAL_MISS, 0.9, "", "", 0.15),
            FailureAnalysis(FailureMode.RETRIEVAL_MISS, 0.8, "", "", 0.15),
            FailureAnalysis(FailureMode.HALLUCINATION, 0.7, "", "", 0.12),
        ]
        recs = engine.generate_recommendations(failures)

        assert len(recs) > 0
        titles = [r.title for r in recs]
        assert "Increase top-k" in titles
        assert "Improve prompt grounding" in titles

    def test_deduplicates_recommendations(self):
        engine = RecommendationEngine()
        failures = [
            FailureAnalysis(FailureMode.RETRIEVAL_MISS, 0.9, "", "", 0.15),
            FailureAnalysis(FailureMode.RETRIEVAL_MISS, 0.8, "", "", 0.15),
        ]
        recs = engine.generate_recommendations(failures)

        titles = [r.title for r in recs]
        assert len(titles) == len(set(titles))  # No duplicates

    def test_sorts_by_improvement(self):
        engine = RecommendationEngine()
        failures = [
            FailureAnalysis(FailureMode.HALLUCINATION, 0.9, "", "", 0.15),
            FailureAnalysis(FailureMode.RETRIEVAL_MISS, 0.5, "", "", 0.05),
        ]
        recs = engine.generate_recommendations(failures)

        # Higher improvement first
        for i in range(len(recs) - 1):
            assert recs[i].expected_improvement >= recs[i + 1].expected_improvement


class TestPredictScores:
    def test_predicts_improvement(self):
        recs = [
            Recommendation("R1", "D1", 0.10, "", "easy", None),
            Recommendation("R2", "D2", 0.08, "", "easy", None),
            Recommendation("R3", "D3", 0.05, "", "easy", None),
        ]
        result = predict_scores(0.60, recs)

        assert result["current_score"] == 0.60
        assert result["improvement"] == 0.23
        assert result["predicted_score"] == 0.83
        assert result["recommendations_applied"] == 3

    def test_caps_at_1_0(self):
        recs = [
            Recommendation("R1", "D1", 0.50, "", "easy", None),
            Recommendation("R2", "D2", 0.40, "", "easy", None),
        ]
        result = predict_scores(0.80, recs)
        assert result["predicted_score"] == 1.0
