"""Tests for failure classifier."""

from ragcheck.analyzers.chunkers import Chunk
from ragcheck.analyzers.failure_classifier import (
    FailureClassifier,
    FailureMode,
)


class TestFailureClassifier:
    def test_retrieval_miss(self):
        classifier = FailureClassifier()
        result = classifier.classify(
            question="What is X?",
            expected_answer="X is a thing",
            generated_answer="",
            retrieved_chunks=[],
            source_chunks=["X is a thing"],
        )
        assert result.failure_mode == FailureMode.RETRIEVAL_MISS
        assert result.confidence == 0.9

    def test_hallucination(self):
        classifier = FailureClassifier()
        result = classifier.classify(
            question="What is X?",
            expected_answer="X is a thing",
            generated_answer="X is something completely different and made up",
            retrieved_chunks=[Chunk("X is a thing", 0, 12, "doc.txt", "fixed")],
            source_chunks=["X is a thing"],
        )
        assert result.failure_mode == FailureMode.HALLUCINATION

    def test_context_overload(self):
        classifier = FailureClassifier()
        chunks = [
            Chunk(f"X is a thing chunk{i}", i * 20, i * 20 + 20, "doc.txt", "fixed")
            for i in range(6)
        ]
        result = classifier.classify(
            question="What is X?",
            expected_answer="X is a thing",
            generated_answer="X is a thing",
            retrieved_chunks=chunks,
            source_chunks=["X is a thing"],
        )
        assert result.failure_mode == FailureMode.CONTEXT_OVERLOAD

    def test_unknown_when_no_issue(self):
        classifier = FailureClassifier()
        result = classifier.classify(
            question="What is X?",
            expected_answer="X is a thing",
            generated_answer="X is a thing",
            retrieved_chunks=[Chunk("X is a thing", 0, 12, "doc.txt", "fixed")],
            source_chunks=["X is a thing"],
        )
        assert result.failure_mode == FailureMode.UNKNOWN
