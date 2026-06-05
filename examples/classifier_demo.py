"""Demo: Failure classification + recommendations."""

from ragcheck.analyzers.chunkers import Chunk
from ragcheck.analyzers.failure_classifier import FailureClassifier
from ragcheck.analyzers.recommender import RecommendationEngine, predict_scores


def main():
    classifier = FailureClassifier()
    engine = RecommendationEngine()

    # Simulate 4 different failure scenarios
    scenarios = [
        {
            "name": "Retrieval Miss",
            "question": "What is quantum computing?",
            "expected": "Quantum computing uses qubits.",
            "generated": "",
            "retrieved": [],
            "source": ["Quantum computing uses qubits for computation."],
        },
        {
            "name": "Hallucination",
            "question": "What is RAG?",
            "expected": "RAG is Retrieval-Augmented Generation.",
            "generated": "RAG is a type of database invented in 2015 by Google.",
            "retrieved": [Chunk("RAG is Retrieval-Augmented Generation.", 0, 40, "doc.txt", "fixed")],
            "source": ["RAG is Retrieval-Augmented Generation."],
        },
        {
            "name": "Context Overload",
            "question": "How does RAG work?",
            "expected": "RAG retrieves documents then generates answers.",
            "generated": "RAG retrieves documents then generates answers.",
            "retrieved": [Chunk(f"chunk{i}", i*10, i*10+10, "doc.txt", "fixed") for i in range(6)],
            "source": ["RAG retrieves documents then generates answers."],
        },
        {
            "name": "Chunk Boundary Error",
            "question": "Explain the full RAG pipeline.",
            "expected": "RAG has retrieval and generation components working together.",
            "generated": "RAG has retrieval and generation components.",
            "retrieved": [
                Chunk("RAG has retrieval components", 0, 28, "doc.txt", "fixed"),
                Chunk("and generation components working", 29, 60, "doc.txt", "fixed"),
            ],
            "source": ["RAG has retrieval and generation components working together."],
        },
    ]

    print("Failure Classification Demo")
    print("=" * 60)

    all_failures = []
    for s in scenarios:
        analysis = classifier.classify(
            question=s["question"],
            expected_answer=s["expected"],
            generated_answer=s["generated"],
            retrieved_chunks=s["retrieved"],
            source_chunks=s["source"],
        )
        all_failures.append(analysis)

        print(f"\n{s['name']}:")
        print(f"  Mode: {analysis.failure_mode.value}")
        print(f"  Confidence: {analysis.confidence}")
        print(f"  Explanation: {analysis.explanation}")
        print(f"  Fix: {analysis.recommendation}")

    # Generate recommendations from all failures
    print("\n" + "=" * 60)
    print("Prioritized Recommendations")
    print("=" * 60)

    recommendations = engine.generate_recommendations(all_failures)
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. {rec.title} [{rec.implementation_difficulty}]")
        print(f"   {rec.description}")
        print(f"   Expected improvement: +{rec.expected_improvement:.1%}")
        print(f"   Tradeoffs: {rec.tradeoffs}")
        if rec.code_example:
            print(f"   Code: {rec.code_example}")

    # Score prediction
    print("\n" + "=" * 60)
    current = 0.55
    prediction = predict_scores(current, recommendations)
    print(f"Score Prediction: {prediction['current_score']:.0%} -> {prediction['predicted_score']:.0%}")
    print(f"  (+{prediction['improvement']:.1%} from top {prediction['recommendations_applied']} recommendations)")


if __name__ == "__main__":
    main()
