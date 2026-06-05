"""Demo: Auto-QA generation + retrieval testing.

NOTE: This demo requires an OpenAI API key set in your environment:
    $env:OPENAI_API_KEY = "sk-..."

If no API key is available, the demo will show the pipeline structure
but skip the LLM call.
"""

import os

from ragcheck.analyzers.chunkers import RecursiveChunker
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore
from ragcheck.testers.auto_qa import generate_questions
from ragcheck.testers.retrieval_tester import DenseRetriever, run_retrieval_tests


def main():
    sample_text = """RAG stands for Retrieval-Augmented Generation.
It combines retrieval systems with generative AI models.
The retrieval component fetches relevant documents from a knowledge base.
The generation component produces answers based on retrieved context.
Vector databases store document embeddings for efficient similarity search."""

    # 1. Chunk
    chunker = RecursiveChunker(chunk_size=100)
    chunks = chunker.chunk(sample_text, "sample.txt")
    print(f"Created {len(chunks)} chunks")

    # 2. Embed & Store
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.embed([c.text for c in chunks])
    store = ChromaVectorStore(collection_name="qa_demo")
    store.reset()
    store.add_chunks(chunks, embeddings)
    print("Stored chunks in ChromaDB")

    # 3. Auto-QA Generation (requires API key)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[WARNING] No OPENAI_API_KEY found. Skipping LLM question generation.")
        print("Set your API key with: $env:OPENAI_API_KEY = 'sk-...'")
        # Use manual questions for demo
        from ragcheck.testers.auto_qa import TestQuestion
        questions = [
            TestQuestion("What is RAG?", "Retrieval-Augmented Generation", [chunks[0].text], "easy"),
            TestQuestion("How does retrieval work?", "Fetches documents from knowledge base", [chunks[2].text], "medium"),
        ]
    else:
        print("\nGenerating questions via LLM...")
        questions = generate_questions(chunks, num_questions=3, api_key=api_key)
        print(f"Generated {len(questions)} questions")

    # 4. Retrieval Test
    retriever = DenseRetriever(embedder, store)
    results, metrics = run_retrieval_tests(questions, retriever, top_k=2)

    print(f"\nRetrieval Test Results ({metrics.total_queries} queries)")
    print("=" * 50)
    for i, r in enumerate(results):
        print(f"\nQ{i+1}: {r.question}")
        print(f"  Expected: {r.expected_answer[:60]}...")
        print(f"  Retrieved {len(r.retrieved_chunks)} chunks | Latency: {r.latency_ms:.1f}ms")
        for chunk, score in zip(r.retrieved_chunks, r.scores):
            print(f"    [{score:.4f}] {chunk.text[:50]}...")

    print(f"\nMetrics: Avg latency = {metrics.avg_latency_ms:.1f}ms | "
          f"Est. cost = ${metrics.estimated_cost_usd:.6f}")


if __name__ == "__main__":
    main()
