"""Demo: Full ragcheck pipeline end-to-end."""

from pathlib import Path


def main():
    data_dir = Path("./sample_data")
    data_dir.mkdir(exist_ok=True)

    (data_dir / "intro.txt").write_text(
        "RAG (Retrieval-Augmented Generation) is a technique that combines "
        "retrieval systems with generative AI models. It works by fetching "
        "relevant documents from a knowledge base and using them to generate "
        "accurate answers. This reduces hallucinations and improves factual accuracy.",
        encoding="utf-8",
    )

    (data_dir / "components.txt").write_text(
        "The key components of RAG are: a document store, an embedding model, "
        "a vector database, and a large language model. The embedding model "
        "converts text into dense vectors. The vector database stores these "
        "vectors for efficient similarity search. The LLM generates the final answer.",
        encoding="utf-8",
    )

    print("Sample data created in ./sample_data")
    print("\nRun the full pipeline with:")
    print("  uv run ragcheck run --docs ./sample_data")
    print("\nOr with CI mode:")
    print("  uv run ragcheck run --docs ./sample_data --ci --min-score 0.80")
    print("\nWith a specific query:")
    print('  uv run ragcheck run --docs ./sample_data --query "What is RAG?"')


if __name__ == "__main__":
    main()
