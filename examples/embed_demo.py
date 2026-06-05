"""Demo: embed chunks and search with ChromaDB."""

from ragcheck.analyzers.chunkers import RecursiveChunker
from ragcheck.core.embeddings import SentenceTransformerEmbedder
from ragcheck.core.vector_store import ChromaVectorStore


def main():
    sample_text = """RAG stands for Retrieval-Augmented Generation.
It combines retrieval systems with generative AI models.
The retrieval component fetches relevant documents from a knowledge base.
The generation component produces answers based on retrieved context.
Vector databases store document embeddings for efficient similarity search.
Embedding models convert text into dense vector representations."""

    # 1. Chunk
    chunker = RecursiveChunker(chunk_size=80)
    chunks = chunker.chunk(sample_text, "sample.txt")
    print(f"Created {len(chunks)} chunks")

    # 2. Embed
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"Embedded {len(chunks)} chunks into {embedder.dimension}D vectors")

    # 3. Store
    store = ChromaVectorStore(collection_name="demo")
    store.reset()
    store.add_chunks(chunks, embeddings)
    print("Stored chunks in ChromaDB")

    # 4. Search
    queries = [
        "What is RAG?",
        "How does retrieval work?",
        "What are embeddings?",
    ]

    print("\nSearch Results:")
    print("=" * 50)
    for query in queries:
        query_emb = embedder.embed([query])[0]
        results = store.search(query_emb, top_k=2)
        print(f"\nQuery: '{query}'")
        for chunk, score in results:
            print(f"  [{score:.4f}] {chunk.text[:60]}...")


if __name__ == "__main__":
    main()
