"""ragcheck CLI entry point."""
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ragcheck import __version__
from ragcheck.core.config_loader import load_config
from ragcheck.core.document_loader import load_documents
from ragcheck.core.progress import get_progress, print_error, print_success

app = typer.Typer(
    name="ragcheck",
    help="Lighthouse for RAG systems. Diagnose and fix your RAG pipeline.",
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    """ragcheck CLI entry point."""
    if version:
        typer.echo(f"ragcheck {__version__}")
        raise typer.Exit()


@app.command()
def init(
    path: str = typer.Argument(".", help="Project path to initialize"),
) -> None:
    """Initialize a new ragcheck project."""
    console.print(Panel(f"[green]Initializing ragcheck project at {path}[/green]"))
    cfg = load_config()
    cfg.docs_path = Path(path) / "data"
    from ragcheck.core.config_loader import save_config
    save_config(cfg, Path(path) / "ragcheck.yaml")
    print_success("Created ragcheck.yaml")


@app.command()
def run(
    docs: str = typer.Option(..., "--docs", "-d", help="Path to documents directory"),
    query: str = typer.Option(None, "--query", "-q", help="Single query to test"),
    config: str = typer.Option("ragcheck.yaml", "--config", "-c", help="Config file path"),
    ci: bool = typer.Option(False, "--ci", help="CI mode: exit code based on threshold"),
    min_score: float = typer.Option(0.80, "--min-score", help="Minimum score for CI pass"),
    output: str = typer.Option("ragcheck_report.html", "--output", "-o", help="Output file path"),
    generate_answers: bool = typer.Option(
        False,
        "--generate-answers",
        help="Generate answers via LLM for faithfulness scoring",
    ),
    answer_model: str = typer.Option(
        "ollama/phi3:mini",
        "--answer-model",
        help="LiteLLM model for answer generation and question generation (e.g. ollama/phi3:mini)",
    ),
    nli_model: str | None = typer.Option(
        None,
        "--nli-model",
        help="NLI model for faithfulness (e.g. cross-encoder/nli-deberta-v3-xsmall)",
    ),
) -> None:
    """Run ragcheck analysis on your documents."""
    from ragcheck.analyzers.chunkers import ChunkerFactory
    from ragcheck.core.embeddings import SentenceTransformerEmbedder
    from ragcheck.core.vector_store import ChromaVectorStore
    from ragcheck.reports.generator import ReportGenerator
    from ragcheck.testers.auto_qa import TestQuestion, generate_questions
    from ragcheck.testers.retrieval_tester import DenseRetriever, run_retrieval_tests

    cfg = load_config(config)
    docs_path = Path(docs)

    if not docs_path.exists():
        print_error(f"Documents path not found: {docs}")
        raise typer.Exit(code=1)

    # Resolve API key: config -> env var -> None
    qa_model = answer_model
    qa_api_key = cfg.llm.api_key if cfg.llm.api_key else None

    # Fallback to environment variables for known providers
    if not qa_api_key:
        provider = qa_model.split("/", 1)[0] if "/" in qa_model else "openai"
        if provider == "groq":
            qa_api_key = os.environ.get("GROQ_API_KEY")
        elif provider == "openrouter":
            qa_api_key = os.environ.get("OPENROUTER_API_KEY")
        elif provider == "anthropic":
            qa_api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "gemini":
            qa_api_key = os.environ.get("GEMINI_API_KEY")
        elif provider == "openai":
            qa_api_key = os.environ.get("OPENAI_API_KEY")

    with get_progress() as progress:
        task = progress.add_task("[cyan]Loading documents...", total=100)
        documents = load_documents(docs_path)
        progress.update(task, advance=20)

        if not documents:
            print_error("No documents found.")
            if ci:
                raise typer.Exit(code=1)
            return

        progress.update(task, description="[cyan]Chunking documents...", advance=20)
        chunker = ChunkerFactory.create(cfg.chunking.strategy)
        all_chunks = []
        for src, text in documents.items():
            all_chunks.extend(chunker.chunk(text, src))

        if not all_chunks:
            print_error("No chunks created. Check your documents.")
            if ci:
                raise typer.Exit(code=1)
            return

        progress.update(task, description="[cyan]Embedding chunks...", advance=20)
        embedder = SentenceTransformerEmbedder(cfg.chunking.model)
        embeddings = embedder.embed([c.text for c in all_chunks])

        progress.update(task, description="[cyan]Storing in vector DB...", advance=20)
        store = ChromaVectorStore()
        store.add_chunks(all_chunks, embeddings)

        progress.update(task, description="[cyan]Generating tests...", advance=10)
        questions = generate_questions(
            all_chunks,
            num_questions=min(cfg.max_test_questions, len(all_chunks) * 2),
            model=qa_model,
            api_key=qa_api_key,
        )
        if not questions:
            questions = []
            for c in all_chunks[:3]:
                questions.append(TestQuestion(
                    question=f"What is discussed in this chunk? {c.text[:50]}...",
                    expected_answer=c.text,
                    source_chunks=[c.text],
                    difficulty="easy",
                ))

        progress.update(task, description="[cyan]Testing retrieval...", advance=10)
        retriever = DenseRetriever(embedder, store)
        retrieval_results, _metrics = run_retrieval_tests(
            questions,
            retriever,
            top_k=cfg.retrieval.top_k,
            generate_answers=generate_answers,
            answer_model=answer_model,
            answer_api_key=qa_api_key,
        )

    generator = ReportGenerator(nli_model=nli_model)
    report_html = generator.generate(
        test_results=retrieval_results,
        chunking_results={"strategy": cfg.chunking.strategy, "chunks": all_chunks},
        project_name=cfg.project_name,
    )

    with open(output, "w", encoding="utf-8") as f:
        f.write(report_html)

    total_tests = len(retrieval_results)
    tests_passed = 0
    for rr in retrieval_results:
        if rr.hit_at_k:
            tests_passed += 1
    score = tests_passed / total_tests if total_tests else 0.0

    if ci:
        if score < min_score:
            print_error(f"CI FAILED: Score {score:.2%} < threshold {min_score:.2%}")
            raise typer.Exit(code=1)
        else:
            print_success(f"CI PASSED: Score {score:.2%} >= threshold {min_score:.2%}")

    print_success(f"Report saved to {output}")
    console.print(f"[dim]Tests: {tests_passed}/{total_tests} passed | Score: {score:.2%}[/dim]")
    if generate_answers:
        if answer_model.startswith("ollama/"):
            console.print(f"[dim]Local model used: {answer_model} | Zero API cost[/dim]")
        else:
            key_status = "OK" if qa_api_key else "MISSING"
            console.print(f"[dim]Cloud model used: {answer_model} | API key: {key_status}[/dim]")


@app.command()
def report(
    output: str = typer.Option("ragcheck_report.html", "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", "-f", help="Report format: html, json"),
) -> None:
    """Generate a report from previous analysis."""
    console.print(Panel(f"[green]Generating {format} report: {output}[/green]"))


if __name__ == "__main__":
    app()
