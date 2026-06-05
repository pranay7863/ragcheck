"""Document loading utilities for ragcheck.

Supports: .txt, .md, .markdown, .pdf (via PyPDF2)
"""

from pathlib import Path
from typing import Protocol, Union


class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self, path: Path) -> str:
        """Load document content as string."""
        ...


class TextLoader:
    """Load plain text files."""

    def load(self, path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()


class MarkdownLoader:
    """Load markdown files (strips frontmatter)."""

    def load(self, path: Path) -> str:
        content = TextLoader().load(path)
        # Simple frontmatter removal
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return content


class PDFLoader:
    """Load PDF files using PyPDF2."""

    def load(self, path: Path) -> str:
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. Install: pip install PyPDF2"
            )

        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts)


def load_documents(docs_path: Union[str, Path]) -> dict[str, str]:
    """Load all documents from a directory.

    Args:
        docs_path: Path to documents directory (str or Path)

    Returns:
        Dictionary mapping filename to document content
    """
    docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents path not found: {docs_path}")

    loaders: dict[str, DocumentLoader] = {
        ".txt": TextLoader(),
        ".md": MarkdownLoader(),
        ".markdown": MarkdownLoader(),
        ".pdf": PDFLoader(),
    }

    documents: dict[str, str] = {}
    for file_path in docs_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in loaders:
            continue

        try:
            text = loaders[ext].load(file_path)
            if text.strip():
                documents[file_path.name] = text
            else:
                print(f"Warning: {file_path.name} loaded empty text (scanned image PDF?)")
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    return documents
