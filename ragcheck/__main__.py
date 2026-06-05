"""Allow running ragcheck as a module: python -m ragcheck."""

from ragcheck.cli import app

if __name__ == "__main__":
    app()