# Contributing to ragcheck

## Development Setup

```bash
git clone https://github.com/pranay7863/ragcheck.git
cd ragcheck
uv sync
uv run pytest
```

## Code Style

- `ruff` for linting and formatting
- `mypy` for type checking
- All code must pass `ruff check .` and `mypy ragcheck/`

## Testing

```bash
uv run pytest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Open a Pull Request
