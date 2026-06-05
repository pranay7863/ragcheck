"""Configuration loading utilities."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from ragcheck.core.config import RagcheckConfig


def load_config(config_path: str | Path = "ragcheck.yaml") -> RagcheckConfig:
    """Load configuration from YAML file or environment variables.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated RagcheckConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist and no env vars set
        ValidationError: If config is invalid
    """
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return RagcheckConfig(**data)

    return RagcheckConfig()


def save_config(config: RagcheckConfig, path: str | Path = "ragcheck.yaml") -> None:
    """Save configuration to YAML file.

    Converts Path objects to strings for safe YAML serialization.
    """
    path = Path(path)

    # Convert to dict, then recursively convert Path objects to strings
    raw = config.model_dump()

    def _convert_paths(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_paths(item) for item in obj]
        return obj

    safe_data = _convert_paths(raw)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(safe_data, f, default_flow_style=False, sort_keys=False)
