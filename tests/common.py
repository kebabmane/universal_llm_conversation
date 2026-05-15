"""Common test helpers for Universal LLM Conversation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_function_tool_yaml(filename: str) -> list[dict[str, Any]]:
    """Load function tool definitions from YAML fixture."""
    path = Path(__file__).parent / "fixtures" / "functions" / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_function_tool_from_yaml(filename: str, index: int = 0) -> dict[str, Any]:
    """Get raw function tool dict from YAML fixture."""
    tools = load_function_tool_yaml(filename)
    return tools[index]
