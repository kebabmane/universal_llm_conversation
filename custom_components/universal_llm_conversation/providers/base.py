"""Base provider abstraction for Universal LLM Conversation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from homeassistant.core import HomeAssistant

from . import ProviderCapabilities


class BaseProvider(ABC):
    """Abstract base for all LLM providers."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_key: str,
        base_url: str | None,
        api_version: str | None,
        organization: str | None,
        model: str,
        capabilities: ProviderCapabilities,
        timeout: float = 60.0,
    ) -> None:
        """Initialize provider."""
        self.hass = hass
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.organization = organization
        self.model = model
        self.capabilities = capabilities
        self.timeout = timeout

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield chunks from a streaming chat completion.

        Each chunk should be a dict with possible keys:
        - "role": "assistant" (first chunk only)
        - "content": str (text delta)
        - "tool_calls": list[dict] (when tool calls are complete)
        - "usage": dict (final chunk, optional)
        - "finish_reason": str (optional)
        """

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate provider connectivity. Return True if OK."""

    def filter_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Remove unsupported params and rename as needed."""
        result: dict[str, Any] = {}
        for key, value in params.items():
            if key in self.capabilities.unsupported_params:
                continue
            mapped = self.capabilities.param_names.get(key, key)
            result[mapped] = value
        return result

    @property
    def supports_tools(self) -> bool:
        return self.capabilities.supports_tools
