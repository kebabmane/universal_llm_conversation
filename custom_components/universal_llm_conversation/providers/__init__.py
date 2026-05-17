"""Provider capability definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderCapabilities:
    """Describe what a provider supports."""

    supports_streaming: bool = True
    supports_tools: bool = True
    supports_tool_choice: bool = True
    supports_temperature: bool = True
    supports_top_p: bool = True
    supports_max_tokens: bool = True
    supports_max_completion_tokens: bool = False
    supports_reasoning_effort: bool = False
    supports_service_tier: bool = False
    supports_strict_schemas: bool = False
    supports_thinking_content: bool = False
    supports_vision: bool = False
    # Parameter name mapping: {our_name: provider_name}
    param_names: dict[str, str] = field(default_factory=dict)
    # Params to omit entirely
    unsupported_params: set[str] = field(default_factory=set)


# Built-in capability presets
OPENAI_COMPATIBLE_CAPABILITIES = ProviderCapabilities(
    supports_streaming=True,
    supports_tools=True,
    supports_tool_choice=True,
    supports_temperature=True,
    supports_top_p=True,
    supports_max_tokens=True,
    supports_max_completion_tokens=False,
    supports_reasoning_effort=False,
    supports_service_tier=False,
    supports_strict_schemas=True,
    supports_thinking_content=False,
    supports_vision=False,
    param_names={},
    unsupported_params=set(),
)

ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    supports_streaming=True,
    supports_tools=True,
    supports_tool_choice=True,
    supports_temperature=True,
    supports_top_p=True,
    supports_max_tokens=True,
    supports_max_completion_tokens=False,
    supports_reasoning_effort=True,
    supports_service_tier=False,
    supports_strict_schemas=False,
    supports_thinking_content=True,
    supports_vision=True,
    param_names={"max_tokens": "max_tokens"},
    unsupported_params={"top_p"},  # Anthropic uses top_p but some deployments choke; we make it opt-in
)

GEMINI_CAPABILITIES = ProviderCapabilities(
    supports_streaming=True,
    supports_tools=True,
    supports_tool_choice=False,
    supports_temperature=True,
    supports_top_p=True,
    supports_max_tokens=True,
    supports_max_completion_tokens=False,
    supports_reasoning_effort=False,
    supports_service_tier=False,
    supports_strict_schemas=False,
    supports_thinking_content=False,
    supports_vision=True,
    param_names={"max_tokens": "maxOutputTokens"},
    unsupported_params={"top_p"},  # Gemini topP is separate, but many use top_p
)

# Per-model overrides for OpenAI-compatible providers
MODEL_CAPABILITY_OVERRIDES: dict[str, ProviderCapabilities] = {
    # Kimi models
    "kimi-k2.6": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=False,
        supports_thinking_content=True,
        supports_vision=True,
        param_names={"max_tokens": "max_completion_tokens"},
        unsupported_params={"temperature", "top_p"},
    ),
    "kimi-k2.5": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=False,
        supports_thinking_content=True,
        supports_vision=True,
        param_names={"max_tokens": "max_completion_tokens"},
        unsupported_params={"temperature", "top_p"},
    ),
    "kimi-k2p6": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=False,
        supports_top_p=False,
        supports_max_tokens=False,
        supports_max_completion_tokens=True,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=False,
        supports_thinking_content=True,
        supports_vision=True,
        param_names={"max_tokens": "max_completion_tokens"},
        unsupported_params={"temperature", "top_p"},
    ),
    # Anthropic models via OpenRouter / OpenAI-compatible
    "claude-": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_max_completion_tokens=False,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=False,
        supports_thinking_content=True,
        supports_vision=True,
        param_names={},
        unsupported_params=set(),
    ),
    # GPT-4o models
    "gpt-4o": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_max_completion_tokens=False,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=True,
        supports_thinking_content=False,
        supports_vision=True,
        param_names={},
        unsupported_params=set(),
    ),
    "gpt-4o-mini": ProviderCapabilities(
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_temperature=True,
        supports_top_p=True,
        supports_max_tokens=True,
        supports_max_completion_tokens=False,
        supports_reasoning_effort=False,
        supports_service_tier=False,
        supports_strict_schemas=True,
        supports_thinking_content=False,
        supports_vision=True,
        param_names={},
        unsupported_params=set(),
    ),
}
