"""Helper functions for Universal LLM Conversation component."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.template import Template

from openai import AsyncOpenAI

from .providers import MODEL_CAPABILITY_OVERRIDES, OPENAI_COMPATIBLE_CAPABILITIES
from .providers.base import BaseProvider
from .providers.openai_compatible import OpenAICompatibleProvider

_LOGGER = logging.getLogger(__name__)


REASONING_CONTENT_PATTERNS = [
    # Match <think>...</think> blocks
    re.compile(r"<think>.*?</think>", re.DOTALL),
    # Match reasoning_content JSON field in text
    re.compile(r'"reasoning_content"\s*:\s*".*?"', re.DOTALL),
    # Match common reasoning prefixes
    re.compile(r"^(Thinking:|Reasoning:|Analysis:|Let me think.*?)[\n\r]+", re.IGNORECASE),
]

# Tool-call leak sanitizer patterns from issue #434
TOOL_CALL_LEAK_PATTERNS = [
    # Bare function call like end_conversation()
    re.compile(r"^\s*\w+\s*\([^)]*\)\s*$"),
    # Orphaned keyword args without function prefix
    re.compile(r'\b\w+\s*=\s*"[^"]*"\s*,?\s*\)?'),
    # Inline param sequences
    re.compile(r'\(\s*\w+\s*=\s*"[^"]*"\s*(,\s*\w+\s*=\s*"[^"]*"\s*)*\)'),
]


def get_exposed_entities(hass: HomeAssistant) -> list[dict[str, Any]]:
    """Get exposed entities for conversation domain."""
    states = [
        state
        for state in hass.states.async_all()
        if async_should_expose(hass, conversation.DOMAIN, state.entity_id)
    ]
    registry = er.async_get(hass)
    exposed: list[dict[str, Any]] = []
    for state in states:
        entity = registry.async_get(state.entity_id)
        aliases: list[str] = []
        if entity and entity.aliases:
            aliases = [str(a) for a in entity.aliases]
        exposed.append(
            {
                "entity_id": state.entity_id,
                "name": state.name,
                "state": state.state,
                "aliases": aliases,
            }
        )
    return exposed


def shorten_tool_call_id(tool_call_id: str) -> str:
    """Shorten tool call ID to 9 chars for Mistral compatibility."""
    return hashlib.sha256(tool_call_id.encode()).hexdigest()[:9]


def sanitize_for_speech(text: str, function_names: list[str] | None = None) -> str | None:
    """Strip leaked tool syntax and reasoning from response text.

    Returns None if the entire response was just a bare tool call.
    """
    if not text:
        return text

    cleaned = text

    # Strip reasoning blocks
    for pattern in REASONING_CONTENT_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Strip tool-call leaks
    for pattern in TOOL_CALL_LEAK_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Dynamic layer: strip known function names + arg blocks
    if function_names:
        for name in function_names:
            # function_name(key="val", ...)
            cleaned = re.sub(
                rf"\b{re.escape(name)}\s*\([^)]*\)",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    return cleaned


def _resolve_capabilities(model: str) -> Any:
    """Match model against override patterns."""
    for pattern, caps in MODEL_CAPABILITY_OVERRIDES.items():
        if pattern in model.lower() or model.lower().startswith(pattern.replace("-", "")):
            return caps
    return OPENAI_COMPATIBLE_CAPABILITIES


def get_provider(
    hass: HomeAssistant,
    provider_key: str,
    api_key: str,
    base_url: str | None,
    api_version: str | None,
    organization: str | None,
    model: str,
    timeout: float,
) -> BaseProvider:
    """Factory to return a provider instance."""
    capabilities = _resolve_capabilities(model)
    if provider_key == "openai_compatible":
        return OpenAICompatibleProvider(
            hass=hass,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            organization=organization,
            model=model,
            timeout=timeout,
            capabilities=capabilities,
        )
    # TODO: Add anthropic and gemini native providers
    raise ValueError(f"Unknown provider: {provider_key}")


async def async_fetch_models(
    hass: HomeAssistant,
    api_key: str,
    base_url: str,
    timeout: float = 10.0,
) -> list[str]:
    """Fetch available chat/completion models from an OpenAI-compatible /v1/models endpoint.

    Filters out non-chat models by excluding embeddings, audio, image, and TTS model names.
    """
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        response = client.models.list(timeout=timeout)
        models: list[str] = []
        excluded_keywords = [
            "embed", "embedding", "image", "audio", "tts", "whisper", "stt",
            "transcription", "speech", "vision-encode", "rerank", "classifier",
        ]
        async for model in response:
            model_id = getattr(model, "id", "")
            if not model_id:
                continue
            # Skip non-chat models
            if any(kw in model_id.lower() for kw in excluded_keywords):
                continue
            models.append(model_id)
        return sorted(models)
    except Exception as err:
        error_str = str(err).lower()
        if "403" in error_str or "forbidden" in error_str:
            _LOGGER.warning(
                "Provider restricts /v1/models for this API key tier. "
                "Enter model name manually."
            )
            raise HomeAssistantError("model_list_restricted") from err
        _LOGGER.error("Failed to fetch models from %s: %s", base_url, err)
        return []
