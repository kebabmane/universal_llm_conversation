"""Pure unit tests for provider modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.universal_llm_conversation.providers import (
    ANTHROPIC_CAPABILITIES,
    GEMINI_CAPABILITIES,
    MODEL_CAPABILITY_OVERRIDES,
    OPENAI_COMPATIBLE_CAPABILITIES,
    ProviderCapabilities,
)
from custom_components.universal_llm_conversation.providers.openai_compatible import (
    is_azure_url,
)


class TestProviderCapabilities:
    """Test capability filtering via BaseProvider."""

    def test_filter_params_removes_unsupported(self) -> None:
        from homeassistant.core import HomeAssistant
        from custom_components.universal_llm_conversation.providers.base import BaseProvider

        class DummyProvider(BaseProvider):
            async def stream_chat(self, messages, tools, options):
                yield {}
            async def validate_connection(self):
                return True

        caps = ProviderCapabilities(
            unsupported_params={"temperature", "top_p"},
        )
        provider = DummyProvider(
            hass=MagicMock(spec=HomeAssistant),
            api_key="test",
            base_url=None,
            api_version=None,
            organization=None,
            model="test",
            capabilities=caps,
            timeout=60.0,
        )
        params = {"temperature": 0.5, "top_p": 1.0, "max_tokens": 500}
        result = provider.filter_params(params)
        assert "temperature" not in result
        assert "top_p" not in result
        assert result["max_tokens"] == 500

    def test_filter_params_renames_mapped(self) -> None:
        from homeassistant.core import HomeAssistant
        from custom_components.universal_llm_conversation.providers.base import BaseProvider

        class DummyProvider(BaseProvider):
            async def stream_chat(self, messages, tools, options):
                yield {}
            async def validate_connection(self):
                return True

        caps = ProviderCapabilities(
            param_names={"max_tokens": "maxOutputTokens"},
        )
        provider = DummyProvider(
            hass=MagicMock(spec=HomeAssistant),
            api_key="test",
            base_url=None,
            api_version=None,
            organization=None,
            model="test",
            capabilities=caps,
            timeout=60.0,
        )
        params = {"max_tokens": 500, "temperature": 0.5}
        result = provider.filter_params(params)
        assert "max_tokens" not in result
        assert result["maxOutputTokens"] == 500
        assert result["temperature"] == 0.5

    def test_openai_compatible_defaults(self) -> None:
        assert OPENAI_COMPATIBLE_CAPABILITIES.supports_streaming is True
        assert OPENAI_COMPATIBLE_CAPABILITIES.supports_tools is True
        assert OPENAI_COMPATIBLE_CAPABILITIES.supports_strict_schemas is True

    def test_anthropic_capabilities(self) -> None:
        assert ANTHROPIC_CAPABILITIES.supports_thinking_content is True
        assert ANTHROPIC_CAPABILITIES.supports_strict_schemas is False
        assert "top_p" in ANTHROPIC_CAPABILITIES.unsupported_params

    def test_gemini_capabilities(self) -> None:
        assert GEMINI_CAPABILITIES.supports_tool_choice is False
        assert GEMINI_CAPABILITIES.param_names.get("max_tokens") == "maxOutputTokens"


class TestIsAzureUrl:
    """Test Azure URL detection."""

    def test_detects_azure_openai(self) -> None:
        assert is_azure_url("https://my-resource.openai.azure.com/") is True

    def test_detects_azure_api_net(self) -> None:
        assert is_azure_url("https://my-resource.azure-api.net/") is True

    def test_detects_ai_services(self) -> None:
        assert is_azure_url("https://my-resource.services.ai.azure.com/") is True

    def test_rejects_openai_com(self) -> None:
        assert is_azure_url("https://api.openai.com/v1") is False

    def test_rejects_none(self) -> None:
        assert is_azure_url(None) is False


class TestKimiOverrides:
    """Test Kimi model capability overrides."""

    def test_kimi_k2_6_has_correct_capabilities(self) -> None:
        caps = MODEL_CAPABILITY_OVERRIDES["kimi-k2.6"]
        assert caps.supports_max_completion_tokens is True
        assert caps.supports_max_tokens is False
        assert caps.supports_temperature is False
        assert caps.supports_top_p is False
        assert caps.supports_strict_schemas is False
        assert caps.supports_thinking_content is True
        assert caps.param_names["max_tokens"] == "max_completion_tokens"

    def test_claude_override_has_thinking(self) -> None:
        caps = MODEL_CAPABILITY_OVERRIDES["claude-"]
        assert caps.supports_thinking_content is True
        assert caps.supports_strict_schemas is False
