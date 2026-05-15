"""Pure unit tests for helpers module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.universal_llm_conversation.helpers import (
    _resolve_capabilities,
    async_fetch_models,
    get_exposed_entities,
    get_provider,
    sanitize_for_speech,
    shorten_tool_call_id,
)
from custom_components.universal_llm_conversation.providers import MODEL_CAPABILITY_OVERRIDES
from custom_components.universal_llm_conversation.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


class TestSanitizeForSpeech:
    """Test response sanitization."""

    def test_strips_thinking_blocks(self) -> None:
        text = "Let me think about this\n\nThe answer is 42"
        result = sanitize_for_speech(text)
        assert "Let me think" not in result
        assert "The answer is 42" in result

    def test_strips_bare_function_call(self) -> None:
        text = "execute_services(domain='light', service='turn_on') and the light is on"
        result = sanitize_for_speech(text, ["execute_services"])
        assert result is not None
        assert "execute_services" not in result
        assert "light is on" in result

    def test_strips_reasoning_content(self) -> None:
        text = 'Some reasoning here. "reasoning_content": "thinking..." More text.'
        result = sanitize_for_speech(text)
        assert "reasoning_content" not in result

    def test_returns_none_for_empty_result(self) -> None:
        text = "execute_services()"
        result = sanitize_for_speech(text, ["execute_services"])
        assert result is None

    def test_preserves_normal_text(self) -> None:
        text = "The light is now on."
        result = sanitize_for_speech(text)
        assert result == "The light is now on."


class TestShortenToolCallId:
    """Test tool call ID shortening."""

    def test_shortens_to_9_chars(self) -> None:
        long_id = "call_very_long_tool_call_id_here"
        result = shorten_tool_call_id(long_id)
        assert len(result) == 9
        assert result.isalnum()

    def test_deterministic(self) -> None:
        assert shorten_tool_call_id("abc123") == shorten_tool_call_id("abc123")


class TestResolveCapabilities:
    """Test capability resolution for different models."""

    def test_kimi_k2_6_overrides(self) -> None:
        caps = _resolve_capabilities("kimi-k2.6")
        assert caps.supports_temperature is False
        assert caps.supports_top_p is False
        assert caps.supports_max_completion_tokens is True
        assert caps.supports_thinking_content is True
        assert "temperature" in caps.unsupported_params
        assert "top_p" in caps.unsupported_params

    def test_kimi_k2_5_overrides(self) -> None:
        caps = _resolve_capabilities("kimi-k2.5")
        assert caps.supports_temperature is False
        assert caps.supports_max_completion_tokens is True

    def test_openai_model_default(self) -> None:
        caps = _resolve_capabilities("gpt-4o")
        assert caps.supports_temperature is True
        assert caps.supports_max_tokens is True

    def test_claude_via_openrouter(self) -> None:
        caps = _resolve_capabilities("claude-sonnet-4")
        assert caps.supports_thinking_content is True


class TestGetProvider:
    """Test provider factory."""

    @patch("custom_components.universal_llm_conversation.providers.openai_compatible.AsyncOpenAI")
    @patch("custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client")
    def test_returns_openai_compatible_provider(self, mock_get_client, mock_async_openai) -> None:
        from homeassistant.core import HomeAssistant

        hass = MagicMock(spec=HomeAssistant)
        provider = get_provider(
            hass=hass,
            provider_key="openai_compatible",
            api_key="test",
            base_url="http://localhost:1234/v1",
            api_version=None,
            organization=None,
            model="kimi-k2.6",
            timeout=60.0,
        )
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model == "kimi-k2.6"
        assert provider.capabilities.supports_temperature is False

    def test_unknown_provider_raises(self) -> None:
        from homeassistant.core import HomeAssistant

        hass = MagicMock(spec=HomeAssistant)
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider(
                hass=hass,
                provider_key="unknown_provider",
                api_key="test",
                base_url=None,
                api_version=None,
                organization=None,
                model="test",
                timeout=60.0,
            )


class TestGetExposedEntities:
    """Test exposed entity retrieval with aliases."""

    def test_returns_entities_with_aliases(self) -> None:
        from homeassistant.core import HomeAssistant
        from homeassistant.helpers import entity_registry as er

        hass = MagicMock()
        hass.states = MagicMock()
        state = MagicMock()
        state.entity_id = "light.test"
        state.name = "Test Light"
        state.state = "on"
        hass.states.async_all.return_value = [state]

        registry_entry = MagicMock()
        registry_entry.aliases = {"alias1", "alias2"}
        registry_mock = MagicMock()
        registry_mock.async_get.return_value = registry_entry

        with patch(
            "custom_components.universal_llm_conversation.helpers.async_should_expose",
            return_value=True,
        ), patch.object(er, "async_get", return_value=registry_mock):
            result = get_exposed_entities(hass)

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.test"
        assert set(result[0]["aliases"]) == {"alias1", "alias2"}

    def test_returns_empty_when_none_exposed(self) -> None:
        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.async_all.return_value = []

        with patch(
            "custom_components.universal_llm_conversation.helpers.async_should_expose",
            return_value=False,
        ):
            result = get_exposed_entities(hass)

        assert result == []


class TestAsyncFetchModels:
    """Test model fetching from OpenAI-compatible endpoint."""

    async def test_fetch_filters_non_chat_models(self) -> None:
        hass = MagicMock()

        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "text-embedding-3"
        mock_model3 = MagicMock()
        mock_model3.id = "tts-1"

        async def mock_list(*args, **kwargs):
            for m in [mock_model1, mock_model2, mock_model3]:
                yield m

        with patch(
            "custom_components.universal_llm_conversation.helpers.AsyncOpenAI"
        ) as mock_client:
            mock_client.return_value.models.list = MagicMock(return_value=mock_list())
            hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))

            result = await async_fetch_models(hass, "key", "http://localhost:1234/v1")

        assert result == ["gpt-4o"]

    async def test_fetch_returns_empty_on_exception(self) -> None:
        hass = MagicMock()
        hass.async_add_executor_job = AsyncMock(side_effect=ConnectionError("refused"))

        with patch(
            "custom_components.universal_llm_conversation.helpers.AsyncOpenAI"
        ):
            result = await async_fetch_models(hass, "key", "http://bad-url")

        assert result == []


class TestSanitizeEdgeCases:
    """Test sanitize_for_speech edge cases."""

    def test_returns_none_for_bare_tool_call(self) -> None:
        result = sanitize_for_speech("end_conversation()")
        assert result is None

    def test_strips_dynamic_function_names(self) -> None:
        result = sanitize_for_speech('execute_services(list=[{"domain":"light"}])', function_names=["execute_services"])
        assert result is None

    def test_returns_original_for_empty_string(self) -> None:
        result = sanitize_for_speech("")
        assert result == ""
