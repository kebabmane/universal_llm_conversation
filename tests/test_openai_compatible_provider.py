"""Unit tests for OpenAICompatibleProvider streaming and lifecycle."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.providers import (
    OPENAI_COMPATIBLE_CAPABILITIES,
)
from custom_components.universal_llm_conversation.providers.openai_compatible import (
    AZURE_DOMAIN_PATTERN,
    OpenAICompatibleProvider,
    is_azure_url,
)


class TestIsAzureUrl:
    """Extended Azure URL detection edge cases."""

    def test_detects_subdomain_variations(self) -> None:
        assert is_azure_url("https://foo.bar.openai.azure.com/") is True

    def test_detects_azure_api_net_subdomain(self) -> None:
        assert is_azure_url("https://my-resource.azure-api.net/openai") is True

    def test_rejects_empty_string(self) -> None:
        assert is_azure_url("") is False

    def test_rejects_ip_address(self) -> None:
        assert is_azure_url("http://127.0.0.1:8000/v1") is False

    def test_azure_domain_pattern_raw(self) -> None:
        import re
        assert re.search(AZURE_DOMAIN_PATTERN, "https://x.openai.azure.com")
        assert re.search(AZURE_DOMAIN_PATTERN, "https://x.azure-api.net")
        assert not re.search(AZURE_DOMAIN_PATTERN, "https://api.openai.com")


def _mocked_get_async_client(hass):
    return httpx.AsyncClient()


class TestProviderInit:
    """Test client selection during construction."""

    def test_uses_azure_client_for_azure_url(self) -> None:
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncAzureOpenAI"
        ) as mock_azure, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncOpenAI"
        ) as mock_openai, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            hass = MagicMock(spec=HomeAssistant)
            hass.data = {}
            OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url="https://my-resource.openai.azure.com/",
                api_version="2024-08-01",
                organization="org",
                model="gpt-4o",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=30.0,
            )
            mock_azure.assert_called_once()
            mock_openai.assert_not_called()
            call_kwargs = mock_azure.call_args.kwargs
            assert call_kwargs["azure_endpoint"] == "https://my-resource.openai.azure.com/"
            assert call_kwargs["api_version"] == "2024-08-01"

    def test_uses_openai_client_for_non_azure_url(self) -> None:
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncAzureOpenAI"
        ) as mock_azure, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncOpenAI"
        ) as mock_openai, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            hass = MagicMock(spec=HomeAssistant)
            hass.data = {}
            OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url="https://api.fireworks.ai/v1",
                api_version=None,
                organization=None,
                model="kimi-k2.6",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=60.0,
            )
            mock_azure.assert_not_called()
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == "https://api.fireworks.ai/v1"

    def test_uses_openai_client_when_no_base_url(self) -> None:
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncAzureOpenAI"
        ) as mock_azure, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.AsyncOpenAI"
        ) as mock_openai, patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            hass = MagicMock(spec=HomeAssistant)
            hass.data = {}
            OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url=None,
                api_version=None,
                organization=None,
                model="gpt-4o",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=60.0,
            )
            mock_azure.assert_not_called()
            mock_openai.assert_called_once()


class TestValidateConnection:
    """Test validate_connection with mocked models.list."""

    async def test_validate_connection_success(self) -> None:
        hass = MagicMock(spec=HomeAssistant)
        hass.data = {}
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            provider = OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url=None,
                api_version=None,
                organization=None,
                model="gpt-4o",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=60.0,
            )
        async def async_list(*args, **kwargs):
            yield MagicMock()

        provider._client.models.list = MagicMock(return_value=async_list())
        provider.hass.async_add_executor_job = AsyncMock(side_effect=lambda target, *args: target(*args))

        result = await provider.validate_connection()
        assert result is True

    async def test_validate_connection_failure(self) -> None:
        hass = MagicMock(spec=HomeAssistant)
        hass.data = {}
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            provider = OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url=None,
                api_version=None,
                organization=None,
                model="gpt-4o",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=60.0,
            )
        provider._client.models.list = MagicMock(side_effect=ConnectionError(" refused"))

        result = await provider.validate_connection()
        assert result is False


class TestStreamChat:
    """Direct tests for stream_chat yielding correct chunks."""

    @pytest.fixture
    def provider(self) -> OpenAICompatibleProvider:
        hass = MagicMock(spec=HomeAssistant)
        hass.data = {}
        with patch(
            "custom_components.universal_llm_conversation.providers.openai_compatible.get_async_client",
            side_effect=_mocked_get_async_client,
        ):
            return OpenAICompatibleProvider(
                hass=hass,
                api_key="key",
                base_url=None,
                api_version=None,
                organization=None,
                model="gpt-4o",
                capabilities=OPENAI_COMPATIBLE_CAPABILITIES,
                timeout=60.0,
            )

    async def _make_chunk(
        self,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[dict] | None = None,
        finish_reason: str | None = None,
        usage: dict | None = None,
        empty_choices: bool = False,
    ) -> MagicMock:
        """Build a mock ChatCompletionChunk."""
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning_content

        if tool_calls:
            delta.tool_calls = []
            for tc in tool_calls:
                tcd = MagicMock()
                tcd.index = tc["index"]
                tcd.id = tc.get("id")
                tcd.function = MagicMock()
                tcd.function.name = tc.get("name")
                tcd.function.arguments = tc.get("arguments", "")
                delta.tool_calls.append(tcd)
        else:
            delta.tool_calls = None

        if empty_choices:
            chunk.choices = []
        else:
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = finish_reason
            chunk.choices = [choice]

        if usage:
            chunk.usage = MagicMock()
            chunk.usage.prompt_tokens = usage.get("prompt_tokens", 0)
            chunk.usage.completion_tokens = usage.get("completion_tokens", 0)
            chunk.usage.total_tokens = usage.get("total_tokens", 0)
        else:
            chunk.usage = None

        return chunk

    async def test_first_chunk_yields_role(self, provider: OpenAICompatibleProvider) -> None:
        chunk = await self._make_chunk(content="Hello")
        provider._client.chat.completions.create = AsyncMock(return_value=self._async_iter([chunk]))

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert result[0] == {"role": "assistant"}

    async def test_content_chunks(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(content="Hello "),
            await self._make_chunk(content="world"),
            await self._make_chunk(finish_reason="stop"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert {"content": "Hello "} in result
        assert {"content": "world"} in result

    async def test_reasoning_content(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(content="", reasoning_content="Thinking..."),
            await self._make_chunk(finish_reason="stop"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert {"reasoning_content": "Thinking..."} in result

    async def test_usage_chunk(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(),
            await self._make_chunk(
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                empty_choices=True,
            ),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        } in result

    async def test_tool_calls_across_chunks(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(
                tool_calls=[{"index": 0, "id": "call_1", "name": "execute_services", "arguments": '{"list": [{"dom'}]
            ),
            await self._make_chunk(
                tool_calls=[{"index": 0, "arguments": "ain\": \"light\", \"service\": \"turn_on\"}]}"}]
            ),
            await self._make_chunk(finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        tool_call_item = next((r for r in result if "tool_calls" in r), None)
        assert tool_call_item is not None
        assert tool_call_item["tool_calls"][0]["tool_name"] == "execute_services"
        assert tool_call_item["tool_calls"][0]["tool_args"]["list"][0]["domain"] == "light"

    async def test_tool_call_json_decode_error_fallback(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(
                tool_calls=[{"index": 0, "id": "call_1", "name": "test", "arguments": "{invalid json"}]
            ),
            await self._make_chunk(finish_reason="tool_calls"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        tool_call_item = next((r for r in result if "tool_calls" in r), None)
        assert tool_call_item is not None
        assert tool_call_item["tool_calls"][0]["tool_args"] == {}

    async def test_finish_reason_length(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(content=" truncated"),
            await self._make_chunk(finish_reason="length"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert {"finish_reason": "length"} in result

    async def test_finish_reason_stop(self, provider: OpenAICompatibleProvider) -> None:
        chunks = [
            await self._make_chunk(content="Done"),
            await self._make_chunk(finish_reason="stop"),
        ]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter(chunks)
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        assert {"content": "Done"} in result
        # Should cleanly break after stop; only role + content yielded
        assert len(result) == 2

    async def test_content_type_coercion(self, provider: OpenAICompatibleProvider) -> None:
        """Non-string content should be coerced or skipped."""
        chunk = await self._make_chunk(content=42)  # type: ignore[arg-type]
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter([chunk, await self._make_chunk(finish_reason="stop")])
        )

        result = []
        async for item in provider.stream_chat([], None, {}):
            result.append(item)

        content_items = [r for r in result if "content" in r]
        assert len(content_items) == 1
        assert content_items[0]["content"] == "42"

    async def test_stream_chat_with_tools_and_options(self, provider: OpenAICompatibleProvider) -> None:
        """Tools and options are forwarded to the API call."""
        provider.capabilities.supports_tools = True
        provider.capabilities.supports_tool_choice = True
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter([await self._make_chunk(finish_reason="stop")])
        )

        tools = [{"type": "function", "function": {"name": "test"}}]
        options = {
            "temperature": 0.5,
            "max_tokens": 100,
            "tool_choice": "auto",
            "schema_strict": False,
        }

        async for _ in provider.stream_chat([{"role": "user", "content": "hi"}], tools, options):
            pass

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["tool_choice"] == "auto"
        assert call_kwargs["tools"] == tools

    async def test_strip_strict_schemas_when_false(self, provider: OpenAICompatibleProvider) -> None:
        """When strict_schemas=False, strict and additionalProperties are removed."""
        provider.capabilities.supports_tools = True
        provider._client.chat.completions.create = AsyncMock(
            return_value=self._async_iter([await self._make_chunk(finish_reason="stop")])
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "strict": True,
                        "additionalProperties": False,
                        "properties": {},
                    },
                },
            }
        ]
        options = {"schema_strict": False, "tool_choice": "auto"}

        async for _ in provider.stream_chat([], tools, options):
            pass

        params = tools[0]["function"]["parameters"]
        assert "strict" not in params
        assert "additionalProperties" not in params

    async def test_exception_propagates_from_api_call(self, provider: OpenAICompatibleProvider) -> None:
        """Test that exceptions from the underlying client are re-raised."""
        provider._client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("API down")
        )

        with pytest.raises(ConnectionError, match="API down"):
            async for _ in provider.stream_chat([], None, {}):
                pass

    @staticmethod
    async def _async_iter(items: list[MagicMock]) -> AsyncGenerator[MagicMock, None]:
        for item in items:
            yield item
