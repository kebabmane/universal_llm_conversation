"""Tests for the native Anthropic provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.providers import ANTHROPIC_CAPABILITIES
from custom_components.universal_llm_conversation.providers.anthropic import AnthropicProvider


class TestAnthropicProviderInit:
    """Test provider initialization."""

    @patch("anthropic.AsyncAnthropic")
    def test_init_creates_client(self, mock_async_anthropic: MagicMock) -> None:
        """Test that init creates the AsyncAnthropic client."""
        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )
        assert provider.model == "claude-sonnet-4-5"
        mock_async_anthropic.assert_called_once_with(api_key="sk-test", timeout=60.0)


class TestAnthropicValidateConnection:
    """Test validate_connection."""

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_success(self, mock_async_anthropic: MagicMock) -> None:
        """Test successful validation."""
        mock_client = MagicMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [MagicMock()]
        mock_client.models.list.return_value = mock_response
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        result = await provider.validate_connection()
        assert result is True

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_auth_error(self, mock_async_anthropic: MagicMock) -> None:
        """Test 401 authentication error."""
        from anthropic import AuthenticationError
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        mock_client.models.list.side_effect = AuthenticationError(
            "invalid auth", response=MagicMock(), body=None
        )
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="bad-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="invalid_auth"):
            await provider.validate_connection()

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_timeout(self, mock_async_anthropic: MagicMock) -> None:
        """Test timeout error."""
        from anthropic import APITimeoutError
        from homeassistant.exceptions import HomeAssistantError
        import httpx

        mock_client = MagicMock()
        mock_client.models.list.side_effect = APITimeoutError(
            request=httpx.Request("GET", "http://test")
        )
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="timeout"):
            await provider.validate_connection()


class TestAnthropicStreamChat:
    """Test stream_chat streaming behavior."""

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_text_only(self, mock_async_anthropic: MagicMock) -> None:
        """Test simple text streaming."""
        mock_client = MagicMock()

        # Build fake streaming events
        class FakeTextDelta:
            type = "text_delta"
            text = "Hello world"

        class FakeDeltaEvent:
            type = "content_block_delta"
            delta = FakeTextDelta()

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeDeltaEvent()
            yield FakeStopEvent()

        mock_client.messages.create = AsyncMock(return_value=fake_stream())
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        results = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options={},
        ):
            results.append(chunk)

        assert results[0] == {"role": "assistant"}
        assert results[1] == {"content": "Hello world"}
        assert results[-1] == {"finish_reason": "stop"}

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_with_tools(self, mock_async_anthropic: MagicMock) -> None:
        """Test tool call accumulation."""
        mock_client = MagicMock()

        class FakeToolUseStart:
            type = "tool_use"
            id = "toolu_123"
            name = "get_weather"

        class FakeToolStartEvent:
            type = "content_block_start"
            content_block = FakeToolUseStart()

        class FakeJsonDelta:
            type = "input_json_delta"
            partial_json = '{"location": "Boston"}'

        class FakeJsonEvent:
            type = "content_block_delta"
            delta = FakeJsonDelta()

        class FakeToolStopEvent:
            type = "content_block_stop"

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeToolStartEvent()
            yield FakeJsonEvent()
            yield FakeToolStopEvent()
            yield FakeStopEvent()

        mock_client.messages.create = AsyncMock(return_value=fake_stream())
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        results = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "weather?"}],
            tools=None,
            options={},
        ):
            results.append(chunk)

        tool_chunks = [r for r in results if "tool_calls" in r]
        assert len(tool_chunks) == 1
        assert tool_chunks[0]["tool_calls"][0]["tool_name"] == "get_weather"
        assert tool_chunks[0]["tool_calls"][0]["tool_args"] == {"location": "Boston"}

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_thinking_content(self, mock_async_anthropic: MagicMock) -> None:
        """Test thinking/reasoning content is emitted."""
        mock_client = MagicMock()

        class FakeThinkingDelta:
            type = "thinking_delta"
            thinking = "I need to check the weather."

        class FakeThinkingEvent:
            type = "content_block_delta"
            delta = FakeThinkingDelta()

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeThinkingEvent()
            yield FakeStopEvent()

        mock_client.messages.create = AsyncMock(return_value=fake_stream())
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        results = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options={},
        ):
            results.append(chunk)

        reasoning_chunks = [r for r in results if "reasoning_content" in r]
        assert len(reasoning_chunks) == 1
        assert reasoning_chunks[0]["reasoning_content"] == "I need to check the weather."

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_usage(self, mock_async_anthropic: MagicMock) -> None:
        """Test usage metadata is yielded."""
        mock_client = MagicMock()

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeMessageDelta:
            type = "message_delta"
            usage = FakeUsage()
            delta = MagicMock(stop_reason="end_turn")

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeMessageDelta()
            yield FakeStopEvent()

        mock_client.messages.create = AsyncMock(return_value=fake_stream())
        mock_async_anthropic.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = AnthropicProvider(
            hass=hass,
            api_key="sk-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="claude-sonnet-4-5",
            timeout=60.0,
            capabilities=ANTHROPIC_CAPABILITIES,
        )

        results = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options={},
        ):
            results.append(chunk)

        usage_chunks = [r for r in results if "usage" in r]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["prompt_tokens"] == 10
        assert usage_chunks[0]["usage"]["completion_tokens"] == 5
        assert usage_chunks[0]["usage"]["total_tokens"] == 15


class TestAnthropicMessageConversion:
    """Test internal message conversion helpers."""

    def test_convert_user_message(self) -> None:
        result = AnthropicProvider._convert_message({"role": "user", "content": "hi"})
        assert result == {"role": "user", "content": "hi"}

    def test_convert_assistant_message(self) -> None:
        result = AnthropicProvider._convert_message({"role": "assistant", "content": "hello"})
        assert result == {"role": "assistant", "content": "hello"}

    def test_convert_tool_message(self) -> None:
        result = AnthropicProvider._convert_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "done"}
        )
        assert result["role"] == "user"
        assert result["content"][0]["type"] == "tool_result"
        assert result["content"][0]["tool_use_id"] == "call_1"

    def test_convert_system_message_returns_none(self) -> None:
        result = AnthropicProvider._convert_message({"role": "system", "content": "sys"})
        assert result is None


class TestAnthropicToolConversion:
    """Test internal tool conversion helpers."""

    def test_convert_openai_tools(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        result = AnthropicProvider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["input_schema"]["type"] == "object"

    def test_convert_tools_none(self) -> None:
        assert AnthropicProvider._convert_tools(None) is None

    def test_convert_tools_strips_strict(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "test",
                    "parameters": {
                        "type": "object",
                        "strict": True,
                        "additionalProperties": False,
                        "properties": {},
                    },
                },
            }
        ]
        result = AnthropicProvider._convert_tools(tools, strict=False)
        assert "strict" not in result[0]["input_schema"]
        assert "additionalProperties" not in result[0]["input_schema"]
