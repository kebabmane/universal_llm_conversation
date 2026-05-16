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

    def test_convert_tools_empty_function_skipped(self) -> None:
        """Tools with empty function dict are skipped."""
        tools = [
            {"type": "function", "function": {}},
            {"type": "function", "function": {"name": "valid", "description": "ok", "parameters": {}}},
        ]
        result = AnthropicProvider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "valid"


class TestAnthropicProviderInitErrors:
    """Test initialization error paths."""

    def test_init_raises_when_sdk_missing(self) -> None:
        """Test ImportError when anthropic SDK is not installed."""
        import sys
        from homeassistant.exceptions import HomeAssistantError
        from homeassistant.core import HomeAssistant

        hass = MagicMock(spec=HomeAssistant)
        # Make the import of AsyncAnthropic fail
        real_anthropic = sys.modules.get("anthropic")
        with patch.dict("sys.modules", {"anthropic": None}):
            # Also patch the import statement inside the module
            with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: __import__(name, *args, **kwargs) if name != "anthropic" else (_ for _ in ()).throw(ImportError("no module"))):
                with pytest.raises(HomeAssistantError, match="Anthropic SDK not installed"):
                    AnthropicProvider(
                        hass=hass,
                        api_key="sk-test",
                        base_url=None,
                        api_version=None,
                        organization=None,
                        model="claude-sonnet-4-5",
                        timeout=60.0,
                        capabilities=ANTHROPIC_CAPABILITIES,
                    )


class TestAnthropicValidateConnectionErrors:
    """Test validate_connection error branches."""

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_api_connection_error(self, mock_async_anthropic: MagicMock) -> None:
        """Test APIConnectionError maps to cannot_connect."""
        from anthropic import APIConnectionError
        from homeassistant.exceptions import HomeAssistantError
        import httpx

        mock_client = MagicMock()
        mock_client.models.list = MagicMock(
            side_effect=APIConnectionError(
                message="connection failed",
                request=httpx.Request("GET", "http://test"),
            )
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

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_bad_request_error(self, mock_async_anthropic: MagicMock) -> None:
        """Test BadRequestError maps to cannot_connect."""
        from anthropic import BadRequestError
        from homeassistant.exceptions import HomeAssistantError
        import httpx

        mock_client = MagicMock()
        mock_client.models.list = MagicMock(
            side_effect=BadRequestError(
                "bad request",
                response=MagicMock(status_code=400),
                body=None,
            )
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

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()

    @patch("anthropic.AsyncAnthropic")
    async def test_validate_connection_generic_exception(self, mock_async_anthropic: MagicMock) -> None:
        """Test generic Exception maps to cannot_connect."""
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        mock_client.models.list = MagicMock(side_effect=RuntimeError("boom"))
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

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()


class TestAnthropicStreamChatEdgeCases:
    """Test stream_chat edge cases."""

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_exception_propagates(self, mock_async_anthropic: MagicMock) -> None:
        """Test that exceptions from messages.create are propagated."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=ConnectionError("boom"))
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

        with pytest.raises(ConnectionError, match="boom"):
            async for _ in provider.stream_chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                options={},
            ):
                pass

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_thinking_block_start(self, mock_async_anthropic: MagicMock) -> None:
        """Test thinking block at content_block_start buffers but does not yield."""
        mock_client = MagicMock()

        class FakeThinkingBlock:
            type = "thinking"
            thinking = "Let me think..."

        class FakeThinkingStartEvent:
            type = "content_block_start"
            content_block = FakeThinkingBlock()

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeThinkingStartEvent()
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
        # Thinking block_start sets buffer but does NOT yield reasoning_content
        assert {"reasoning_content": "Let me think..."} not in results
        assert {"finish_reason": "stop"} in results

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_text_block_start(self, mock_async_anthropic: MagicMock) -> None:
        """Test text block at content_block_start."""
        mock_client = MagicMock()

        class FakeTextBlock:
            type = "text"
            text = "Hello from block"

        class FakeTextStartEvent:
            type = "content_block_start"
            content_block = FakeTextBlock()

        class FakeStopEvent:
            type = "message_stop"

        async def fake_stream():
            yield FakeTextStartEvent()
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
        assert {"content": "Hello from block"} in results

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_json_decode_error(self, mock_async_anthropic: MagicMock) -> None:
        """Test JSON decode error on tool args falls back to empty dict."""
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
            partial_json = "not valid json"

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
        assert tool_chunks[0]["tool_calls"][0]["tool_args"] == {}

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_max_tokens_finish(self, mock_async_anthropic: MagicMock) -> None:
        """Test max_tokens stop_reason yields finish_reason length."""
        mock_client = MagicMock()

        class FakeDelta:
            stop_reason = "max_tokens"

        class FakeMessageDelta:
            type = "message_delta"
            delta = FakeDelta()
            usage = None

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

        assert {"finish_reason": "length"} in results

    @patch("anthropic.AsyncAnthropic")
    async def test_stream_chat_with_system_message(self, mock_async_anthropic: MagicMock) -> None:
        """Test stream_chat with system message in messages list."""
        mock_client = MagicMock()

        class FakeTextDelta:
            type = "text_delta"
            text = "Hello"

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
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
            options={"temperature": 0.5, "max_tokens": 512},
        ):
            results.append(chunk)

        assert results[0] == {"role": "assistant"}
        assert {"content": "Hello"} in results
        # Verify system was passed to messages.create
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"
        assert call_kwargs["max_tokens"] == 512


class TestAnthropicAttachmentConversion:
    """Test _convert_message with image and PDF attachments."""

    def test_convert_message_with_image(self) -> None:
        import base64
        from custom_components.universal_llm_conversation.providers.anthropic import AnthropicProvider

        msg = {
            "role": "user",
            "content": "describe this",
            "attachments": [
                {"mime_type": "image/jpeg", "data_base64": base64.b64encode(b"fake_img").decode()}
            ],
        }
        result = AnthropicProvider._convert_message(msg)
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "describe this"}
        assert result["content"][1]["type"] == "image"
        assert result["content"][1]["source"]["type"] == "base64"
        assert result["content"][1]["source"]["media_type"] == "image/jpeg"
        assert result["content"][1]["source"]["data"] == base64.b64encode(b"fake_img").decode()

    def test_convert_message_with_pdf(self) -> None:
        import base64
        from custom_components.universal_llm_conversation.providers.anthropic import AnthropicProvider

        msg = {
            "role": "user",
            "content": "summarize this",
            "attachments": [
                {"mime_type": "application/pdf", "data_base64": base64.b64encode(b"fake_pdf").decode()}
            ],
        }
        result = AnthropicProvider._convert_message(msg)
        assert result["content"][1]["type"] == "document"
        assert result["content"][1]["source"]["media_type"] == "application/pdf"

    def test_convert_message_without_attachments(self) -> None:
        from custom_components.universal_llm_conversation.providers.anthropic import AnthropicProvider

        msg = {"role": "user", "content": "hello"}
        result = AnthropicProvider._convert_message(msg)
        assert result == {"role": "user", "content": "hello"}
