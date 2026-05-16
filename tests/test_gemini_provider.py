"""Tests for the native Google Gemini provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import google.genai  # noqa: F401
import pytest

from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.providers import GEMINI_CAPABILITIES
from custom_components.universal_llm_conversation.providers.gemini import GeminiProvider


class TestGeminiProviderInit:
    """Test provider initialization."""

    @patch("google.genai")
    def test_init_creates_client(self, mock_genai: MagicMock) -> None:
        """Test that init creates the Gemini client."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )
        assert provider.model == "gemini-2.5-flash"
        mock_genai.Client.assert_called_once_with(api_key="gemini-test-key")


class TestGeminiValidateConnection:
    """Test validate_connection."""

    @patch("google.genai")
    async def test_validate_connection_success(self, mock_genai: MagicMock) -> None:
        """Test successful validation."""
        mock_client = MagicMock()
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [MagicMock()]
        mock_client.aio.models.list = MagicMock(return_value=mock_response)
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        result = await provider.validate_connection()
        assert result is True

    @patch("google.genai")
    async def test_validate_connection_auth_error(self, mock_genai: MagicMock) -> None:
        """Test 401 authentication error."""
        from google.genai.errors import ClientError
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        err = ClientError(401, {"error": "auth failed"})
        mock_client.aio.models.list = MagicMock(side_effect=err)
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="bad-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="invalid_auth"):
            await provider.validate_connection()

    @patch("google.genai")
    async def test_validate_connection_server_error(self, mock_genai: MagicMock) -> None:
        """Test 5xx server error."""
        from google.genai.errors import ServerError
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        mock_client.aio.models.list = MagicMock(side_effect=ServerError(500, {"error": "server down"}))
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()


class TestGeminiStreamChat:
    """Test stream_chat streaming behavior."""

    @patch("google.genai")
    async def test_stream_chat_text_only(self, mock_genai: MagicMock) -> None:
        """Test simple text streaming."""
        mock_client = MagicMock()

        class FakeChunk:
            text = "Hello world"
            function_calls = None
            usage_metadata = None
            candidates = None

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
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

    @patch("google.genai")
    async def test_stream_chat_with_tools(self, mock_genai: MagicMock) -> None:
        """Test function call streaming."""
        mock_client = MagicMock()

        class FakeFunctionCall:
            name = "get_weather"
            args = {"location": "Boston"}
            id = "fc_123"

        class FakeChunk:
            text = None
            function_calls = [FakeFunctionCall()]
            usage_metadata = None
            candidates = None

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
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

    @patch("google.genai")
    async def test_stream_chat_usage(self, mock_genai: MagicMock) -> None:
        """Test usage metadata is yielded."""
        mock_client = MagicMock()

        class FakeUsageMeta:
            prompt_token_count = 10
            candidates_token_count = 5
            total_token_count = 15

        class FakeChunk:
            text = None
            function_calls = None
            usage_metadata = FakeUsageMeta()
            candidates = None

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
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

    @patch("google.genai")
    async def test_stream_chat_system_instruction(self, mock_genai: MagicMock) -> None:
        """Test system message is extracted to config."""
        mock_client = MagicMock()
        mock_genai.types = MagicMock()

        class FakeChunk:
            text = None
            function_calls = None
            usage_metadata = None
            candidates = None

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test-key",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        async for _ in provider.stream_chat(
            messages=[
                {"role": "system", "content": "You are a bot"},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
            options={},
        ):
            pass

        # Verify generate_content_stream was called with system_instruction in config
        call_kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]
        assert config.system_instruction == "You are a bot"


class TestGeminiToolChoiceMapping:
    """Test tool choice mapping."""

    def test_map_auto(self) -> None:
        assert GeminiProvider._map_tool_choice("auto") == "AUTO"

    def test_map_none(self) -> None:
        assert GeminiProvider._map_tool_choice("none") == "NONE"

    def test_map_any(self) -> None:
        assert GeminiProvider._map_tool_choice("any") == "ANY"

    def test_map_required(self) -> None:
        assert GeminiProvider._map_tool_choice("required") == "AUTO"

    def test_map_unknown(self) -> None:
        assert GeminiProvider._map_tool_choice("foo") == "ANY"


class TestGeminiMessageConversion:
    """Test internal message conversion helpers."""

    def test_convert_user_message(self) -> None:
        types = MagicMock()
        types.Content.side_effect = lambda **kwargs: MagicMock(**kwargs)
        result = GeminiProvider._convert_message({"role": "user", "content": "hi"}, types)
        assert result.role == "user"

    def test_convert_assistant_message(self) -> None:
        types = MagicMock()
        types.Content.side_effect = lambda **kwargs: MagicMock(**kwargs)
        result = GeminiProvider._convert_message({"role": "assistant", "content": "hello"}, types)
        assert result.role == "model"

    def test_convert_tool_message(self) -> None:
        types = MagicMock()
        types.Content.side_effect = lambda **kwargs: MagicMock(**kwargs)
        result = GeminiProvider._convert_message(
            {"role": "tool", "tool_call_id": "call_1", "content": "done"}, types
        )
        assert result.role == "user"

    def test_convert_system_returns_none(self) -> None:
        types = MagicMock()
        result = GeminiProvider._convert_message({"role": "system", "content": "sys"}, types)
        assert result is None


class TestGeminiToolConversion:
    """Test internal tool conversion helpers."""

    def test_convert_openai_tools(self) -> None:
        types = MagicMock()
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
        result = GeminiProvider._convert_tools(tools, types)
        assert result is not None
        assert len(result) == 1

    def test_convert_tools_none(self) -> None:
        types = MagicMock()
        assert GeminiProvider._convert_tools(None, types) is None

    def test_convert_tools_strips_strict(self) -> None:
        types = MagicMock()
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
        result = GeminiProvider._convert_tools(tools, types, strict=False)
        assert result is not None
        assert len(result) == 1
        # Verify strict was stripped from the schema
        func_decl = result[0].function_declarations[0]
        assert "strict" not in func_decl.parameters_json_schema
        assert "additionalProperties" not in func_decl.parameters_json_schema
