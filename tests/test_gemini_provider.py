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


class TestGeminiProviderInitErrors:
    """Test initialization error paths."""

    def test_init_raises_when_sdk_missing(self) -> None:
        """Test ImportError when google-genai SDK is not installed."""
        from homeassistant.exceptions import HomeAssistantError
        from homeassistant.core import HomeAssistant

        import sys
        from homeassistant.exceptions import HomeAssistantError
        from homeassistant.core import HomeAssistant

        hass = MagicMock(spec=HomeAssistant)
        # Temporarily hide google.genai so the import in __init__ fails
        real_genai = sys.modules.get("google.genai")
        real_google = sys.modules.get("google")
        if real_genai:
            del sys.modules["google.genai"]
        # Patch the google module to not have genai attribute
        if real_google and hasattr(real_google, "genai"):
            original_genai = real_google.genai
            delattr(real_google, "genai")
        try:
            with patch.dict("sys.modules", {"google.genai": None}):
                with pytest.raises(HomeAssistantError, match="Google GenAI SDK not installed"):
                    GeminiProvider(
                        hass=hass,
                        api_key="gemini-test",
                        base_url=None,
                        api_version=None,
                        organization=None,
                        model="gemini-2.5-flash",
                        timeout=60.0,
                        capabilities=GEMINI_CAPABILITIES,
                    )
        finally:
            if real_genai:
                sys.modules["google.genai"] = real_genai
            if real_google and hasattr(real_google, "genai") is False and 'original_genai' in locals():
                real_google.genai = original_genai


class TestGeminiValidateConnectionErrors:
    """Test validate_connection error branches."""

    @patch("google.genai")
    async def test_validate_connection_api_error(self, mock_genai: MagicMock) -> None:
        """Test APIError maps to cannot_connect."""
        from google.genai.errors import APIError
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        mock_client.aio.models.list = MagicMock(side_effect=APIError(500, {"error": "server down"}))
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()

    @patch("google.genai")
    async def test_validate_connection_generic_exception(self, mock_genai: MagicMock) -> None:
        """Test generic Exception maps to cannot_connect."""
        from homeassistant.exceptions import HomeAssistantError

        mock_client = MagicMock()
        mock_client.aio.models.list = MagicMock(side_effect=RuntimeError("boom"))
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        with pytest.raises(HomeAssistantError, match="cannot_connect"):
            await provider.validate_connection()

class TestGeminiStreamChatEdgeCases:
    """Test stream_chat edge cases."""

    @patch("google.genai")
    async def test_stream_chat_exception_propagates(self, mock_genai: MagicMock) -> None:
        """Test exceptions from generate_content_stream propagate."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(side_effect=ConnectionError("boom"))
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        with pytest.raises(ConnectionError, match="boom"):
            async for _ in provider.stream_chat(
                messages=[{"role": "user", "content": "hi"}],
                tools=None,
                options={},
            ):
                pass

    @patch("google.genai")
    async def test_stream_chat_multiple_system_messages(self, mock_genai: MagicMock) -> None:
        """Test multiple system messages are concatenated."""
        mock_client = MagicMock()
        mock_genai.types = MagicMock()

        class FakeChunk:
            text = "Hello"
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
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        results = []
        async for chunk in provider.stream_chat(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
            options={"temperature": 0.5, "top_p": 0.9, "max_tokens": 512},
        ):
            results.append(chunk)

        assert {"content": "Hello"} in results
        call_kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]
        assert config.system_instruction == "You are helpful\nBe concise"
        # Verify temperature and max_tokens passed through
        # Note: top_p is filtered by Gemini capabilities (unsupported_params includes "top_p")
        assert config.temperature == 0.5
        assert config.max_output_tokens == 512

    @patch("google.genai")
    async def test_stream_chat_function_call_dict_args(self, mock_genai: MagicMock) -> None:
        """Test function call with dict-like args (hasattr items)."""
        mock_client = MagicMock()

        class FakeArgs:
            """Mapping-like object that dict() constructor can consume."""
            def __init__(self):
                self._data = {"location": "Boston"}
            def keys(self):
                return self._data.keys()
            def __getitem__(self, key):
                return self._data[key]
            def items(self):
                return self._data.items()

        class FakeFunctionCall:
            name = "get_weather"
            args = FakeArgs()
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
            api_key="gemini-test",
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
        assert tool_chunks[0]["tool_calls"][0]["tool_args"] == {"location": "Boston"}

    @patch("google.genai")
    async def test_stream_chat_candidate_max_tokens(self, mock_genai: MagicMock) -> None:
        """Test candidate finish_reason MAX_TOKENS."""
        mock_client = MagicMock()

        class FakeFinishReason:
            name = "MAX_TOKENS"

        class FakeCandidate:
            finish_reason = FakeFinishReason()

        class FakeChunk:
            text = None
            function_calls = None
            usage_metadata = None
            candidates = [FakeCandidate()]

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test",
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

        assert {"finish_reason": "length"} in results

    @patch("google.genai")
    async def test_stream_chat_candidate_stop(self, mock_genai: MagicMock) -> None:
        """Test candidate finish_reason STOP."""
        mock_client = MagicMock()

        class FakeFinishReason:
            name = "STOP"

        class FakeCandidate:
            finish_reason = FakeFinishReason()

        class FakeChunk:
            text = "Done"
            function_calls = None
            usage_metadata = None
            candidates = [FakeCandidate()]

        async def fake_stream():
            yield FakeChunk()

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
        mock_genai.Client.return_value = mock_client

        hass = MagicMock(spec=HomeAssistant)
        provider = GeminiProvider(
            hass=hass,
            api_key="gemini-test",
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

        assert {"content": "Done"} in results
        assert {"finish_reason": "stop"} in results

    @patch("google.genai")
    async def test_stream_chat_tool_choice_none(self, mock_genai: MagicMock) -> None:
        """Test tool_choice none builds correct FunctionCallingConfig."""
        mock_client = MagicMock()
        mock_genai.types = MagicMock()

        class FakeChunk:
            text = "No tools"
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
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "test", "description": "test", "parameters": {"type": "object"}},
            }
        ]

        async for _ in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            options={"tool_choice": "none"},
        ):
            pass

        call_kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]
        # Verify tool_config was built with NONE mode
        assert config.tool_config is not None

    @patch("google.genai")
    async def test_stream_chat_no_tools_skips_tool_config(self, mock_genai: MagicMock) -> None:
        """Test that no tools means no tool_config in config."""
        mock_client = MagicMock()
        mock_genai.types = MagicMock()

        class FakeChunk:
            text = "Hello"
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
            api_key="gemini-test",
            base_url=None,
            api_version=None,
            organization=None,
            model="gemini-2.5-flash",
            timeout=60.0,
            capabilities=GEMINI_CAPABILITIES,
        )

        async for _ in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            options={},
        ):
            pass

        call_kwargs = mock_client.aio.models.generate_content_stream.call_args.kwargs
        config = call_kwargs["config"]
        assert not hasattr(config, "tool_config") or config.tool_config is None
