"""Integration tests for entity behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, llm

from custom_components.universal_llm_conversation.const import DOMAIN
from custom_components.universal_llm_conversation.entity import _convert_content_to_param


@pytest.mark.usefixtures("mock_validate_connection")
async def test_entity_device_info(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test entity device info shows model and fallback."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Configure fallback
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )
    await hass.async_block_till_done()

    device_registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(
        device_registry, mock_config_entry.entry_id
    )

    assert len(devices) > 0
    # Check model shows primary → fallback
    assert "gpt-4o-mini" in devices[0].model or "→" in (devices[0].model or "")


@pytest.mark.usefixtures("mock_validate_connection")
async def test_entity_state_after_setup(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test entity appears in state machine."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # The conversation entity should be registered
    states = hass.states.async_all("conversation")
    assert len(states) > 0


def test_convert_content_to_param_system_user() -> None:
    """Test conversion of system and user content."""
    content = [
        conversation.SystemContent(content="You are a bot"),
        conversation.UserContent(content="hello"),
    ]
    result = _convert_content_to_param(content)
    assert result == [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "hello"},
    ]


def test_convert_content_to_param_assistant_with_tools() -> None:
    """Test conversion of assistant content with tool calls."""
    content = [
        conversation.AssistantContent(
            agent_id="agent_1",
            content="Turn on the light",
            tool_calls=[
                llm.ToolInput(
                    id="call_123",
                    tool_name="execute_services",
                    tool_args={"list": [{"domain": "light", "service": "turn_on"}]},
                )
            ],
        ),
    ]
    result = _convert_content_to_param(content)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "Turn on the light"
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["id"] == "call_123"


def test_convert_content_to_param_tool_result() -> None:
    """Test conversion of tool result content."""
    content = [
        conversation.ToolResultContent(
            agent_id="agent_1",
            tool_call_id="call_123",
            tool_name="execute_services",
            tool_result="done",
        ),
    ]
    result = _convert_content_to_param(content)
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_123"
    assert result[0]["content"] == '"done"'


def test_convert_content_to_param_shorten_tool_call_id() -> None:
    """Test tool call ID shortening."""
    from custom_components.universal_llm_conversation.helpers import shorten_tool_call_id as shortener
    content = [
        conversation.AssistantContent(
            agent_id="agent_1",
            content="",
            tool_calls=[
                llm.ToolInput(
                    id="call_very_long_id_here",
                    tool_name="test",
                    tool_args={},
                )
            ],
        ),
    ]
    result = _convert_content_to_param(content, shorten_tool_call_id=shortener)
    assert result[0]["tool_calls"][0]["id"] != "call_very_long_id_here"
    assert len(result[0]["tool_calls"][0]["id"]) < len("call_very_long_id_here")


def test_convert_content_to_param_empty_tool_calls_popped() -> None:
    """Test empty tool_calls list is removed from assistant message."""
    content = [
        conversation.AssistantContent(
            agent_id="agent_1",
            content="",
            tool_calls=[],
        ),
    ]
    result = _convert_content_to_param(content)
    assert "tool_calls" not in result[0]


def test_convert_content_to_param_with_image() -> None:
    """Test user message with image attachment gets base64-encoded."""
    from pathlib import Path
    from unittest.mock import MagicMock

    att = MagicMock()
    att.mime_type = "image/jpeg"
    att.path = MagicMock()
    att.path.read_bytes = MagicMock(return_value=b"fake_image_data")

    content = [
        conversation.UserContent(content="describe this", attachments=[att]),
    ]
    result = _convert_content_to_param(content)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "describe this"
    assert "attachments" in result[0]
    assert len(result[0]["attachments"]) == 1
    assert result[0]["attachments"][0]["mime_type"] == "image/jpeg"
    import base64
    assert result[0]["attachments"][0]["data_base64"] == base64.b64encode(b"fake_image_data").decode()


def test_convert_content_to_param_with_pdf() -> None:
    """Test user message with PDF attachment gets base64-encoded."""
    from unittest.mock import MagicMock

    att = MagicMock()
    att.mime_type = "application/pdf"
    att.path = MagicMock()
    att.path.read_bytes = MagicMock(return_value=b"fake_pdf_data")

    content = [
        conversation.UserContent(content="summarize this", attachments=[att]),
    ]
    result = _convert_content_to_param(content)
    assert result[0]["attachments"][0]["mime_type"] == "application/pdf"
    import base64
    assert result[0]["attachments"][0]["data_base64"] == base64.b64encode(b"fake_pdf_data").decode()


def test_resize_image_if_needed_passthrough_non_image() -> None:
    """Test _resize_image_if_needed passes through non-image data unchanged."""
    from custom_components.universal_llm_conversation.entity import _resize_image_if_needed
    data = b"not_an_image"
    assert _resize_image_if_needed(data, "application/pdf") == data


def test_resize_image_if_needed_passthrough_no_pillow() -> None:
    """Test _resize_image_if_needed passes through when Pillow is unavailable."""
    from unittest.mock import patch
    from custom_components.universal_llm_conversation.entity import _resize_image_if_needed
    with patch.dict("sys.modules", {"PIL": None}):
        data = b"fake_image"
        assert _resize_image_if_needed(data, "image/jpeg") == data


def test_resize_image_if_needed_resizes_oversized() -> None:
    """Test _resize_image_if_needed downsamples images exceeding max dimension."""
    from PIL import Image
    import io
    from custom_components.universal_llm_conversation.entity import _resize_image_if_needed

    # Create a 2000x1000 image
    img = Image.new("RGB", (2000, 1000), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    original_data = buf.getvalue()

    result = _resize_image_if_needed(original_data, "image/jpeg")

    # Verify it was resized
    result_img = Image.open(io.BytesIO(result))
    assert max(result_img.size) <= 1568


@pytest.mark.usefixtures("mock_validate_connection")
async def test_entity_device_info_no_fallback(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test device info without fallback shows only primary model."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    device_registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(
        device_registry, mock_config_entry.entry_id
    )

    assert len(devices) > 0
    assert "→" not in (devices[0].model or "")


@pytest.mark.usefixtures("mock_validate_connection")
async def test_execute_function_tool_direct(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test direct function tool execution via entity."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent is not None

    # Test _execute_function_tool with mocked get_function
    with patch("custom_components.universal_llm_conversation.entity.get_function") as mock_get_function:
        mock_func = MagicMock()
        mock_func.execute = AsyncMock(return_value="MockResult")
        mock_func.validate_schema = MagicMock(return_value={"type": "test"})
        mock_get_function.return_value = mock_func

        tool_input = llm.ToolInput(
            id="call_1",
            tool_name="mock_tool",
            tool_args={"foo": "bar"},
        )
        result = await agent._execute_function_tool(
            {"spec": {"name": "mock_tool"}, "function": {"type": "test"}},
            tool_input,
            None,
            [],
        )
        assert result.tool_result == {"result": "MockResult"}


class TestEntityHelpers:
    """Test entity helper methods directly."""

    def test_adjust_schema_strict_mode(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
        }
        _adjust_schema(schema, strict=True)
        assert schema["strict"] is True
        assert schema["additionalProperties"] is False
        assert "name" in schema["required"]
        assert schema["properties"]["name"]["type"] == ["string", "null"]

    def test_adjust_schema_non_strict_mode(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        schema = {
            "type": "object",
            "strict": True,
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
            },
        }
        _adjust_schema(schema, strict=False)
        assert "strict" not in schema
        assert "additionalProperties" not in schema

    def test_adjust_schema_nested_object(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                    },
                },
            },
        }
        _adjust_schema(schema, strict=True)
        assert "nested" in schema["required"]
        assert "inner" in schema["properties"]["nested"]["required"]

    def test_adjust_schema_array_items(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        _adjust_schema(schema, strict=True)
        # Primitive array items are not changed (only object types get nullable)
        assert schema["properties"]["items"]["items"]["type"] == "string"
        # The array property itself is required
        assert "items" in schema["required"]

    def test_adjust_schema_non_dict(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        _adjust_schema("not a dict", strict=True)
        # Should not raise

    def test_adjust_schema_no_properties(self) -> None:
        from custom_components.universal_llm_conversation.entity import _adjust_schema
        schema = {"type": "object"}
        _adjust_schema(schema, strict=True)
        assert schema.get("strict") is True
        assert schema.get("additionalProperties") is False

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_should_run_in_background(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)
        assert agent._should_run_in_background({"delay": 5}) is True
        assert agent._should_run_in_background({"foo": "bar"}) is False
        assert agent._should_run_in_background(None) is False

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_truncate_message_history_clear(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        chat_log = MagicMock()
        chat_log.content = [
            conversation.SystemContent(content="sys"),
            conversation.UserContent(content="hi"),
            conversation.AssistantContent(agent_id="a", content="hello"),
            conversation.UserContent(content="bye"),
            conversation.AssistantContent(agent_id="a", content="goodbye"),
        ]
        await agent._truncate_message_history(chat_log)
        # Should delete everything between system prompt (index 0) and last user (index 3)
        assert len(chat_log.content) == 3
        assert chat_log.content[0].content == "sys"
        assert isinstance(chat_log.content[1], conversation.UserContent)
        assert chat_log.content[1].content == "bye"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_raises_on_length(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test that finish_reason=='length' raises TokenLengthExceededError."""
        from custom_components.universal_llm_conversation.exceptions import TokenLengthExceededError

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "truncated"}
            yield {"finish_reason": "length"}

        with pytest.raises(TokenLengthExceededError):
            async for _ in agent._transform_stream(
                MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ):
                pass

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_background_execution_schedules_task(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test that delay arguments trigger background task creation."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        with patch.object(agent.entry, "async_create_task") as mock_create_task:
            result = await agent._execute_function_tool(
                {"spec": {"name": "test"}, "function": {"type": "native", "name": "test"}},
                llm.ToolInput(id="call_1", tool_name="test", tool_args={"delay": 5}),
                None,
                [],
            )
        assert result.tool_result == {"result": "Scheduled"}
        mock_create_task.assert_called_once()

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_get_provider_resolves_base_url_from_preset(self, hass: HomeAssistant) -> None:
        """Test that _get_provider() resolves base_url from preset at conversation runtime."""
        from pytest_homeassistant_custom_component.common import MockConfigEntry
        from homeassistant.config_entries import ConfigSubentryData
        from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

        # Create entry with fireworks preset but NO base_url (simulating old/pre-fix entry)
        entry = MockConfigEntry(
            domain=DOMAIN,
            title="Fireworks Test",
            data={
                "api_key": "fp-test-key",
                "provider_preset": "fireworks",
                # base_url intentionally omitted
                "skip_authentication": False,
            },
            version=1,
            subentries_data=[
                ConfigSubentryData(
                    data=dict(DEFAULT_OPTIONS),
                    subentry_type="conversation",
                    title="Test Conversation",
                    unique_id=None,
                ),
            ],
        )
        entry.add_to_hass(hass)
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        agent = conversation.get_agent_manager(hass).async_get_agent(entry.entry_id)
        assert agent is not None

        with patch(
            "custom_components.universal_llm_conversation.entity.get_provider",
            return_value=MagicMock(),
        ) as mock_get_provider:
            provider = agent._get_provider()

        # Verify get_provider was called with the resolved Fireworks base_url
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_get_provider_resolves_provider_key_from_preset(self, hass: HomeAssistant) -> None:
        """Test that _get_provider() resolves provider key from preset at runtime."""
        from pytest_homeassistant_custom_component.common import MockConfigEntry
        from homeassistant.config_entries import ConfigSubentryData
        from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

        # Create entry with fireworks preset but NO provider key (simulating old entry)
        entry = MockConfigEntry(
            domain=DOMAIN,
            title="Fireworks Test",
            data={
                "api_key": "fp-test-key",
                "provider_preset": "fireworks",
                # provider key intentionally omitted
                "skip_authentication": False,
            },
            version=1,
            subentries_data=[
                ConfigSubentryData(
                    data=dict(DEFAULT_OPTIONS),
                    subentry_type="conversation",
                    title="Test Conversation",
                    unique_id=None,
                ),
            ],
        )
        entry.add_to_hass(hass)
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        agent = conversation.get_agent_manager(hass).async_get_agent(entry.entry_id)
        assert agent is not None

        with patch(
            "custom_components.universal_llm_conversation.entity.get_provider",
            return_value=MagicMock(),
        ) as mock_get_provider:
            provider = agent._get_provider()

        # Verify get_provider was called with the resolved openai_compatible provider key
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider_key"] == "openai_compatible"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_should_run_in_background(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)
        assert agent._should_run_in_background({"delay": 5}) is True
        assert agent._should_run_in_background({"foo": "bar"}) is False
        assert agent._should_run_in_background(None) is False

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_get_function_tools_error_handling(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        from custom_components.universal_llm_conversation.exceptions import FunctionLoadFailed, InvalidFunction

        with patch("yaml.safe_load", side_effect=InvalidFunction("bad yaml")):
            with pytest.raises(InvalidFunction):
                agent._get_function_tools()

        with patch("yaml.safe_load", side_effect=RuntimeError("unexpected")):
            with pytest.raises(FunctionLoadFailed):
                agent._get_function_tools()

    def test_format_structured_output(self) -> None:
        """Test structured output format conversion."""
        from custom_components.universal_llm_conversation.entity import _format_structured_output
        import voluptuous as vol

        schema = vol.Schema({vol.Required("name"): str})
        result = _format_structured_output(schema, None)
        assert result["type"] == "object"
        assert "properties" in result
        assert "name" in result["properties"]

    def test_truncate_message_history_no_user_content(self) -> None:
        """Test truncate when there is no user content after system."""
        from custom_components.universal_llm_conversation.entity import UniversalLLMBaseEntity
        chat_log = MagicMock()
        chat_log.content = [
            conversation.SystemContent(content="sys"),
            conversation.AssistantContent(agent_id="a", content="hello"),
        ]
        # Should not crash when no user content exists
        import asyncio
        asyncio.run(UniversalLLMBaseEntity._truncate_message_history(MagicMock(), chat_log))
        assert len(chat_log.content) == 2

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_accumulates_usage(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test _transform_stream populates usage_accumulator."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "hi"}
            yield {
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 4,
                    "total_tokens": 12,
                }
            }
            yield {"finish_reason": "stop"}

        usage_accumulator = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        async for _ in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator=usage_accumulator
        ):
            pass

        assert usage_accumulator["prompt_tokens"] == 8
        assert usage_accumulator["completion_tokens"] == 4
        assert usage_accumulator["total_tokens"] == 12

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_sentence_mode_single(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test sentence mode yields one sentence when boundary found."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "Hello world."}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 1
        assert content_deltas[0]["content"] == "Hello world."

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_sentence_mode_multi(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test sentence mode splits multi-sentence chunks."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "First sentence. Second sentence? Third!"}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 3
        assert content_deltas[0]["content"] == "First sentence."
        assert content_deltas[1]["content"] == " Second sentence?"
        assert content_deltas[2]["content"] == " Third!"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_sentence_mode_no_boundary(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test sentence mode buffers unterminated text until stream end."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "No boundary here"}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 1
        assert content_deltas[0]["content"] == "No boundary here"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_sentence_mode_tool_call_flush(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test tool call forces sentence buffer flush."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "Before tool "}
            yield {"tool_calls": [{"id": "call_1", "tool_name": "test", "tool_args": {}}]}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 1
        assert content_deltas[0]["content"] == "Before tool "
        tool_deltas = [d for d in results if "tool_calls" in d]
        assert len(tool_deltas) == 1

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_sentence_mode_unicode(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test Unicode sentence boundaries (。, ？, ！)."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "你好。怎么样？很好！"}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 3
        assert content_deltas[0]["content"] == "你好。"
        assert content_deltas[1]["content"] == "怎么样？"
        assert content_deltas[2]["content"] == "很好！"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_token_mode_unchanged(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test token mode still yields raw chunks."""
        subentry = next(iter(mock_config_entry.subentries.values()))
        hass.config_entries.async_update_subentry(
            mock_config_entry,
            subentry,
            data={**subentry.data, "tts_streaming_mode": "token"},
        )
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "Hello"}
            yield {"content": " world"}
            yield {"finish_reason": "stop"}

        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=[], usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 2
        assert content_deltas[0]["content"] == "Hello"
        assert content_deltas[1]["content"] == " world"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_reasoning_content_hidden(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test reasoning_content is collected but not yielded when hide_thinking=True."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "Hello"}
            yield {"reasoning_content": "thinking..."}
            yield {"finish_reason": "stop"}

        reasoning_parts = []
        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=True, reasoning_parts=reasoning_parts, usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 1
        assert content_deltas[0]["content"] == "Hello"
        assert reasoning_parts == ["thinking..."]

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_transform_stream_reasoning_content_visible(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test reasoning_content is yielded when hide_thinking=False in token mode."""
        subentry = next(iter(mock_config_entry.subentries.values()))
        hass.config_entries.async_update_subentry(
            mock_config_entry,
            subentry,
            data={**subentry.data, "tts_streaming_mode": "token"},
        )
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "Hello"}
            yield {"reasoning_content": "thinking..."}
            yield {"finish_reason": "stop"}

        reasoning_parts = []
        results = []
        async for delta in agent._transform_stream(
            MagicMock(), fake_stream(), hide_thinking=False, reasoning_parts=reasoning_parts, usage_accumulator={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        ):
            results.append(delta)

        content_deltas = [d for d in results if "content" in d]
        assert len(content_deltas) == 2
        assert content_deltas[0]["content"] == "Hello"
        assert content_deltas[1]["content"] == "thinking..."
        assert reasoning_parts == []

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_structured_output_formatting(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test _async_handle_chat_log passes response_format when structure is provided."""
        import voluptuous as vol
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"content": "hi"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4"
        mock_provider.supports_tools = False
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            chat_log = MagicMock()
            chat_log.content = [
                conversation.SystemContent(content="sys"),
                conversation.UserContent(content="hi"),
            ]

            async def mock_delta_stream(entity_id, stream):
                async for _ in stream:
                    pass
                return
                yield  # force async generator

            chat_log.async_add_delta_content_stream = mock_delta_stream
            chat_log.unresponded_tool_results = False

            await agent._async_handle_chat_log(
                chat_log,
                function_tools=[],
                exposed_entities=[],
                structure=vol.Schema({"type": "object", "properties": {"name": {"type": "string"}}}),
                structure_name="test_schema",
            )

        call_kwargs = mock_provider.stream_chat.call_args.kwargs
        assert "options" in call_kwargs
        assert "response_format" in call_kwargs["options"]
        assert call_kwargs["options"]["response_format"]["type"] == "json_schema"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_function_not_found_raises(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test _async_handle_chat_log raises FunctionNotFound for unknown tool."""
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"role": "assistant"}
            yield {"tool_calls": [{"id": "call_1", "tool_name": "unknown_tool", "tool_args": {}}]}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4"
        mock_provider.supports_tools = True
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            chat_log = MagicMock()
            chat_log.content = [
                conversation.SystemContent(content="sys"),
                conversation.UserContent(content="hi"),
            ]

            async def mock_delta_stream(entity_id, stream):
                async for chunk in stream:
                    if isinstance(chunk, dict) and chunk.get("tool_calls"):
                        yield conversation.AssistantContent(
                            agent_id="agent",
                            content="",
                            tool_calls=[llm.ToolInput(id="call_1", tool_name="unknown_tool", tool_args={}, external=True)],
                        )
                    else:
                        yield conversation.AssistantContent(agent_id="agent", content="hi")
                return
                yield  # force async generator

            chat_log.async_add_delta_content_stream = mock_delta_stream
            chat_log.unresponded_tool_results = False

            from custom_components.universal_llm_conversation.exceptions import FunctionNotFound

            with pytest.raises(FunctionNotFound):
                await agent._async_handle_chat_log(
                    chat_log,
                    function_tools=[{"spec": {"name": "known_tool"}}],
                    exposed_entities=[],
                )

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_happy_path(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images returns streamed response for vision-capable provider."""
        import asyncio
        from homeassistant.exceptions import HomeAssistantError
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "It is a cat"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.object(hass.config, "is_allowed_path", return_value=True):
                    with patch("custom_components.universal_llm_conversation.entity.Path") as mock_path_cls:
                        mock_path_inst = MagicMock()
                        mock_path_inst.exists.return_value = True
                        mock_path_inst.read_bytes.return_value = b"fake_jpeg"
                        mock_path_cls.return_value = mock_path_inst
                        with patch(
                            "custom_components.universal_llm_conversation.entity.mimetypes.guess_type",
                            return_value=("image/jpeg", None),
                        ):
                            with patch(
                                "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                                return_value=b"fake_jpeg",
                            ):
                                result = await agent.async_analyze_images(
                                    "What is this?", ["/tmp/test.jpg"], max_tokens=3000
                                )

        assert result == "It is a cat"
        call_kwargs = mock_provider.stream_chat.call_args.kwargs
        assert call_kwargs.get("tools") is None
        assert call_kwargs["options"]["tool_choice"] == "none"
        assert call_kwargs["options"]["schema_strict"] is False
        assert "messages" in call_kwargs
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "What is this?"
        assert len(call_kwargs["messages"][0]["attachments"]) == 1
        assert call_kwargs["options"]["max_tokens"] == 3000

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_reasoning_content(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images collects reasoning_content chunks (e.g. Kimi)."""
        import asyncio
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"reasoning_content": "It is a cat"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "kimi-k2p6"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.object(hass.config, "is_allowed_path", return_value=True):
                    with patch("custom_components.universal_llm_conversation.entity.Path") as mock_path_cls:
                        mock_path_inst = MagicMock()
                        mock_path_inst.exists.return_value = True
                        mock_path_inst.read_bytes.return_value = b"fake_jpeg"
                        mock_path_cls.return_value = mock_path_inst
                        with patch(
                            "custom_components.universal_llm_conversation.entity.mimetypes.guess_type",
                            return_value=("image/jpeg", None),
                        ):
                            with patch(
                                "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                                return_value=b"fake_jpeg",
                            ):
                                result = await agent.async_analyze_images(
                                    "What is this?", ["/tmp/test.jpg"]
                                )

        assert result == "It is a cat"

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_vision_disabled(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images raises when model lacks vision support."""
        from homeassistant.exceptions import HomeAssistantError
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        mock_provider = MagicMock()
        mock_provider.model = "gpt-3.5-turbo"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=False)

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with pytest.raises(HomeAssistantError) as exc_info:
                await agent.async_analyze_images("What is this?", ["/tmp/test.jpg"])
        assert "does not support image analysis" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_token_limit_warning(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images warns and stops when token limit is hit."""
        import asyncio
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "Partial"}
            yield {"finish_reason": "length"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.object(hass.config, "is_allowed_path", return_value=True):
                    with patch("custom_components.universal_llm_conversation.entity.Path") as mock_path_cls:
                        mock_path_inst = MagicMock()
                        mock_path_inst.exists.return_value = True
                        mock_path_inst.read_bytes.return_value = b"fake_jpeg"
                        mock_path_cls.return_value = mock_path_inst
                        with patch(
                            "custom_components.universal_llm_conversation.entity.mimetypes.guess_type",
                            return_value=("image/jpeg", None),
                        ):
                            with patch(
                                "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                                return_value=b"fake_jpeg",
                            ):
                                with patch("custom_components.universal_llm_conversation.entity._LOGGER") as mock_logger:
                                    result = await agent.async_analyze_images(
                                        "What is this?", ["/tmp/test.jpg"]
                                    )

        assert result == "Partial"
        mock_logger.warning.assert_called_once()
        assert "token limit" in mock_logger.warning.call_args[0][0].lower()

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_path_not_allowed(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images raises for disallowed local path."""
        from homeassistant.exceptions import HomeAssistantError
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        mock_provider = MagicMock()
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass.config, "is_allowed_path", return_value=False):
                with pytest.raises(HomeAssistantError) as exc_info:
                    await agent.async_analyze_images("What is this?", ["/etc/passwd"])
        assert "Path not allowed" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_file_not_found(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images raises ServiceValidationError for missing file."""
        from homeassistant.exceptions import ServiceValidationError
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        mock_provider = MagicMock()
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass.config, "is_allowed_path", return_value=True):
                with patch("custom_components.universal_llm_conversation.entity.Path") as mock_path_cls:
                    mock_path_inst = MagicMock()
                    mock_path_inst.exists.return_value = False
                    mock_path_cls.return_value = mock_path_inst
                    with pytest.raises(ServiceValidationError) as exc_info:
                        await agent.async_analyze_images("What is this?", ["/tmp/missing.jpg"])
        assert "File not found" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_camera_source(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images resolves camera media-source URIs."""
        import asyncio
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "A camera view"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        mock_camera_mod = MagicMock()
        mock_camera_mod.async_get_image = AsyncMock(
            return_value=MagicMock(content=b"snapshot", content_type="image/jpeg")
        )

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.dict("sys.modules", {"homeassistant.components.camera": mock_camera_mod}):
                    with patch(
                        "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                        return_value=b"snapshot",
                    ):
                        result = await agent.async_analyze_images(
                            "Describe the camera",
                            ["media-source://camera/camera.front_door"],
                        )

        assert result == "A camera view"
        mock_camera_mod.async_get_image.assert_awaited_once_with(hass, "camera.front_door")

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_image_source(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images resolves image entity media-source URIs."""
        import asyncio
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "An image"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        mock_image_mod = MagicMock()
        mock_image_mod.async_get_image = AsyncMock(
            return_value=MagicMock(content=b"img_data", content_type="image/png")
        )

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.dict("sys.modules", {"homeassistant.components.image": mock_image_mod}):
                    with patch(
                        "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                        return_value=b"img_data",
                    ):
                        result = await agent.async_analyze_images(
                            "Describe the image",
                            ["media-source://image/image.test"],
                        )

        assert result == "An image"
        mock_image_mod.async_get_image.assert_awaited_once_with(hass, "image.test")

    @pytest.mark.usefixtures("mock_validate_connection")
    async def test_async_analyze_images_generic_media_source(self, hass: HomeAssistant, mock_config_entry: object) -> None:
        """Test async_analyze_images resolves generic media-source URIs."""
        import asyncio
        from custom_components.universal_llm_conversation.providers import ProviderCapabilities

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()
        agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

        async def fake_stream():
            yield {"content": "A video frame"}
            yield {"finish_reason": "stop"}

        mock_provider = MagicMock()
        mock_provider.model = "gpt-4o"
        mock_provider.capabilities = ProviderCapabilities(supports_vision=True)
        mock_provider.filter_params = lambda p: p
        mock_provider.stream_chat = MagicMock(return_value=fake_stream())

        def mock_executor(fn, *args):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(fn(*args))
            return future

        mock_media = MagicMock()
        mock_media.path = "/media/test.jpg"
        mock_media.mime_type = "image/jpeg"

        mock_media_source_mod = MagicMock()
        mock_media_source_mod.async_resolve_media = AsyncMock(return_value=mock_media)

        with patch.object(agent, "_get_provider", return_value=mock_provider):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor):
                with patch.dict("sys.modules", {"homeassistant.components.media_source": mock_media_source_mod}):
                    with patch("custom_components.universal_llm_conversation.entity.Path") as mock_path_cls:
                        mock_path_inst = MagicMock()
                        mock_path_inst.read_bytes.return_value = b"media_bytes"
                        mock_path_cls.return_value = mock_path_inst
                        with patch(
                            "custom_components.universal_llm_conversation.entity._resize_image_if_needed",
                            return_value=b"media_bytes",
                        ):
                            result = await agent.async_analyze_images(
                                "Describe the media",
                                ["media-source://media_source/local/test.jpg"],
                            )

        assert result == "A video frame"
        mock_media_source_mod.async_resolve_media.assert_awaited_once_with(hass, "media-source://media_source/local/test.jpg", None)


