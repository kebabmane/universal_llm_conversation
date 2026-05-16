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


