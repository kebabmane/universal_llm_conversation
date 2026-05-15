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
