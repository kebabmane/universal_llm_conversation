"""Integration tests for conversation entity."""

from __future__ import annotations

import pytest

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent

from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.mark.usefixtures("mock_validate_connection")
async def test_conversation_agent_setup(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test conversation entity is created."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Get the first subentry id as agent id
    subentry_id = next(iter(mock_config_entry.subentries.keys()))
    agent = conversation.async_get_agent_manager(hass).async_get_agent(
        f"{mock_config_entry.entry_id}.{subentry_id}"
    )
    assert agent is not None
    assert agent.supported_languages == "*"


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream")
async def test_converse_returns_response(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test basic conversation returns a response."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    subentry_id = next(iter(mock_config_entry.subentries.keys()))
    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=f"{mock_config_entry.entry_id}.{subentry_id}",
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "Hello from test" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_tool")
async def test_converse_executes_tool_call(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test conversation triggers tool execution."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    subentry_id = next(iter(mock_config_entry.subentries.keys()))

    # Track service calls
    service_calls = []
    original_call = hass.services.async_call

    async def tracking_call(domain, service, service_data, **kwargs):
        service_calls.append((domain, service, service_data))
        return await original_call(domain, service, service_data, **kwargs)

    hass.services.async_call = tracking_call

    result = await conversation.async_converse(
        hass,
        "turn on the light",
        None,
        Context(),
        agent_id=f"{mock_config_entry.entry_id}.{subentry_id}",
    )

    # Verify service was called
    assert any(
        call[0] == "light" and call[1] == "turn_on"
        for call in service_calls
    )


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_fallback")
async def test_fallback_model_retry(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test fallback model is used when primary fails."""
    # Configure fallback model
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    subentry_id = next(iter(mock_config_entry.subentries.keys()))
    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=f"{mock_config_entry.entry_id}.{subentry_id}",
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "Fallback response" in result.response.speech["plain"]["speech"]
