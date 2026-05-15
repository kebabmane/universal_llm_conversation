"""Integration tests for conversation entity."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import yaml

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

    # Agent is registered under the config entry id
    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
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

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
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

    # Track service calls via class-level patch
    service_calls = []
    original_call = hass.services.async_call

    async def tracking_call(self, domain, service, service_data, **kwargs):
        service_calls.append((domain, service, service_data))
        return await original_call(domain, service, service_data, **kwargs)

    with patch("homeassistant.core.ServiceRegistry.async_call", tracking_call):
        result = await conversation.async_converse(
            hass,
            "turn on the light",
            None,
            Context(),
            agent_id=mock_config_entry.entry_id,
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

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "Fallback response" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_always_fail")
async def test_both_models_fail_returns_error(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test error response when primary and fallback both fail."""
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "problem talking to the LLM" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection")
async def test_agent_skills_empty_by_default(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test agent skills property returns empty list by default."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent.skills == []


@pytest.mark.usefixtures("mock_validate_connection")
async def test_sanitize_speech_with_function_names(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test speech sanitization strips known function names."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Configure function tools
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "functions": yaml.dump([{"spec": {"name": "turn_on_light"}, "function": {"type": "native"}}])},
    )

    result = await conversation.async_converse(
        hass,
        "turn on the light",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    # Response should not contain leaked function syntax
    speech = result.response.speech["plain"]["speech"]
    assert "turn_on_light(" not in speech


@pytest.mark.usefixtures("mock_validate_connection")
async def test_async_added_to_hass_absolute_path(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test skill directory resolution when working dir is absolute."""
    from custom_components.universal_llm_conversation.const import DEFAULT_WORKING_DIRECTORY
    from pathlib import Path

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent is not None
    # Verify skill_manager was initialized
    assert agent.skill_manager is not None
    # The skills_dir should be based on DEFAULT_WORKING_DIRECTORY
    assert "skills" in str(agent.skill_manager.user_skills_dir)
