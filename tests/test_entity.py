"""Integration tests for entity behavior."""

from __future__ import annotations

import pytest

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from custom_components.universal_llm_conversation.const import DOMAIN


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
