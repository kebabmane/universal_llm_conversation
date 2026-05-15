"""Integration tests for component initialization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.mark.usefixtures("mock_validate_connection")
async def test_setup_entry(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test successful setup."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.LOADED


async def test_setup_entry_validation_fails(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test setup fails when provider validation fails."""
    with patch(
        "custom_components.universal_llm_conversation.helpers.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=False)
        ),
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


@pytest.mark.usefixtures("mock_validate_connection")
async def test_unload_entry(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test unloading removes the integration."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.LOADED

    await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED


@pytest.mark.usefixtures("mock_validate_connection")
async def test_services_registered(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test that services are registered during setup."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert hass.services.has_service(DOMAIN, "reload_skills")
    assert hass.services.has_service(DOMAIN, "download_skill")
