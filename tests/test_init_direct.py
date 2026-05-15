"""Direct tests for init logic without full HA integration setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.const import DOMAIN


class TestSetupEntryLogic:
    """Test setup entry logic with mocked dependencies."""

    async def test_setup_validates_provider(self) -> None:
        import custom_components.universal_llm_conversation as init_module

        hass = MagicMock(spec=HomeAssistant)
        hass.data = {}
        hass.bus = MagicMock()
        hass.config_entries.async_forward_entry_setups = AsyncMock()
        entry = MagicMock()
        entry.data = {
            "api_key": "test-key",
            "provider": "openai_compatible",
            "base_url": "http://localhost:1234/v1",
            "skip_authentication": False,
        }
        entry.runtime_data = None
        entry.add_update_listener = MagicMock(return_value=MagicMock())
        entry.async_on_unload = MagicMock()

        mock_provider = MagicMock()
        mock_provider.validate_connection = AsyncMock(return_value=True)

        with patch.object(init_module, "get_provider", return_value=mock_provider):
            result = await init_module.async_setup_entry(hass, entry)

        assert result is True
        hass.config_entries.async_forward_entry_setups.assert_called_once()

    @patch("custom_components.universal_llm_conversation.async_setup_services")
    async def test_setup_fails_on_bad_provider(self, mock_setup_services) -> None:
        from homeassistant.exceptions import ConfigEntryNotReady
        from custom_components.universal_llm_conversation import async_setup_entry

        hass = MagicMock(spec=HomeAssistant)
        entry = MagicMock()
        entry.data = {
            "api_key": "bad-key",
            "provider": "openai_compatible",
            "base_url": "http://bad-url",
            "skip_authentication": False,
        }

        with patch(
            "custom_components.universal_llm_conversation.helpers.get_provider",
            return_value=MagicMock(
                validate_connection=AsyncMock(return_value=False)
            ),
        ), pytest.raises(ConfigEntryNotReady):
            await async_setup_entry(hass, entry)
