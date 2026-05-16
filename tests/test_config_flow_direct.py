"""Direct tests for config flow logic without full HA integration setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.universal_llm_conversation.config_flow import (
    UniversalLLMConversationConfigFlow,
    _get_base_url_from_preset,
    validate_input,
)
from custom_components.universal_llm_conversation.const import DOMAIN


class TestValidateInput:
    """Test the validate_input helper."""

    async def test_valid_input_passes(self) -> None:
        hass = MagicMock()
        with patch(
            "custom_components.universal_llm_conversation.config_flow.get_provider",
            return_value=MagicMock(
                validate_connection=AsyncMock(return_value=True)
            ),
        ):
            await validate_input(
                hass,
                {
                    "api_key": "test-key",
                    "provider": "openai_compatible",
                    "base_url": "http://localhost:1234/v1",
                },
            )

    async def test_invalid_input_raises(self) -> None:
        from homeassistant.exceptions import HomeAssistantError

        hass = MagicMock()
        with patch(
            "custom_components.universal_llm_conversation.config_flow.get_provider",
            return_value=MagicMock(
                validate_connection=AsyncMock(
                    side_effect=HomeAssistantError("cannot_connect")
                )
            ),
        ), pytest.raises(HomeAssistantError):
            await validate_input(
                hass,
                {
                    "api_key": "bad-key",
                    "provider": "openai_compatible",
                    "base_url": "http://bad-url",
                },
            )

    async def test_skip_authentication_bypasses_validation(self) -> None:
        hass = MagicMock()
        with patch(
            "custom_components.universal_llm_conversation.config_flow.get_provider",
        ) as mock_get_provider:
            await validate_input(
                hass,
                {
                    "api_key": "",
                    "provider": "openai_compatible",
                    "base_url": "http://localhost:1234/v1",
                    "skip_authentication": True,
                },
            )
        mock_get_provider.assert_not_called()


class TestGetBaseUrlFromPreset:
    """Test base URL resolution from preset."""

    def test_manual_override_different_from_preset(self) -> None:
        result = _get_base_url_from_preset({
            "provider_preset": "fireworks",
            "base_url": "https://custom.example.com/v1",
        })
        assert result == "https://custom.example.com/v1"

    def test_uses_preset_default_when_no_manual(self) -> None:
        result = _get_base_url_from_preset({
            "provider_preset": "fireworks",
            "base_url": "",
        })
        assert "fireworks" in result

    def test_returns_none_for_custom_preset(self) -> None:
        result = _get_base_url_from_preset({
            "provider_preset": "custom",
            "base_url": "",
        })
        assert result is None


class TestConfigFlowClass:
    """Test config flow class directly."""

    def test_version_is_1(self) -> None:
        assert UniversalLLMConversationConfigFlow.VERSION == 1

    def test_has_user_step(self) -> None:
        assert hasattr(UniversalLLMConversationConfigFlow, "async_step_user")
