"""Direct tests for config flow logic without full HA integration setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.universal_llm_conversation.config_flow import (
    UniversalLLMConversationConfigFlow,
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
                validate_connection=AsyncMock(return_value=False)
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


class TestConfigFlowClass:
    """Test config flow class directly."""

    def test_version_is_1(self) -> None:
        assert UniversalLLMConversationConfigFlow.VERSION == 1

    def test_has_user_step(self) -> None:
        assert hasattr(UniversalLLMConversationConfigFlow, "async_step_user")
