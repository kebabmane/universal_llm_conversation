"""Integration tests for config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.mark.usefixtures("mock_validate_connection")
async def test_form(hass: HomeAssistant) -> None:
    """Test successful config flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {}

    with patch(
        "custom_components.universal_llm_conversation.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Test LLM",
                "api_key": "test-key",
                "provider": "openai_compatible",
                "base_url": "http://localhost:1234/v1",
                "skip_authentication": False,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Test LLM"
    assert result2["data"]["api_key"] == "test-key"
    assert result2["data"]["provider"] == "openai_compatible"
    mock_setup_entry.assert_called_once()


async def test_form_cannot_connect(hass: HomeAssistant) -> None:
    """Test config flow with connection failure."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "custom_components.universal_llm_conversation.helpers.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=False)
        ),
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Test LLM",
                "api_key": "bad-key",
                "provider": "openai_compatible",
                "base_url": "http://bad-url",
                "skip_authentication": False,
            },
        )

    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {"base": "cannot_connect"}


@pytest.mark.usefixtures("mock_validate_connection")
async def test_conversation_subentry_flow(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test creating a conversation subentry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "My Agent",
            "prompt": "You are a helpful assistant.",
            "chat_model": "kimi-k2.6",
            "max_tokens": 1000,
            "max_function_calls_per_conv": 5,
            "functions": "[]",
            "context_threshold": 40000,
            "context_truncate_strategy": "clear",
            "advanced_options": False,
        },
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "My Agent"
