"""Integration tests for config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.mark.usefixtures("mock_validate_connection")
async def test_form(hass: HomeAssistant) -> None:
    """Test successful two-step config flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    # Step 1: provider credentials
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test LLM",
            "api_key": "test-key",
            "provider_preset": "custom",
            "base_url": "http://localhost:1234/v1",
            "skip_authentication": False,
        },
    )
    await hass.async_block_till_done()

    # Should transition to model step
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "model"


async def test_form_cannot_connect(hass: HomeAssistant) -> None:
    """Test config flow with connection failure on step 1."""
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
                "provider_preset": "custom",
                "base_url": "http://bad-url",
                "skip_authentication": False,
            },
        )

    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"] == {"base": "cannot_connect"}


@pytest.mark.usefixtures("mock_validate_connection")
async def test_form_with_model_fetch(hass: HomeAssistant) -> None:
    """Test config flow when provider supports model enumeration."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.universal_llm_conversation.config_flow.async_fetch_models",
        return_value=["accounts/fireworks/models/kimi-k2.6", "accounts/fireworks/models/llama-v3p1-70b"],
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Firepass LLM",
                "api_key": "fp-test-key",
                "provider_preset": "fireworks",
                "skip_authentication": False,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "model"
    # No model_fetch_failed error when fetch succeeds
    assert result2.get("errors") is None or result2.get("errors") == {}


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


@pytest.mark.usefixtures("mock_validate_connection")
async def test_conversation_subentry_advanced_options(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test advanced options step in conversation subentry flow."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    # Submit init with advanced_options=True to branch to advanced step
    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "Advanced Agent",
            "prompt": "You are a helpful assistant.",
            "chat_model": "kimi-k2.6",
            "max_tokens": 1000,
            "max_function_calls_per_conv": 5,
            "functions": "[]",
            "context_threshold": 40000,
            "context_truncate_strategy": "clear",
            "advanced_options": True,
        },
    )
    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "advanced"

    # Submit advanced step
    result3 = await hass.config_entries.subentries.async_configure(
        result2["flow_id"],
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "request_timeout": 120,
            "shorten_tool_call_id": False,
            "schema_strict": False,
            "hide_thinking": True,
            "fallback_model": "gpt-4o-mini",
        },
    )
    assert result3["type"] is FlowResultType.CREATE_ENTRY
    assert result3["title"] == "Advanced Agent"
    assert result3["data"]["temperature"] == 0.7
    assert result3["data"]["fallback_model"] == "gpt-4o-mini"


@pytest.mark.usefixtures("mock_validate_connection")
async def test_ai_task_subentry_flow(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test creating an AI task subentry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "ai_task_data"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result2 = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "name": "My AI Task",
            "chat_model": "gpt-4o-mini",
            "max_tokens": 500,
            "advanced_options": False,
        },
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "My AI Task"


@pytest.mark.usefixtures("mock_validate_connection")
async def test_model_fetch_failure_shows_error(
    hass: HomeAssistant,
) -> None:
    """Test model step shows error when fetch returns empty list."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.universal_llm_conversation.config_flow.async_fetch_models",
        return_value=[],
    ):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Firepass LLM",
                "api_key": "fp-test-key",
                "provider_preset": "fireworks",
                "skip_authentication": False,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] is FlowResultType.FORM
    assert result2["step_id"] == "model"
    assert result2["errors"] == {"base": "model_fetch_failed"}
