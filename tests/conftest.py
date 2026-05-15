"""Test fixtures for Universal LLM Conversation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS, UniversalLLMConversationConfigFlow
from custom_components.universal_llm_conversation.const import DOMAIN

# Register config flow handler for tests
import homeassistant.config_entries as ce
ce.HANDLERS.register(DOMAIN)(UniversalLLMConversationConfigFlow)


@pytest.fixture(autouse=True)
async def init_components(hass: HomeAssistant) -> None:
    """Initialize required components."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    # Register our custom component domain for testing
    hass.config.components.add("universal_llm_conversation")
    await hass.async_block_till_done()


@pytest.fixture
async def enable_custom_integrations(
    hass: HomeAssistant,
) -> None:
    """Enable custom integrations for discovery."""
    hass.config.components.add("custom_integrations")
    await hass.async_block_till_done()


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Create a mock config entry with subentries."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test Universal LLM",
        data={
            "api_key": "test-api-key",
            "provider": "openai_compatible",
            "base_url": "http://localhost:1234/v1",
            "api_version": None,
            "organization": None,
            "skip_authentication": False,
        },
        version=1,
        subentries_data=[
            ConfigSubentryData(
                data=dict(DEFAULT_OPTIONS),
                subentry_type="conversation",
                title="Test Conversation",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={"chat_model": "gpt-4o-mini", "max_tokens": 500},
                subentry_type="ai_task_data",
                title="Test AI Task",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_validate_connection() -> Generator[AsyncMock]:
    """Mock provider validation to always succeed."""
    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.validate_connection",
        new_callable=AsyncMock,
        return_value=True,
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_stream() -> Generator[AsyncMock]:
    """Mock provider stream_chat to yield simple assistant response."""

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        yield {"role": "assistant"}
        yield {"content": "Hello from test"}
        yield {"finish_reason": "stop"}

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_stream_with_tool() -> Generator[AsyncMock]:
    """Mock provider stream_chat to yield a tool call."""

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        yield {"role": "assistant"}
        yield {
            "tool_calls": [
                {
                    "id": "call_123",
                    "tool_name": "execute_services",
                    "tool_args": {
                        "list": [
                            {
                                "domain": "light",
                                "service": "turn_on",
                                "service_data": {"entity_id": ["light.test"]},
                            }
                        ]
                    },
                }
            ]
        }
        yield {"finish_reason": "stop"}

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_stream_with_fallback() -> Generator[AsyncMock]:
    """Mock provider stream_chat that fails on first call then succeeds."""
    call_count = 0

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Primary model failed")
        yield {"role": "assistant"}
        yield {"content": "Fallback response"}
        yield {"finish_reason": "stop"}

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


@pytest.fixture
def mock_exposed_entities(hass: HomeAssistant) -> None:
    """Set up fake exposed entities in hass.states."""
    hass.states.async_set("light.test", "off", {"friendly_name": "Test Light"})
    hass.states.async_set("switch.test", "on", {"friendly_name": "Test Switch"})
    hass.states.async_set("sensor.test", "42", {"friendly_name": "Test Sensor"})

    # Mock async_should_expose to return True for all entities
    with patch(
        "homeassistant.components.homeassistant.exposed_entities.async_should_expose",
        return_value=True,
    ):
        yield
