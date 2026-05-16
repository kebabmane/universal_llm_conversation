"""Test fixtures for Universal LLM Conversation."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.core import HomeAssistant
from homeassistant.loader import DATA_CUSTOM_COMPONENTS
from homeassistant.setup import async_setup_component

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS
from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.fixture(autouse=True)
async def init_components(hass: HomeAssistant) -> None:
    """Initialize required components and ensure custom component is discoverable."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    # Force HA to rescan custom_components/ so our integration is found
    hass.data.pop(DATA_CUSTOM_COMPONENTS, None)
    await hass.async_block_till_done()


@pytest.fixture(autouse=True)
def mock_dependencies(hass: HomeAssistant) -> None:
    """Mock out manifest dependencies that are heavy or missing in test env."""
    deps = {"ai_task", "energy", "history", "recorder", "rest", "scrape"}
    original = async_setup_component

    async def _mocked_setup(component_hass: HomeAssistant, domain: str, config: Any = None) -> bool:
        if domain in deps:
            return True
        return await original(component_hass, domain, config)

    with patch("homeassistant.setup.async_setup_component", side_effect=_mocked_setup):
        yield


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Create a mock config entry with subentries."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test Universal LLM",
        data={
            "api_key": "test-api-key",
            "provider_preset": "custom",
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
def mock_provider_stream_always_fail() -> Generator[AsyncMock]:
    """Mock provider stream_chat to always raise an error."""

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        raise ConnectionError("Always fails")

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_stream_non_retryable() -> Generator[AsyncMock]:
    """Mock provider stream_chat to raise a non-retryable error."""

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        raise ValueError("Bad request parameter")

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


@pytest.fixture
def mock_provider_stream_with_usage() -> Generator[AsyncMock]:
    """Mock provider stream_chat to yield a response with usage metadata."""

    async def fake_stream(*args: object, **kwargs: object) -> AsyncGenerator[dict[str, object], None]:
        yield {"role": "assistant"}
        yield {"content": "Hello with usage"}
        yield {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        }
        yield {"finish_reason": "stop"}

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.stream_chat",
        side_effect=fake_stream,
    ) as mock:
        yield mock


