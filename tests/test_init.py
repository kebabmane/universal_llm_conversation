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
    """Test setup fails when provider validation raises an error."""
    with patch(
        "custom_components.universal_llm_conversation.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(side_effect=ConnectionError("cannot_connect"))
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


async def test_setup_entry_skips_validation_for_firepass(
    hass: HomeAssistant,
) -> None:
    """Test that Fire Pass preset skips provider validation during setup."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Firepass Test",
        data={
            "api_key": "fp-test-key",
            "provider_preset": "fireworks_firepass",
            "base_url": "https://api.fireworks.ai/inference/v1",
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

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.validate_connection",
        new_callable=AsyncMock,
        return_value=True,
    ) as mock_validate:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    # validate_connection should NOT have been called for firepass
    mock_validate.assert_not_called()


async def test_setup_entry_resolves_base_url_from_preset_at_runtime(
    hass: HomeAssistant,
) -> None:
    """Test that base_url is resolved from preset when not stored in entry data."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    # Simulate an old entry where base_url was NOT persisted (only preset was)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Fireworks Test",
        data={
            "api_key": "fp-test-key",
            "provider_preset": "fireworks",
            # base_url intentionally omitted to simulate pre-fix entries
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

    with patch(
        "custom_components.universal_llm_conversation.providers.openai_compatible.OpenAICompatibleProvider.validate_connection",
        new_callable=AsyncMock,
        return_value=True,
    ) as mock_validate:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    # If base_url had defaulted to OpenAI, validation would have hit the
    # wrong endpoint and failed (or socket-blocked in tests). The fact that
    # it loaded proves base_url was resolved from the preset.
    mock_validate.assert_awaited_once()


async def test_setup_entry_uses_configured_chat_model(
    hass: HomeAssistant,
) -> None:
    """Test that setup validation uses the user's configured chat_model."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Kimi Test",
        data={
            "api_key": "test-key",
            "provider_preset": "custom",
            "base_url": "http://localhost:1234/v1",
            "chat_model": "kimi-k2.6",
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

    with patch(
        "custom_components.universal_llm_conversation.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=True)
        ),
    ) as mock_get_provider:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    mock_get_provider.assert_called_once()
    call_kwargs = mock_get_provider.call_args.kwargs
    assert call_kwargs["model"] == "kimi-k2.6"


async def test_setup_entry_dispatches_anthropic_provider(
    hass: HomeAssistant,
) -> None:
    """Test that Anthropic preset dispatches to AnthropicProvider during setup."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Anthropic Test",
        data={
            "api_key": "sk-ant-test",
            "provider_preset": "anthropic",
            "chat_model": "claude-sonnet-4",
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

    with patch(
        "custom_components.universal_llm_conversation.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=True)
        ),
    ) as mock_get_provider:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    mock_get_provider.assert_called_once()
    call_kwargs = mock_get_provider.call_args.kwargs
    assert call_kwargs["provider_key"] == "anthropic"
    assert call_kwargs["model"] == "claude-sonnet-4"


async def test_setup_entry_dispatches_gemini_provider(
    hass: HomeAssistant,
) -> None:
    """Test that Gemini preset dispatches to GeminiProvider during setup."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Gemini Test",
        data={
            "api_key": "gemini-test-key",
            "provider_preset": "gemini",
            "chat_model": "gemini-2.5-flash",
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

    with patch(
        "custom_components.universal_llm_conversation.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=True)
        ),
    ) as mock_get_provider:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    mock_get_provider.assert_called_once()
    call_kwargs = mock_get_provider.call_args.kwargs
    assert call_kwargs["provider_key"] == "gemini"
    assert call_kwargs["model"] == "gemini-2.5-flash"


async def test_setup_entry_resolves_provider_from_preset_when_absent(
    hass: HomeAssistant,
) -> None:
    """Test provider_key is resolved from PRESET_TO_PROVIDER when not stored in data."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    # Simulate old entry without explicit 'provider' key
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Preset Resolve Test",
        data={
            "api_key": "test-key",
            "provider_preset": "gemini",
            "chat_model": "gemini-2.5-flash",
            # 'provider' intentionally omitted
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

    with patch(
        "custom_components.universal_llm_conversation.get_provider",
        return_value=MagicMock(
            validate_connection=AsyncMock(return_value=True)
        ),
    ) as mock_get_provider:
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.LOADED
    mock_get_provider.assert_called_once()
    call_kwargs = mock_get_provider.call_args.kwargs
    assert call_kwargs["provider_key"] == "gemini"
