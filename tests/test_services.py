"""Tests for service registration and handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant, ServiceCall

from custom_components.universal_llm_conversation.const import DOMAIN
from custom_components.universal_llm_conversation.services import async_setup_services


@pytest.mark.usefixtures("mock_validate_connection")
async def test_reload_skills_service_handler(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test reload_skills service triggers SkillManager.async_reload."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.universal_llm_conversation.services.SkillManager.async_get_instance"
    ) as mock_get_instance:
        mock_manager = MagicMock()
        mock_manager.async_reload = AsyncMock()
        mock_get_instance.return_value = mock_manager

        await hass.services.async_call(DOMAIN, "reload_skills", {})
        await hass.async_block_till_done()

    mock_manager.async_reload.assert_awaited_once()


@pytest.mark.usefixtures("mock_validate_connection")
async def test_download_skill_service_handler_success(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test download_skill service triggers SkillManager.async_download_skill."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.universal_llm_conversation.services.SkillManager.async_get_instance"
    ) as mock_get_instance:
        mock_manager = MagicMock()
        mock_manager.async_download_skill = AsyncMock(return_value=True)
        mock_get_instance.return_value = mock_manager

        await hass.services.async_call(DOMAIN, "download_skill", {"skill_name": "test_skill"})
        await hass.async_block_till_done()

    mock_manager.async_download_skill.assert_awaited_once_with("test_skill")


@pytest.mark.usefixtures("mock_validate_connection")
async def test_download_skill_service_handler_failure(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test download_skill service logs error on failure."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    with patch(
        "custom_components.universal_llm_conversation.services.SkillManager.async_get_instance"
    ) as mock_get_instance:
        mock_manager = MagicMock()
        mock_manager.async_download_skill = AsyncMock(return_value=False)
        mock_get_instance.return_value = mock_manager

        await hass.services.async_call(DOMAIN, "download_skill", {"skill_name": "bad_skill"})
        await hass.async_block_till_done()

    mock_manager.async_download_skill.assert_awaited_once_with("bad_skill")


@pytest.mark.usefixtures("mock_validate_connection")
async def test_analyze_image_service_registered(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test analyze_image service is registered after setup."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert hass.services.has_service(DOMAIN, "analyze_image")


@pytest.mark.usefixtures("mock_validate_connection")
async def test_analyze_image_service_success(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test analyze_image service returns analysis text."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent is not None

    with patch.object(
        agent, "async_analyze_images", new=AsyncMock(return_value="There is a cat.")
    ):
        result = await hass.services.async_call(
            DOMAIN,
            "analyze_image",
            {
                "agent_id": mock_config_entry.entry_id,
                "images": ["/config/www/test.jpg"],
                "prompt": "What do you see?",
            },
            blocking=True,
            return_response=True,
        )

    assert result == {"analysis": "There is a cat."}


@pytest.mark.usefixtures("mock_validate_connection")
async def test_analyze_image_service_agent_not_found(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test analyze_image raises when agent_id is invalid."""
    from homeassistant.exceptions import ServiceValidationError

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    with pytest.raises(ServiceValidationError, match="not found"):
        await hass.services.async_call(
            DOMAIN,
            "analyze_image",
            {
                "agent_id": "nonexistent.agent",
                "images": ["/config/www/test.jpg"],
            },
            blocking=True,
            return_response=True,
        )
