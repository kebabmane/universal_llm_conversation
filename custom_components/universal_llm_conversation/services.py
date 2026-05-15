"""Services for Universal LLM Conversation."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN, SERVICE_DOWNLOAD_SKILL, SERVICE_RELOAD_SKILLS
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)

SERVICE_DOWNLOAD_SKILL_SCHEMA = vol.Schema(
    {
        vol.Required("skill_name"): cv.string,
    }
)


async def async_setup_services(hass: HomeAssistant, config: Any) -> None:
    """Set up Universal LLM Conversation services."""

    async def handle_reload_skills(call: ServiceCall) -> None:
        """Reload all skills."""
        skill_manager = await SkillManager.async_get_instance(hass)
        await skill_manager.async_reload()
        _LOGGER.info("Skills reloaded")

    async def handle_download_skill(call: ServiceCall) -> None:
        """Download a skill from GitHub."""
        skill_name = call.data["skill_name"]
        skill_manager = await SkillManager.async_get_instance(hass)
        success = await skill_manager.async_download_skill(skill_name)
        if success:
            _LOGGER.info("Skill %s downloaded", skill_name)
        else:
            _LOGGER.error("Failed to download skill %s", skill_name)

    hass.services.async_register(
        DOMAIN, SERVICE_RELOAD_SKILLS, handle_reload_skills
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DOWNLOAD_SKILL,
        handle_download_skill,
        schema=SERVICE_DOWNLOAD_SKILL_SCHEMA,
    )
