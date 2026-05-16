"""Services for Universal LLM Conversation."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv

from .const import DOMAIN, SERVICE_ANALYZE_IMAGE, SERVICE_DOWNLOAD_SKILL, SERVICE_RELOAD_SKILLS
from .entity import UniversalLLMBaseEntity
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)

SERVICE_DOWNLOAD_SKILL_SCHEMA = vol.Schema(
    {
        vol.Required("skill_name"): cv.string,
    }
)

SERVICE_ANALYZE_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("agent_id"): cv.string,
        vol.Required("images"): vol.All(cv.ensure_list, [cv.string], vol.Length(min=1)),
        vol.Optional("prompt", default="Describe this image."): cv.string,
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

    async def handle_analyze_image(call: ServiceCall) -> ServiceResponse:
        """Analyze image(s) with a Universal LLM agent."""
        agent_id = call.data["agent_id"]
        images = call.data["images"]
        prompt = call.data["prompt"]

        agent = conversation.async_get_agent(hass, agent_id)
        if agent is None:
            raise ServiceValidationError(f"Agent {agent_id} not found")
        if not isinstance(agent, UniversalLLMBaseEntity):
            raise ServiceValidationError(
                f"Agent {agent_id} is not a Universal LLM Conversation agent"
            )

        analysis = await agent.async_analyze_images(
            prompt=prompt, image_sources=images
        )

        return {"analysis": analysis}

    hass.services.async_register(
        DOMAIN, SERVICE_RELOAD_SKILLS, handle_reload_skills
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_DOWNLOAD_SKILL,
        handle_download_skill,
        schema=SERVICE_DOWNLOAD_SKILL_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_ANALYZE_IMAGE,
        handle_analyze_image,
        schema=SERVICE_ANALYZE_IMAGE_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
