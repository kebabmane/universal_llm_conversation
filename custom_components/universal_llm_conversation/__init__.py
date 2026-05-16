"""The Universal LLM Conversation integration."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import CONF_PROVIDER_PRESET, DOMAIN
from .helpers import _get_base_url_from_preset, get_provider
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.CONVERSATION]
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type UniversalLLMConfigEntry = ConfigEntry[None]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    await async_setup_services(hass, config)
    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: UniversalLLMConfigEntry
) -> bool:
    data = entry.data
    provider_key = data.get("provider", "openai_compatible")
    api_key = data.get("api_key", "")
    # Resolve base_url from preset if not stored directly (e.g. preset with known URL)
    base_url = data.get("base_url") or _get_base_url_from_preset(data)
    api_version = data.get("api_version")
    organization = data.get("organization")
    skip_auth = data.get("skip_authentication", False)
    preset_key = data.get(CONF_PROVIDER_PRESET, "custom")

    if not skip_auth and preset_key != "fireworks_firepass":
        try:
            provider = get_provider(
                hass=hass,
                provider_key=provider_key,
                api_key=api_key,
                base_url=base_url,
                api_version=api_version,
                organization=organization,
                model="gpt-4o-mini",
                timeout=10.0,
            )
            await provider.validate_connection()
        except Exception as err:
            _LOGGER.error("Provider validation error: %s", err)
            raise ConfigEntryNotReady(err) from err

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(update_listener))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)
