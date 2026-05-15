"""Skill management for Universal LLM Conversation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    DEFAULT_SKILLS_DIRECTORY,
    DEFAULT_WORKING_DIRECTORY,
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    GITHUB_SKILLS_BRANCH,
    GITHUB_SKILLS_PATH,
    SKILL_FILE_NAME,
)

_LOGGER = logging.getLogger(__name__)


class Skill:
    """Represents a loaded skill."""

    def __init__(self, name: str, description: str, path: Path) -> None:
        self.name = name
        self.description = description
        self.path = str(path)


class SkillManager:
    """Manages loading and discovery of skills."""

    def __init__(self, hass: HomeAssistant, user_skills_dir: str) -> None:
        self.hass = hass
        self.user_skills_dir = Path(user_skills_dir)
        self._skills: dict[str, Skill] = {}

    @classmethod
    async def async_get_instance(
        cls, hass: HomeAssistant, user_skills_dir: str | None = None
    ) -> SkillManager:
        """Get or create SkillManager instance."""
        if user_skills_dir is None:
            user_skills_dir = str(
                Path(hass.config.config_dir) / DEFAULT_WORKING_DIRECTORY / DEFAULT_SKILLS_DIRECTORY
            )
        manager = cls(hass, user_skills_dir)
        await manager._async_load_skills()
        return manager

    async def _async_load_skills(self) -> None:
        """Load skills from the skills directory."""
        self._skills.clear()
        if not self.user_skills_dir.exists():
            return

        for skill_dir in self.user_skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / SKILL_FILE_NAME
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text()
                # Simple header parsing for name/description
                name = skill_dir.name
                description = ""
                for line in content.splitlines()[:20]:
                    if line.startswith("# "):
                        description = line[2:].strip()
                        break
                self._skills[name] = Skill(name, description, skill_dir)
            except Exception as err:
                _LOGGER.warning("Failed to load skill %s: %s", skill_dir.name, err)

    def get_all_skills(self) -> list[Skill]:
        """Return all loaded skills."""
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill | None:
        """Get a specific skill by name."""
        return self._skills.get(name)

    async def async_download_skill(self, skill_name: str) -> bool:
        """Download a skill from the GitHub repository."""
        session = async_get_clientsession(self.hass)
        url = (
            f"https://raw.githubusercontent.com/"
            f"{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/"
            f"{GITHUB_SKILLS_BRANCH}/{GITHUB_SKILLS_PATH}/{skill_name}/{SKILL_FILE_NAME}"
        )
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    _LOGGER.error("Failed to download skill %s: %s", skill_name, response.status)
                    return False
                content = await response.text()
                target_dir = self.user_skills_dir / skill_name
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / SKILL_FILE_NAME).write_text(content)
                await self._async_load_skills()
                return True
        except Exception as err:
            _LOGGER.error("Error downloading skill %s: %s", skill_name, err)
            return False

    async def async_reload(self) -> None:
        """Reload all skills."""
        await self._async_load_skills()
