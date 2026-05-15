"""Tests for skill management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.core import HomeAssistant

from custom_components.universal_llm_conversation.skills import SkillManager


class TestSkillManagerLoading:
    """Test skill loading from disk."""

    async def test_missing_skills_directory(self) -> None:
        hass = MagicMock(spec=HomeAssistant)
        manager = await SkillManager.async_get_instance(hass, "/tmp/nonexistent_skills")
        assert manager.get_all_skills() == []

    async def test_empty_subdirs_skipped(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "empty_dir").mkdir()

        manager = await SkillManager.async_get_instance(hass, str(skills_dir))
        assert manager.get_all_skills() == []

    async def test_description_from_header(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "test_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Test Skill\nversion: 1\n")

        manager = await SkillManager.async_get_instance(hass, str(skills_dir))
        skills = manager.get_all_skills()
        assert len(skills) == 1
        assert skills[0].name == "test_skill"
        assert skills[0].description == "My Test Skill"

    async def test_read_error_skips_skill(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "bad_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("content")

        with patch.object(Path, "read_text", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad bytes")):
            manager = await SkillManager.async_get_instance(hass, str(skills_dir))

        assert manager.get_all_skills() == []

    async def test_get_skill_missing_returns_none(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        manager = await SkillManager.async_get_instance(hass, str(skills_dir))
        assert manager.get_skill("nonexistent") is None

    async def test_reload_clears_old_skills(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "old_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Old\n")

        manager = await SkillManager.async_get_instance(hass, str(skills_dir))
        assert len(manager.get_all_skills()) == 1

        # Remove directory and reload
        import shutil
        shutil.rmtree(skill_dir)
        await manager.async_reload()
        assert manager.get_all_skills() == []


class TestSkillManagerDownload:
    """Test skill download from remote."""

    async def test_download_success(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        manager = SkillManager(hass, str(skills_dir))

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="# Downloaded Skill\n")

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_ctx)

        with patch(
            "custom_components.universal_llm_conversation.skills.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await manager.async_download_skill("cool_skill")

        assert result is True
        downloaded_file = skills_dir / "cool_skill" / "SKILL.md"
        assert downloaded_file.exists()
        assert downloaded_file.read_text() == "# Downloaded Skill\n"

    async def test_download_404(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        manager = SkillManager(hass, str(skills_dir))

        mock_response = MagicMock()
        mock_response.status = 404

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_ctx)

        with patch(
            "custom_components.universal_llm_conversation.skills.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await manager.async_download_skill("missing_skill")

        assert result is False

    async def test_skips_non_directory_entries(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        # Place a file instead of a directory
        (skills_dir / "not_a_skill.txt").write_text("I'm a file")

        manager = await SkillManager.async_get_instance(hass, str(skills_dir))
        assert manager.get_all_skills() == []

    async def test_download_network_error(self, tmp_path: Path) -> None:
        hass = MagicMock(spec=HomeAssistant)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        manager = SkillManager(hass, str(skills_dir))

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=ConnectionError(" refused"))

        with patch(
            "custom_components.universal_llm_conversation.skills.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await manager.async_download_skill("broken_skill")

        assert result is False
