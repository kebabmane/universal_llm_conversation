"""Pure unit tests for function execution framework."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.universal_llm_conversation.exceptions import (
    FunctionNotFound,
    InvalidFunction,
)
from custom_components.universal_llm_conversation.functions import (
    BashFunction,
    CompositeFunction,
    ExecuteServiceFunction,
    TemplateFunction,
    get_function,
)
from tests.common import get_function_tool_from_yaml


class TestGetFunction:
    """Test function registry."""

    def test_get_native_function(self) -> None:
        func = get_function("native")
        assert isinstance(func, ExecuteServiceFunction)

    def test_get_template_function(self) -> None:
        func = get_function("template")
        assert isinstance(func, TemplateFunction)

    def test_get_unknown_function_raises(self) -> None:
        with pytest.raises(FunctionNotFound):
            get_function("nonexistent")


class TestExecuteServiceFunction:
    """Test native HA service execution."""

    async def test_executes_service(self) -> None:
        hass = MagicMock()
        hass.services.async_call = AsyncMock()
        func = ExecuteServiceFunction()

        result = await func.execute(
            hass=hass,
            function_config={},
            arguments={
                "list": [
                    {
                        "domain": "light",
                        "service": "turn_on",
                        "service_data": {"entity_id": ["light.test"]},
                    }
                ]
            },
            llm_context=None,
            exposed_entities=[],
        )

        assert result == "Executed"
        hass.services.async_call.assert_called_once_with(
            "light", "turn_on", {"entity_id": ["light.test"]}, blocking=False
        )


class TestTemplateFunction:
    """Test template rendering."""

    async def test_renders_template(self) -> None:
        from homeassistant.core import HomeAssistant
        from homeassistant.helpers.template import Template

        hass = MagicMock(spec=HomeAssistant)
        hass.config = MagicMock()
        # Template needs a real template environment
        with patch.object(Template, "async_render", return_value="Hello World"):
            func = TemplateFunction()
            result = await func.execute(
                hass=hass,
                function_config={"value_template": "Hello {{ name }}"},
                arguments={"name": "World"},
                llm_context=None,
                exposed_entities=[],
            )

        assert "Hello World" in result

    def test_validate_schema_requires_value_template(self) -> None:
        func = TemplateFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})


class TestBashFunction:
    """Test bash command execution."""

    async def test_executes_safe_command(self) -> None:
        hass = MagicMock()
        func = BashFunction()

        result = await func.execute(
            hass=hass,
            function_config={"command": "echo 'hello'"},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "hello" in result

    async def test_blocks_dangerous_command(self) -> None:
        hass = MagicMock()
        func = BashFunction()

        result = await func.execute(
            hass=hass,
            function_config={"command": "rm -rf /"},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "blocked" in result.lower() or "Error" in result

    @patch("asyncio.create_subprocess_shell")
    async def test_times_out_long_command(self, mock_subprocess) -> None:
        import asyncio

        hass = MagicMock()
        func = BashFunction()

        # Mock a subprocess that never returns
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_subprocess.return_value = mock_proc

        result = await func.execute(
            hass=hass,
            function_config={"command": "sleep 400"},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "timed out" in result.lower() or "Error" in result


class TestCompositeFunction:
    """Test composite sequence execution."""

    async def test_executes_sequence(self) -> None:
        hass = MagicMock()
        hass.config = MagicMock()
        func = CompositeFunction()

        result = await func.execute(
            hass=hass,
            function_config={
                "sequence": [
                    {"type": "template", "value_template": "Step 1"},
                    {"type": "template", "value_template": "Step 2"},
                ]
            },
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "Step 2" in result

    def test_validate_schema_requires_sequence(self) -> None:
        func = CompositeFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})
