"""Pure unit tests for function execution framework."""

from __future__ import annotations

import asyncio
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
    ReadFileFunction,
    ScriptFunction,
    TemplateFunction,
    get_function,
)


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

        assert "blocked" in result.lower() or "error" in result.lower()

    @patch("asyncio.create_subprocess_shell")
    async def test_times_out_long_command(self, mock_subprocess) -> None:
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

        assert "timed out" in result.lower() or "error" in result.lower()


class TestBashFunctionExtended:
    """Extended BashFunction edge cases."""

    async def test_template_rendered_then_blocked(self) -> None:
        from homeassistant.helpers.template import Template

        hass = MagicMock()
        func = BashFunction()

        with patch.object(Template, "async_render", return_value="rm -rf /"):
            result = await func.execute(
                hass=hass,
                function_config={"command": "rm -rf {{ file }}"},
                arguments={"file": "/"},
                llm_context=None,
                exposed_entities=[],
            )

        assert "blocked" in result.lower() or "error" in result.lower()

    @pytest.mark.parametrize(
        "bad_command",
        [
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            "shutdown -h now",
            "reboot",
            "format C:",
            "diskpart /clean",
            "del /f /q C:\\\\*",
            "rmdir /s /q C:\\\\Windows",
            ":( ){ :|: };:",
        ],
    )
    async def test_dangerous_patterns_blocked(self, bad_command: str) -> None:
        hass = MagicMock()
        func = BashFunction()

        result = await func.execute(
            hass=hass,
            function_config={"command": bad_command},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "blocked" in result.lower() or "error" in result.lower()

    @patch("asyncio.create_subprocess_shell")
    async def test_output_truncation(self, mock_subprocess) -> None:
        hass = MagicMock()
        func = BashFunction()

        mock_proc = MagicMock()
        huge_output = "x" * 20_000
        mock_proc.communicate = AsyncMock(return_value=(huge_output.encode(), b""))
        mock_subprocess.return_value = mock_proc

        result = await func.execute(
            hass=hass,
            function_config={"command": "cat bigfile"},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert len(result) <= 10_000

    @patch("asyncio.create_subprocess_shell")
    async def test_command_not_found_error(self, mock_subprocess) -> None:
        hass = MagicMock()
        func = BashFunction()

        mock_subprocess.side_effect = OSError("Command not found")

        result = await func.execute(
            hass=hass,
            function_config={"command": "nonexistent_command_12345"},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )

        assert "Error" in result
        assert "not found" in result.lower() or "Command not found" in result


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

    async def test_empty_sequence_returns_none(self) -> None:
        hass = MagicMock()
        func = CompositeFunction()
        result = await func.execute(
            hass=hass,
            function_config={"sequence": []},
            arguments={},
            llm_context=None,
            exposed_entities=[],
        )
        assert result is None

    async def test_function_not_found_in_sequence(self) -> None:
        hass = MagicMock()
        func = CompositeFunction()
        with pytest.raises(FunctionNotFound):
            await func.execute(
                hass=hass,
                function_config={"sequence": [{"type": "nonexistent"}]},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

    async def test_invalid_step_schema_propagates(self) -> None:
        hass = MagicMock()
        func = CompositeFunction()
        with pytest.raises(InvalidFunction):
            await func.execute(
                hass=hass,
                function_config={"sequence": [{"type": "template"}]},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

    async def test_execute_exception_mid_sequence(self) -> None:
        hass = MagicMock()
        func = CompositeFunction()
        with patch(
            "custom_components.universal_llm_conversation.functions.TemplateFunction.execute",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await func.execute(
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


class TestScriptFunction:
    """Test ScriptFunction execution."""

    async def test_executes_script_sequence(self) -> None:
        from homeassistant.helpers.script import Script

        hass = MagicMock()
        func = ScriptFunction()

        with patch.object(Script, "async_run", new_callable=AsyncMock) as mock_run:
            result = await func.execute(
                hass=hass,
                function_config={"sequence": [{"service": "light.turn_on", "target": {"entity_id": "light.test"}}]},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

        assert result == "Executed"
        mock_run.assert_called_once()

    def test_validate_schema_requires_sequence(self) -> None:
        func = ScriptFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})


class TestReadFileFunction:
    """Test ReadFileFunction execution."""

    async def test_reads_existing_file(self) -> None:
        from pathlib import Path

        hass = MagicMock()
        func = ReadFileFunction()

        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="file contents"):
            result = await func.execute(
                hass=hass,
                function_config={"path": "/tmp/test.txt"},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

        assert result == "file contents"

    async def test_file_not_found(self) -> None:
        from pathlib import Path

        hass = MagicMock()
        func = ReadFileFunction()

        with patch.object(Path, "is_file", return_value=False):
            result = await func.execute(
                hass=hass,
                function_config={"path": "/tmp/nonexistent.txt"},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

        assert "not found" in result.lower()

    async def test_read_error(self) -> None:
        from pathlib import Path

        hass = MagicMock()
        func = ReadFileFunction()

        with patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = await func.execute(
                hass=hass,
                function_config={"path": "/root/secret.txt"},
                arguments={},
                llm_context=None,
                exposed_entities=[],
            )

        assert "Error" in result

    def test_validate_schema_requires_path(self) -> None:
        func = ReadFileFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})

    async def test_template_path_rendering(self) -> None:
        from homeassistant.helpers.template import Template
        from pathlib import Path

        hass = MagicMock()
        func = ReadFileFunction()

        with patch.object(Template, "async_render", return_value="/tmp/rendered.txt"), \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="rendered content"):
            result = await func.execute(
                hass=hass,
                function_config={"path": "/tmp/{{ filename }}.txt"},
                arguments={"filename": "rendered"},
                llm_context=None,
                exposed_entities=[],
            )

        assert result == "rendered content"


class TestBaseFunction:
    """Test BaseFunction abstract methods."""

    async def test_execute_raises_not_implemented(self) -> None:
        from custom_components.universal_llm_conversation.functions import BaseFunction
        func = BaseFunction()
        with pytest.raises(NotImplementedError):
            await func.execute(None, {}, {}, None, [])

    def test_validate_schema_returns_config(self) -> None:
        from custom_components.universal_llm_conversation.functions import BaseFunction
        func = BaseFunction()
        assert func.validate_schema({"foo": "bar"}) == {"foo": "bar"}


class TestBashFunctionValidation:
    """Test BashFunction schema validation."""

    def test_validate_schema_requires_command(self) -> None:
        func = BashFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})

    def test_validate_schema_accepts_command(self) -> None:
        func = BashFunction()
        result = func.validate_schema({"command": "echo hello"})
        assert result == {"command": "echo hello"}


class TestScriptFunctionValidation:
    """Test ScriptFunction schema validation."""

    def test_validate_schema_requires_sequence(self) -> None:
        func = ScriptFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})

    def test_validate_schema_accepts_sequence(self) -> None:
        func = ScriptFunction()
        result = func.validate_schema({"sequence": [{"delay": 1}]})
        assert result == {"sequence": [{"delay": 1}]}


class TestCompositeFunctionValidation:
    """Test CompositeFunction schema validation."""

    def test_validate_schema_requires_sequence(self) -> None:
        func = CompositeFunction()
        with pytest.raises(InvalidFunction):
            func.validate_schema({})

    def test_validate_schema_accepts_sequence(self) -> None:
        func = CompositeFunction()
        result = func.validate_schema({"sequence": [{"type": "template", "value_template": "hi"}]})
        assert result == {"sequence": [{"type": "template", "value_template": "hi"}]}
