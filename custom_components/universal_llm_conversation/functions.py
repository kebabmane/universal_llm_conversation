"""Function execution framework for Universal LLM Conversation."""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceNotFound, TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.template import Template
import voluptuous as vol

from .exceptions import FunctionNotFound, InvalidFunction

_LOGGER = logging.getLogger(__name__)


class BaseFunction:
    """Base class for a function implementation."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        """Execute the function and return a result."""
        raise NotImplementedError

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate the function configuration."""
        return config


class ExecuteServiceFunction(BaseFunction):
    """Native function to execute Home Assistant services."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        """Execute HA services from arguments."""
        service_list = arguments.get("list", [])
        for item in service_list:
            domain = item["domain"]
            service = item["service"]
            service_data = item.get("service_data", {})
            await hass.services.async_call(domain, service, service_data, blocking=False)
        return "Executed"

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        return config


class TemplateFunction(BaseFunction):
    """Return a templated string."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        template_str = function_config.get("value_template", "")
        tmpl = Template(template_str, hass)
        return tmpl.async_render(arguments, parse_result=False)

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        if "value_template" not in config:
            raise InvalidFunction("template function requires value_template")
        return config


class ScriptFunction(BaseFunction):
    """Execute a Home Assistant script sequence."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        from homeassistant.helpers.script import Script

        sequence = function_config.get("sequence", [])
        script = Script(hass, sequence, "UniversalLLMFunction", "universal_llm_conversation")
        await script.async_run(variables=arguments, context=None)
        return "Executed"

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        if "sequence" not in config:
            raise InvalidFunction("script function requires sequence")
        return config


class CompositeFunction(BaseFunction):
    """Execute a sequence of functions."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        sequence = function_config.get("sequence", [])
        result = None
        for step in sequence:
            step_type = step.get("type")
            func = get_function(step_type)
            step_config = func.validate_schema(step)
            result = await func.execute(
                hass, step_config, arguments, llm_context, exposed_entities
            )
        return result

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        if "sequence" not in config:
            raise InvalidFunction("composite function requires sequence")
        return config


class BashFunction(BaseFunction):
    """Execute a bash command (restricted)."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        import asyncio
        import shlex
        import subprocess

        from .const import SHELL_DENY_PATTERNS, SHELL_OUTPUT_LIMIT, SHELL_TIMEOUT

        command = function_config.get("command", "")
        if arguments:
            command = Template(command, hass).async_render(arguments, parse_result=False)

        for pattern in SHELL_DENY_PATTERNS:
            if re.search(pattern, command):
                return "Error: Command blocked by security policy"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SHELL_TIMEOUT
            )
            output = (stdout.decode() + "\n" + stderr.decode()).strip()
            return output[:SHELL_OUTPUT_LIMIT]
        except asyncio.TimeoutError:
            proc.kill()
            return "Error: Command timed out"
        except Exception as err:
            return f"Error: {err}"

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        if "command" not in config:
            raise InvalidFunction("bash function requires command")
        return config


class ReadFileFunction(BaseFunction):
    """Read a file from the workspace."""

    async def execute(
        self,
        hass: HomeAssistant,
        function_config: dict[str, Any],
        arguments: dict[str, Any],
        llm_context: Any,
        exposed_entities: list[dict[str, Any]],
    ) -> Any:
        from pathlib import Path

        from .const import FILE_READ_SIZE_LIMIT

        path = function_config.get("path", "")
        if arguments:
            path = Template(path, hass).async_render(arguments, parse_result=False)

        file_path = Path(path)
        if not file_path.is_file():
            return f"Error: File not found: {path}"

        try:
            content = file_path.read_text()[:FILE_READ_SIZE_LIMIT]
            return content
        except Exception as err:
            return f"Error reading file: {err}"

    def validate_schema(self, config: dict[str, Any]) -> dict[str, Any]:
        if "path" not in config:
            raise InvalidFunction("read_file function requires path")
        return config


_FUNCTION_REGISTRY: dict[str, BaseFunction] = {
    "native": ExecuteServiceFunction(),
    "template": TemplateFunction(),
    "script": ScriptFunction(),
    "composite": CompositeFunction(),
    "bash": BashFunction(),
    "read_file": ReadFileFunction(),
}


def get_function(function_type: str) -> BaseFunction:
    """Get a function implementation by type."""
    func = _FUNCTION_REGISTRY.get(function_type)
    if func is None:
        raise FunctionNotFound(f"Function type '{function_type}' not found")
    return func
