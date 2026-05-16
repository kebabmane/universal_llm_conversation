"""Native Anthropic provider for Universal LLM Conversation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from ..const import CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT
from .base import BaseProvider

_LOGGER = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Native Anthropic Claude provider."""

    def __init__(self, hass: HomeAssistant, **kwargs: Any) -> None:
        """Initialize provider and build Anthropic client."""
        super().__init__(hass=hass, **kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as err:
            raise HomeAssistantError("Anthropic SDK not installed") from err

        self._client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    async def validate_connection(self) -> bool:
        """Validate by listing models with a short timeout."""
        try:
            from anthropic import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
            )
        except ImportError:
            raise HomeAssistantError("cannot_connect")

        try:
            response = self._client.models.list(timeout=10)
            async for _ in response:
                break
            return True
        except AuthenticationError as err:
            _LOGGER.error("Anthropic authentication failed: %s", err)
            raise HomeAssistantError("invalid_auth")
        except APITimeoutError as err:
            _LOGGER.error("Anthropic connection timed out: %s", err)
            raise HomeAssistantError("timeout")
        except APIConnectionError as err:
            _LOGGER.error("Anthropic connection error: %s", err)
            raise HomeAssistantError("cannot_connect")
        except BadRequestError as err:
            _LOGGER.error("Anthropic bad request: %s", err)
            raise HomeAssistantError("cannot_connect")
        except Exception as err:
            _LOGGER.error("Anthropic validation failed: %s", err)
            raise HomeAssistantError("cannot_connect")

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat completion via native Anthropic Messages API."""
        strict_schemas = options.get(CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT)

        # Separate system messages from conversation messages
        system_prompts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompts.append(str(msg.get("content", "")))
            else:
                converted = self._convert_message(msg)
                if converted:
                    anthropic_messages.append(converted)

        system = "\n".join(system_prompts) if system_prompts else None

        # Build API parameters and filter by capabilities
        api_params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_tokens"):
            if key in options:
                api_params[key] = options[key]
        api_params = self.filter_params(api_params)

        # Ensure max_tokens is present (required by Anthropic)
        if "max_tokens" not in api_params:
            api_params["max_tokens"] = 1024

        # Convert tools
        anthropic_tools = self._convert_tools(tools, strict=strict_schemas)

        # Tool choice
        tool_choice = options.get("tool_choice", "auto")
        if anthropic_tools and tool_choice in ("auto", "any", "none"):
            api_params["tool_choice"] = {"type": tool_choice}

        # Remove internal option keys
        api_params.pop("schema_strict", None)

        try:
            stream = await self._client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system,
                tools=anthropic_tools or [],
                stream=True,
                **api_params,
            )
        except Exception:
            raise

        first_chunk = True
        current_tool: dict[str, Any] | None = None
        tool_json_buffer = ""
        reasoning_buffer = ""

        async for event in stream:
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            event_type = getattr(event, "type", None)

            if event_type == "content_block_start":
                block = event.content_block
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    current_tool = {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                    }
                    tool_json_buffer = ""
                elif block_type == "thinking":
                    reasoning_buffer = getattr(block, "thinking", "")
                elif block_type == "text":
                    text = getattr(block, "text", "")
                    if text:
                        yield {"content": text}

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)
                if delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        yield {"content": text}
                elif delta_type == "input_json_delta":
                    partial = getattr(delta, "partial_json", "")
                    if partial:
                        tool_json_buffer += partial
                elif delta_type == "thinking_delta":
                    thinking = getattr(delta, "thinking", "")
                    if thinking:
                        yield {"reasoning_content": thinking}

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    try:
                        args = json.loads(tool_json_buffer) if tool_json_buffer else {}
                    except json.JSONDecodeError:
                        args = {}
                    yield {
                        "tool_calls": [
                            {
                                "id": current_tool["id"],
                                "tool_name": current_tool["name"],
                                "tool_args": args,
                            }
                        ]
                    }
                    current_tool = None
                    tool_json_buffer = ""

            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    yield {
                        "usage": {
                            "prompt_tokens": getattr(usage, "input_tokens", 0),
                            "completion_tokens": getattr(usage, "output_tokens", 0),
                            "total_tokens": getattr(usage, "input_tokens", 0)
                            + getattr(usage, "output_tokens", 0),
                        }
                    }
                stop_reason = getattr(event.delta, "stop_reason", None)
                if stop_reason == "max_tokens":
                    yield {"finish_reason": "length"}
                elif stop_reason:
                    yield {"finish_reason": "stop"}

            elif event_type == "message_stop":
                yield {"finish_reason": "stop"}
                break

    @staticmethod
    def _convert_message(msg: dict[str, Any]) -> dict[str, Any] | None:
        """Convert an OpenAI-style message to Anthropic format."""
        role = msg.get("role")
        if role == "user":
            return {"role": "user", "content": str(msg.get("content", ""))}
        if role == "assistant":
            result: dict[str, Any] = {"role": "assistant"}
            content = msg.get("content")
            if content:
                result["content"] = content
            return result
        if role == "tool":
            # Anthropic uses tool_result content blocks
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": str(msg.get("content", "")),
                    }
                ],
            }
        return None

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]] | None, strict: bool = False
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI-style tools to Anthropic format."""
        if not tools:
            return None
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools or []:
            func = tool.get("function", {})
            if not func:
                continue
            schema = dict(func.get("parameters", {}))
            if not strict:
                schema.pop("strict", None)
                schema.pop("additionalProperties", None)
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": schema,
                }
            )
        return anthropic_tools or None