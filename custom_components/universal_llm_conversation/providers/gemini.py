"""Native Google Gemini provider for Universal LLM Conversation."""

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


class GeminiProvider(BaseProvider):
    """Native Google Gemini provider using the google-genai SDK."""

    def __init__(self, hass: HomeAssistant, **kwargs: Any) -> None:
        """Initialize provider and build Gemini client."""
        super().__init__(hass=hass, **kwargs)
        try:
            from google import genai
        except ImportError as err:
            raise HomeAssistantError("Google GenAI SDK not installed") from err

        self._client = genai.Client(api_key=self.api_key)

    async def validate_connection(self) -> bool:
        """Validate by listing models with a short timeout."""
        try:
            from google.genai.errors import APIError, ClientError, ServerError
        except ImportError:
            raise HomeAssistantError("cannot_connect")

        try:
            response = self._client.aio.models.list(timeout=10)
            async for _ in response:
                break
            return True
        except ClientError as err:
            status = getattr(err, "status", None) or getattr(err, "code", None)
            if status in (401, 403):
                _LOGGER.error("Gemini authentication failed: %s", err)
                raise HomeAssistantError("invalid_auth")
            _LOGGER.error("Gemini client error: %s", err)
            raise HomeAssistantError("cannot_connect")
        except ServerError as err:
            _LOGGER.error("Gemini server error: %s", err)
            raise HomeAssistantError("cannot_connect")
        except APIError as err:
            _LOGGER.error("Gemini API error: %s", err)
            raise HomeAssistantError("cannot_connect")
        except Exception as err:
            _LOGGER.error("Gemini validation failed: %s", err)
            raise HomeAssistantError("cannot_connect")

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat completion via native Gemini API."""
        try:
            from google.genai import types
        except ImportError as err:
            raise HomeAssistantError("Google GenAI SDK not installed") from err

        strict_schemas = options.get(CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT)

        # Separate system messages from conversation messages
        system_instruction: str | None = None
        gemini_contents: list[Any] = []
        for msg in messages:
            if msg.get("role") == "system":
                if system_instruction:
                    system_instruction += "\n" + str(msg.get("content", ""))
                else:
                    system_instruction = str(msg.get("content", ""))
            else:
                converted = self._convert_message(msg, types)
                if converted:
                    gemini_contents.append(converted)

        # Build config
        api_params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_tokens"):
            if key in options:
                api_params[key] = options[key]
        api_params = self.filter_params(api_params)

        # Rename max_tokens → max_output_tokens for Gemini config
        config_kwargs: dict[str, Any] = {}
        if "temperature" in api_params:
            config_kwargs["temperature"] = api_params["temperature"]
        if "top_p" in api_params:
            config_kwargs["top_p"] = api_params["top_p"]
        if "maxOutputTokens" in api_params:
            config_kwargs["max_output_tokens"] = api_params["maxOutputTokens"]
        elif "max_tokens" in api_params:
            config_kwargs["max_output_tokens"] = api_params["max_tokens"]

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Convert tools
        gemini_tools = self._convert_tools(tools, types, strict=strict_schemas)
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
            # Map tool_choice to FunctionCallingConfig
            tool_choice = options.get("tool_choice", "auto")
            mode = self._map_tool_choice(tool_choice)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            )

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
        except Exception:
            raise

        first_chunk = True

        async for chunk in stream:
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            # Text content
            text = getattr(chunk, "text", None)
            if text:
                yield {"content": text}

            # Function calls (complete objects, not incremental)
            function_calls = getattr(chunk, "function_calls", None)
            if function_calls:
                tool_calls_list = []
                for fc in function_calls:
                    args = {}
                    fc_args = getattr(fc, "args", None)
                    if fc_args:
                        try:
                            args = dict(fc_args) if hasattr(fc_args, "items") else json.loads(str(fc_args))
                        except Exception:
                            args = {}
                    tool_calls_list.append(
                        {
                            "id": getattr(fc, "id", "") or f"call_{len(tool_calls_list)}",
                            "tool_name": getattr(fc, "name", ""),
                            "tool_args": args,
                        }
                    )
                if tool_calls_list:
                    yield {"tool_calls": tool_calls_list}

            # Usage metadata on final chunks
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(usage_metadata, "total_token_count", 0) or (prompt_tokens + completion_tokens)
                yield {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    }
                }

            # Finish reason
            candidates = getattr(chunk, "candidates", None)
            if candidates:
                for candidate in candidates:
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason:
                        reason_name = str(finish_reason.name) if hasattr(finish_reason, "name") else str(finish_reason)
                        if reason_name.upper() == "MAX_TOKENS":
                            yield {"finish_reason": "length"}
                        elif reason_name.upper() in ("STOP", "OTHER"):
                            yield {"finish_reason": "stop"}

    @staticmethod
    def _convert_message(msg: dict[str, Any], types: Any) -> Any:
        """Convert an OpenAI-style message to Gemini Content format."""
        role = msg.get("role")
        if role == "user":
            return types.Content(
                role="user",
                parts=[types.Part.from_text(text=str(msg.get("content", "")))],
            )
        if role == "assistant":
            parts: list[Any] = []
            content = msg.get("content")
            if content:
                parts.append(types.Part.from_text(text=str(content)))
            # Note: Gemini doesn't use inline tool_calls in assistant messages;
            # function calls are returned as separate model parts in the response
            return types.Content(role="model", parts=parts) if parts else None
        if role == "tool":
            return types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=msg.get("tool_call_id", ""),
                        response={"result": str(msg.get("content", ""))},
                    )
                ],
            )
        return None

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]] | None, types: Any, strict: bool = False
    ) -> list[Any] | None:
        """Convert OpenAI-style tools to Gemini Tool format."""
        if not tools:
            return None
        function_decls: list[Any] = []
        for tool in tools or []:
            func = tool.get("function", {})
            if not func:
                continue
            schema = dict(func.get("parameters", {}))
            if not strict:
                schema.pop("strict", None)
                schema.pop("additionalProperties", None)
            function_decls.append(
                types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters_json_schema=schema,
                )
            )
        return [types.Tool(function_declarations=function_decls)] if function_decls else None

    @staticmethod
    def _map_tool_choice(tool_choice: str) -> str:
        """Map OpenAI-style tool_choice to Gemini FunctionCallingConfig mode."""
        if tool_choice == "none":
            return "NONE"
        if tool_choice in ("auto", "required"):
            return "AUTO"
        return "ANY"