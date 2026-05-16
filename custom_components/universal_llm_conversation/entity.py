"""Base entity for Universal LLM Conversation with provider-aware streaming."""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify

from .const import (
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_FUNCTION_TOOLS,
    CONF_HIDE_THINKING,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_REQUEST_TIMEOUT,
    CONF_SCHEMA_STRICT,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_SKILLS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TTS_STREAMING_MODE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_HIDE_THINKING,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SCHEMA_STRICT,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TTS_STREAMING_MODE,
    DOMAIN,
    PRESET_TO_PROVIDER,
)

_SENTENCE_BOUNDARY_RE = re.compile(
    r'(?<=[.!?…])(?=\s|$)|(?<=。)(?=.)|(?<=？)(?=.)|(?<=！)(?=.)|\n\n|\n'
)


def _resize_image_if_needed(data: bytes, mime_type: str) -> bytes:
    """Resize image if it exceeds API-friendly dimensions."""
    if not mime_type.startswith("image/"):
        return data  # PDFs pass through unchanged
    try:
        from PIL import Image
    except ImportError:
        return data
    try:
        img = Image.open(io.BytesIO(data))
        max_dim = 1568  # OpenAI's recommended max for vision
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=img.format or "JPEG")
        return buf.getvalue()
    except Exception:
        return data


from .exceptions import FunctionNotFound, TokenLengthExceededError
from .functions import get_function
from .helpers import _get_base_url_from_preset, get_exposed_entities, get_provider, shorten_tool_call_id
from .skills import Skill, SkillManager

if TYPE_CHECKING:
    from . import UniversalLLMConfigEntry

_LOGGER = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 20


def _adjust_schema(schema: dict[str, Any], strict: bool = False) -> None:
    """Adjust schema for provider compatibility."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        if strict:
            schema.setdefault("strict", True)
            schema.setdefault("additionalProperties", False)
        else:
            schema.pop("strict", None)
            schema.pop("additionalProperties", None)
        if "properties" not in schema:
            return
        if "required" not in schema:
            schema["required"] = []
        for prop, prop_info in schema.get("properties", {}).items():
            _adjust_schema(prop_info, strict)
            if prop not in schema["required"]:
                # Make nullable so optional params don't break strict mode
                if isinstance(prop_info, dict) and "type" in prop_info:
                    existing = prop_info["type"]
                    if isinstance(existing, str):
                        prop_info["type"] = [existing, "null"]
                schema["required"].append(prop)
    elif schema.get("type") == "array":
        if "items" in schema:
            _adjust_schema(schema["items"], strict)


def _format_structured_output(schema: vol.Schema, llm_api: llm.APIInstance | None) -> dict[str, Any]:
    result: dict[str, Any] = convert(
        schema,
        custom_serializer=(llm_api.custom_serializer if llm_api else llm.selector_serializer),
    )
    _adjust_schema(result)
    return result


def _convert_content_to_param(
    chat_content: list[conversation.Content],
    shorten_tool_call_id: "Callable[[str], str] | None" = None,
) -> list[dict[str, Any]]:
    """Convert chat log content to provider message format."""
    messages: list[dict[str, Any]] = []
    for content in chat_content:
        if content.role == "system":
            messages.append({"role": "system", "content": content.content})
        elif content.role == "user":
            msg: dict[str, Any] = {"role": "user", "content": content.content}
            attachments = getattr(content, "attachments", None)
            if attachments:
                msg["attachments"] = []
                for att in attachments:
                    data = att.path.read_bytes()
                    data = _resize_image_if_needed(data, att.mime_type)
                    msg["attachments"].append({
                        "mime_type": att.mime_type,
                        "data_base64": base64.b64encode(data).decode(),
                    })
            messages.append(msg)
        elif content.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant"}
            if content.content:
                msg["content"] = content.content
            if content.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": shorten_tool_call_id(tool_call.id) if shorten_tool_call_id else tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args),
                        },
                    }
                    for tool_call in content.tool_calls
                ]
            if msg.get("tool_calls") == []:
                msg.pop("tool_calls", None)
            messages.append(msg)
        elif content.role == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": shorten_tool_call_id(content.tool_call_id) if shorten_tool_call_id else content.tool_call_id,
                    "content": json.dumps(content.tool_result),
                }
            )
    return messages


class UniversalLLMBaseEntity(Entity):
    """Base entity for Universal LLM Conversation."""

    _attr_has_entity_name = True
    _attr_name = None
    skill_manager: SkillManager

    def __init__(
        self, entry: UniversalLLMConfigEntry, subentry: ConfigSubentry
    ) -> None:
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        model = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        fallback = subentry.data.get("fallback_model", "")
        display_model = f"{model} → {fallback}" if fallback else model
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Universal LLM",
            model=display_model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    def _get_provider(self, model_override: str | None = None) -> Any:
        """Build provider instance from config."""
        data = self.entry.data
        options = self.subentry.data
        model = model_override or options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        timeout = options.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT)
        # Resolve base_url from preset when not explicitly stored (e.g. preset providers)
        base_url = data.get("base_url") or _get_base_url_from_preset(data)
        # Resolve provider key from preset when not explicitly stored
        preset_key = data.get("provider_preset", "custom")
        provider_key = data.get("provider") or PRESET_TO_PROVIDER.get(preset_key, "openai_compatible")
        return get_provider(
            hass=self.hass,
            provider_key=provider_key,
            api_key=data.get("api_key", ""),
            base_url=base_url,
            api_version=data.get("api_version"),
            organization=data.get("organization"),
            model=model,
            timeout=float(timeout),
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        function_tools: list[dict[str, Any]],
        exposed_entities: list[dict[str, Any]],
        llm_context: llm.LLMContext | None = None,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
        model_override: str | None = None,
    ) -> dict[str, int]:
        """Generate an answer with streaming and tool execution.

        Returns accumulated token usage across all tool iterations.
        """
        options = self.subentry.data
        provider = self._get_provider(model_override)
        max_function_calls = options.get(
            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        )
        do_shorten_tool_call_id = options.get(
            CONF_SHORTEN_TOOL_CALL_ID,
            DEFAULT_SHORTEN_TOOL_CALL_ID,
        )
        hide_thinking = options.get(CONF_HIDE_THINKING, DEFAULT_HIDE_THINKING)
        strict_schemas = options.get(CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT)

        # Build messages
        messages = _convert_content_to_param(
            chat_log.content,
            shorten_tool_call_id if do_shorten_tool_call_id else None,
        )

        # Validate vision capability if attachments are present
        if any("attachments" in m for m in messages) and not provider.capabilities.supports_vision:
            from homeassistant.exceptions import HomeAssistantError
            raise HomeAssistantError(
                f"Model {provider.model} does not support image or PDF attachments. "
                "Configure a vision-capable model or remove the attachment."
            )

        # Build tools
        tools: list[dict[str, Any]] = []
        for func_spec in function_tools:
            spec = dict(func_spec["spec"])
            _adjust_schema(spec, strict=strict_schemas)
            tools.append({"type": "function", "function": spec})

        # Build API parameters and filter by provider capabilities
        api_params: dict[str, Any] = {}
        for key in (CONF_TEMPERATURE, CONF_TOP_P, CONF_MAX_TOKENS):
            if key in options:
                api_params[key] = options[key]
        api_params = provider.filter_params(api_params)

        # Structured output
        if structure is not None:
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": slugify(structure_name),
                    "strict": strict_schemas,
                    "schema": _format_structured_output(structure, chat_log.llm_api),
                },
            }

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Execute conversation loop
        for n_requests in range(MAX_TOOL_ITERATIONS):
            tool_choice = "auto"
            if tools and 0 <= max_function_calls <= n_requests:
                tool_choice = "none"

            _LOGGER.debug("Prompt for %s: %s", provider.model, json.dumps(messages))

            stream = provider.stream_chat(
                messages=messages,
                tools=tools if provider.supports_tools else None,
                options={**api_params, "tool_choice": tool_choice, "schema_strict": strict_schemas},
            )

            pending_tool_calls: list[llm.ToolInput] = []
            reasoning_parts: list[str] = []
            usage_accumulator: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            async for chunk in chat_log.async_add_delta_content_stream(
                self.entity_id, self._transform_stream(chat_log, stream, hide_thinking, reasoning_parts, usage_accumulator)
            ):
                if isinstance(chunk, conversation.AssistantContent) and chunk.tool_calls:
                    pending_tool_calls.extend(chunk.tool_calls)

            total_usage["prompt_tokens"] += usage_accumulator["prompt_tokens"]
            total_usage["completion_tokens"] += usage_accumulator["completion_tokens"]
            total_usage["total_tokens"] += usage_accumulator["total_tokens"]

            if pending_tool_calls:
                _LOGGER.debug("Tool calls: %s", pending_tool_calls)

            for tool_input in pending_tool_calls:
                function_tool = next(
                    (f for f in function_tools if f["spec"]["name"] == tool_input.tool_name),
                    None,
                )
                if function_tool is None:
                    raise FunctionNotFound(tool_input.tool_name)

                tool_result = await self._execute_function_tool(
                    function_tool, tool_input, llm_context, exposed_entities
                )
                chat_log.async_add_assistant_content_without_tools(tool_result)

            messages = _convert_content_to_param(
                chat_log.content,
                shorten_tool_call_id if do_shorten_tool_call_id else None,
            )

            if not chat_log.unresponded_tool_results:
                break

        return total_usage

    async def _transform_stream(
        self,
        chat_log: conversation.ChatLog,
        stream: Any,
        hide_thinking: bool,
        reasoning_parts: list[str],
        usage_accumulator: dict[str, int],
    ) -> Any:
        """Transform provider stream to Home Assistant format."""
        current_tool_calls: dict[int, dict[str, Any]] = {}
        first_chunk = True
        sentence_mode = (
            self.subentry.data.get(CONF_TTS_STREAMING_MODE, DEFAULT_TTS_STREAMING_MODE)
            == "sentence"
        )
        sentence_buffer = ""

        def _flush_sentence_buffer(final: bool = False) -> list[dict[str, Any]]:
            """Extract complete sentences from buffer, optionally flushing remainder."""
            nonlocal sentence_buffer
            deltas: list[dict[str, Any]] = []
            while True:
                match = _SENTENCE_BOUNDARY_RE.search(sentence_buffer)
                if not match:
                    break
                end = match.end()
                sentence = sentence_buffer[:end]
                sentence_buffer = sentence_buffer[end:]
                if sentence:
                    deltas.append({"content": sentence})
            if final and sentence_buffer:
                deltas.append({"content": sentence_buffer})
                sentence_buffer = ""
            return deltas

        async for chunk in stream:
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if "usage" in chunk:
                usage = chunk["usage"]
                chat_log.async_trace(
                    {
                        "stats": {
                            "input_tokens": usage.get("prompt_tokens"),
                            "output_tokens": usage.get("completion_tokens"),
                        }
                    }
                )
                usage_accumulator["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_accumulator["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_accumulator["total_tokens"] += usage.get("total_tokens", 0)
                total = usage.get("total_tokens", 0)
                if total > self.subentry.data.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD):
                    await self._truncate_message_history(chat_log)
                continue

            if "content" in chunk:
                if sentence_mode:
                    sentence_buffer += chunk["content"]
                    for delta in _flush_sentence_buffer():
                        yield delta
                else:
                    yield {"content": chunk["content"]}

            if "reasoning_content" in chunk:
                if not hide_thinking:
                    if sentence_mode:
                        sentence_buffer += chunk["reasoning_content"]
                        for delta in _flush_sentence_buffer():
                            yield delta
                    else:
                        yield {"content": chunk["reasoning_content"]}
                else:
                    reasoning_parts.append(chunk["reasoning_content"])

            if "tool_calls" in chunk:
                if sentence_mode and sentence_buffer:
                    for delta in _flush_sentence_buffer(final=True):
                        yield delta
                tool_calls_list = []
                for tc in chunk["tool_calls"]:
                    tool_calls_list.append(
                        llm.ToolInput(
                            id=tc["id"],
                            tool_name=tc["tool_name"],
                            tool_args=tc["tool_args"],
                            external=True,
                        )
                    )
                if tool_calls_list:
                    yield {"tool_calls": tool_calls_list}

            if chunk.get("finish_reason") == "length":
                if sentence_mode and sentence_buffer:
                    for delta in _flush_sentence_buffer(final=True):
                        yield delta
                raise TokenLengthExceededError(
                    self.subentry.data.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
                )

            if chunk.get("finish_reason") == "stop":
                if sentence_mode and sentence_buffer:
                    for delta in _flush_sentence_buffer(final=True):
                        yield delta
                break

    async def _execute_function_tool(
        self,
        function_tool: dict[str, Any],
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext | None,
        exposed_entities: list[dict[str, Any]],
    ) -> conversation.ToolResultContent:
        """Execute a custom function."""
        arguments: dict[str, Any] = tool_input.tool_args
        function_config = function_tool["function"]
        function = get_function(function_config["type"])

        if self._should_run_in_background(arguments):
            function_config = self._get_delayed_function_config(function_config, arguments)
            function = get_function(function_config["type"])
            self.entry.async_create_task(
                self.hass,
                function.execute(
                    self.hass, function_config, arguments, llm_context, exposed_entities
                ),
            )
            result = "Scheduled"
        else:
            result = await function.execute(
                self.hass, function_config, arguments, llm_context, exposed_entities
            )

        return conversation.ToolResultContent(
            agent_id=self.entity_id,
            tool_call_id=tool_input.id,
            tool_name=tool_input.tool_name,
            tool_result={"result": str(result)},
        )

    def _should_run_in_background(self, arguments: dict[str, Any]) -> bool:
        return isinstance(arguments, dict) and arguments.get("delay") is not None

    def _get_delayed_function_config(
        self, function_config: dict[str, Any], arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "type": "composite",
            "sequence": [
                {"type": "script", "sequence": [{"delay": arguments["delay"]}]},
                function_config,
            ],
        }

    async def _truncate_message_history(self, chat_log: conversation.ChatLog) -> None:
        """Truncate message history by clearing to last user message."""
        _LOGGER.info("Context threshold exceeded, clearing history")
        messages = chat_log.content
        last_user_index = None
        for i in reversed(range(len(messages))):
            if isinstance(messages[i], conversation.UserContent):
                last_user_index = i
                break
        if last_user_index is not None:
            del messages[1:last_user_index]

    def _get_enabled_skills(self) -> list[Skill]:
        enabled_names = self.subentry.data.get(CONF_SKILLS, []) or []
        all_skills = self.skill_manager.get_all_skills()
        return [s for s in all_skills if s.name in enabled_names]

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        return get_exposed_entities(self.hass)

    def _get_function_tools(self) -> list[dict[str, Any]]:
        """Get custom functions configuration."""
        import yaml

        from .exceptions import FunctionLoadFailed, InvalidFunction, FunctionNotFound

        try:
            function_tools_config = self.subentry.data.get(CONF_FUNCTION_TOOLS)
            function_tools: list[dict[str, Any]] | None = (
                yaml.safe_load(function_tools_config)
                if function_tools_config
                else None
            )
            if not function_tools:
                from .const import DEFAULT_CONF_FUNCTION_TOOLS
                function_tools = DEFAULT_CONF_FUNCTION_TOOLS

            for ft in function_tools:
                if isinstance(ft, dict) and "function" in ft:
                    fcfg = ft["function"]
                    if isinstance(fcfg, dict) and "type" in fcfg:
                        func = get_function(fcfg["type"])
                        ft["function"] = func.validate_schema(fcfg)
            return function_tools or []
        except (InvalidFunction, FunctionNotFound):
            raise
        except Exception as e:
            raise FunctionLoadFailed() from e

    async def async_analyze_images(
        self,
        prompt: str,
        image_sources: list[str],
    ) -> str:
        """Run one-off vision analysis without conversation state or tools."""
        provider = self._get_provider()
        if not provider.capabilities.supports_vision:
            from homeassistant.exceptions import HomeAssistantError

            raise HomeAssistantError(
                f"Model {provider.model} does not support image analysis. "
                "Configure a vision-capable model for this agent."
            )

        # Resolve images to (bytes, mime_type)
        resolved: list[tuple[bytes, str]] = []
        for source in image_sources:
            if source.startswith("media-source://camera/"):
                from homeassistant.components import camera

                entity_id = source.removeprefix("media-source://camera/")
                snapshot = await camera.async_get_image(self.hass, entity_id)
                resolved.append((snapshot.content, snapshot.content_type))
            elif source.startswith("media-source://image/"):
                from homeassistant.components import image as image_comp

                entity_id = source.removeprefix("media-source://image/")
                img = await image_comp.async_get_image(self.hass, entity_id)
                resolved.append((img.content, img.content_type))
            elif source.startswith("media-source://"):
                from homeassistant.components.media_source import async_resolve_media

                media = await async_resolve_media(self.hass, source, None)
                if media.path is None:
                    from homeassistant.exceptions import ServiceValidationError

                    raise ServiceValidationError(
                        f"Cannot resolve media source {source}"
                    )
                data = await self.hass.async_add_executor_job(
                    Path(media.path).read_bytes
                )
                mime = media.mime_type or "application/octet-stream"
                resolved.append((data, mime))
            else:
                # Local file path
                if not self.hass.config.is_allowed_path(source):
                    from homeassistant.exceptions import HomeAssistantError

                    raise HomeAssistantError(f"Path not allowed: {source}")
                path = Path(source)
                if not path.exists():
                    from homeassistant.exceptions import ServiceValidationError

                    raise ServiceValidationError(f"File not found: {source}")
                data = await self.hass.async_add_executor_job(path.read_bytes)
                mime = mimetypes.guess_type(source)[0] or "application/octet-stream"
                resolved.append((data, mime))

        # Build generic attachments
        attachments: list[dict[str, Any]] = []
        for data, mime_type in resolved:
            data = _resize_image_if_needed(data, mime_type)
            attachments.append({
                "mime_type": mime_type,
                "data_base64": base64.b64encode(data).decode(),
            })

        messages = [
            {
                "role": "user",
                "content": prompt,
                "attachments": attachments,
            }
        ]

        # Reuse agent generation params, filtering by provider capabilities
        options: dict[str, Any] = {}
        for key in (CONF_TEMPERATURE, CONF_TOP_P, CONF_MAX_TOKENS):
            if key in self.subentry.data:
                options[key] = self.subentry.data[key]
        options = provider.filter_params(options)

        stream = provider.stream_chat(
            messages=messages,
            tools=None,
            options={**options, "tool_choice": "none", "schema_strict": False},
        )

        response_text = ""
        async for chunk in stream:
            if "content" in chunk:
                response_text += chunk["content"]
            if chunk.get("finish_reason") == "length":
                _LOGGER.warning("Image analysis hit token limit")
                break

        return response_text.strip()
