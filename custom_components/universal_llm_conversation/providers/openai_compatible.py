"""OpenAI-compatible provider for Universal LLM Conversation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from openai import AsyncAzureOpenAI, AsyncOpenAI

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.httpx_client import get_async_client

from ..const import CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT
from .base import BaseProvider

_LOGGER = logging.getLogger(__name__)

AZURE_DOMAIN_PATTERN = r"\.(openai\.azure\.com|azure-api\.net|services\.ai\.azure\.com)"


def is_azure_url(base_url: str | None) -> bool:
    """Check if base URL is Azure OpenAI."""
    import re

    return bool(base_url and re.search(AZURE_DOMAIN_PATTERN, base_url))


class OpenAICompatibleProvider(BaseProvider):
    """Provider for any OpenAI-compatible endpoint."""

    def __init__(self, hass: HomeAssistant, **kwargs: Any) -> None:
        """Initialize provider and build client."""
        super().__init__(hass=hass, **kwargs)
        self._client: AsyncOpenAI | AsyncAzureOpenAI
        if self.base_url and is_azure_url(self.base_url):
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version or "2024-06-01",
                organization=self.organization,
                http_client=get_async_client(hass),
                timeout=self.timeout,
            )
        else:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                http_client=get_async_client(hass),
                timeout=self.timeout,
            )

    async def validate_connection(self) -> bool:
        """Validate by listing models with a short timeout."""
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            APIStatusError,
        )

        try:
            response = self._client.models.list(timeout=10)
            async for _ in response:
                break
            return True
        except AuthenticationError as err:
            _LOGGER.error("Provider authentication failed: %s", err)
            raise HomeAssistantError("invalid_auth")
        except APITimeoutError as err:
            _LOGGER.error("Provider connection timed out: %s", err)
            raise HomeAssistantError("timeout")
        except APIConnectionError as err:
            _LOGGER.error("Provider connection error: %s", err)
            raise HomeAssistantError("cannot_connect")
        except APIStatusError as err:
            if err.status_code == 403:
                _LOGGER.warning(
                    "Provider returned 403 on /v1/models — API key tier may restrict "
                    "model listing. Proceeding with setup; enter model manually."
                )
                return True
            _LOGGER.error("Provider returned HTTP %s: %s", err.status_code, err.message)
            raise HomeAssistantError("cannot_connect")
        except Exception as err:
            _LOGGER.error("Provider validation failed: %s", err)
            raise HomeAssistantError("cannot_connect")

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream chat completion."""
        strict_schemas = options.get(CONF_SCHEMA_STRICT, DEFAULT_SCHEMA_STRICT)

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Convert attachment messages to OpenAI multimodal format
        for msg in messages:
            if "attachments" in msg:
                content_list: list[dict[str, Any]] = [
                    {"type": "text", "text": msg.pop("content", "") or ""}
                ]
                for att in msg.pop("attachments", []):
                    if att["mime_type"] == "application/pdf":
                        _LOGGER.warning(
                            "PDF attachments are not supported by the OpenAI Chat Completions API "
                            "used by %s. Skipping PDF.",
                            self.model,
                        )
                        continue
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{att['mime_type']};base64,{att['data_base64']}"
                        },
                    })
                msg["content"] = content_list

        # Merge options that have already been filtered by capabilities
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens"):
            if key in options:
                api_kwargs[key] = options[key]

        # Add tools if supported and present
        if tools and self.capabilities.supports_tools:
            api_kwargs["tools"] = tools
            if self.capabilities.supports_tool_choice:
                api_kwargs["tool_choice"] = options.get("tool_choice", "auto")

            # Apply strict mode to schemas
            if not strict_schemas:
                # Remove strict and additionalProperties from tool schemas
                for tool in tools:
                    func = tool.get("function", {})
                    params = func.get("parameters", {})
                    params.pop("strict", None)
                    params.pop("additionalProperties", None)

        # Remove internal option keys before sending to API
        api_kwargs.pop("schema_strict", None)
        tool_choice = api_kwargs.pop("tool_choice", "auto")

        try:
            stream = await self._client.chat.completions.create(
                **api_kwargs,
                tool_choice=tool_choice,
                timeout=self.timeout,
            )
        except Exception:
            raise

        current_tool_calls: dict[int, dict[str, Any]] = {}
        first_chunk = True

        async for chunk in stream:
            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if not chunk.choices:
                if chunk.usage:
                    yield {
                        "usage": {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    }
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                content_value = delta.content
                if not isinstance(content_value, str):
                    content_value = str(content_value) if content_value else ""
                if content_value:
                    yield {"content": content_value}

            # Kimi and some providers may emit reasoning_content
            if getattr(delta, "reasoning_content", None):
                yield {"reasoning_content": delta.reasoning_content}

            if delta.tool_calls:
                for tcd in delta.tool_calls:
                    idx = tcd.index
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tcd.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tcd.function:
                        if tcd.function.name:
                            current_tool_calls[idx]["name"] = tcd.function.name
                        if tcd.function.arguments:
                            current_tool_calls[idx]["arguments"] += tcd.function.arguments

            if current_tool_calls and (choice.finish_reason in {"tool_calls", "stop"}):
                tool_calls_list = []
                for idx in sorted(current_tool_calls.keys()):
                    tc = current_tool_calls[idx]
                    try:
                        args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls_list.append(
                        {
                            "id": tc["id"],
                            "tool_name": tc["name"],
                            "tool_args": args,
                        }
                    )
                if tool_calls_list:
                    yield {"tool_calls": tool_calls_list}
                current_tool_calls.clear()

            if choice.finish_reason == "length":
                yield {"finish_reason": "length"}
                break

            if choice.finish_reason == "stop":
                break
