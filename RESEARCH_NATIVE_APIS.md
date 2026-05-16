# Native API Research: Anthropic Claude & Google Gemini

> Research conducted for implementing native providers in the `universal_llm_conversation` Home Assistant custom component.

---

## 1. Anthropic Claude API

### SDK
- **Package:** `anthropic` (PyPI)
- **Latest version:** `~0.102.0` (as of May 2026)
- **Python:** 3.9+
- **HTTP client:** Pure async via `httpx`. The SDK exposes `AsyncAnthropic`.

### Async Client
```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="...")
```

The SDK is **pure async** — no executor wrapping needed. It uses `httpx` internally and supports passing a custom `http_client` (an `httpx.AsyncClient`), which is important for Home Assistant integration.

### Async Streaming
```python
stream = await client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
async for event in stream:
    ...
```

Alternatively, the SDK exposes a dedicated `.stream()` context manager on the sync client, but the async path is `create(..., stream=True)`.

### Message Format
- **Roles:** Only `"user"` and `"assistant"`. There is **no `"system"` role** in the `messages` array.
- **Content:** A string or a list of content blocks (`{"type": "text", "text": "..."}`).

### System Prompts
Passed as a **top-level `system` parameter**, not as a message:
```python
await client.messages.create(
    model="...",
    system="You are a helpful assistant.",  # string or list of text blocks
    messages=[...],
)
```

### Tool Format
Anthropic tools use `input_schema` (not `parameters`):
```python
tools = [{
    "name": "get_weather",
    "description": "Get the current weather",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]
```

### Tool Choice
Supports `tool_choice` with values: `"auto"`, `"any"`, `"none"`, or `{"type": "tool", "name": "..."}`.

### Streaming Tool Calls
In the streaming response, tool calls arrive as a sequence of **events**:
1. `content_block_start` event with `content_block` of type `tool_use` containing `id`, `name`, and empty `input`.
2. `content_block_delta` events with `delta` of type `input_json_delta` containing `partial_json` (string fragments).
3. `content_block_stop` event when the JSON is complete.

You must accumulate `partial_json` strings and `json.loads()` them at `content_block_stop`.

### Capabilities
- **temperature:** `0.0` – `1.0` (default `1.0`).
- **top_p:** Supported (`0.0` – `1.0`).
- **max_tokens:** `max_tokens` (required parameter).
- **thinking / reasoning:** Supported via `thinking` config on newer models; emits `thinking` and `redacted_thinking` blocks.

### Error Types (from `anthropic._exceptions`)
Base class: `AnthropicError`

| Exception | HTTP Status | Meaning |
|-----------|-------------|---------|
| `BadRequestError` | 400 | Invalid request |
| `AuthenticationError` | 401 | Invalid API key |
| `PermissionDeniedError` | 403 | No permission for resource |
| `NotFoundError` | 404 | Resource not found |
| `RequestTooLargeError` | 413 | Payload too large |
| `RateLimitError` | 429 | Rate limited |
| `InternalServerError` | 5xx | Server error |
| `OverloadedError` | 529 | API overloaded |
| `DeadlineExceededError` | 504 | Request timeout |
| `APITimeoutError` | — | Network / connection timeout |
| `APIConnectionError` | — | Connection error |

All 4xx/5xx errors subclass `APIStatusError`, which has `.status_code`, `.request_id`, and `.type`.

---

## 2. Google Gemini API

### SDK
- **Package:** `google-genai` (PyPI) — **the new unified SDK**, NOT the legacy `google-generativeai`.
- **Latest version:** `~1.28.0`+ (as of May 2026)
- **Python:** 3.9+
- **HTTP client:** `httpx` by default; optionally `aiohttp` via `google-genai[aiohttp]`.

### Async Client
```python
from google import genai

client = genai.Client(api_key="...")
aclient = client.aio
```

The async client is accessed via `.aio` and is **pure async**.

### Async Streaming
```python
async for chunk in await aclient.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=[types.Content(role="user", parts=[types.Part.from_text(text="Hello")])],
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.3,
        max_output_tokens=1024,
    ),
):
    print(chunk.text, end="")
```

### Message Format
Gemini uses its own format, not OpenAI-compatible:
- **Role:** `"user"` or `"model"` (assistant is `"model"`).
- **Content:** `types.Content` with a list of `types.Part`.
- **Function calls:** Model emits `types.Part.from_function_call(name=..., args=...)` → role is `"model"`.
- **Function responses:** User sends back `types.Part.from_function_response(name=..., response={...})` inside a `types.Content(role="tool", parts=[...])`.

### System Prompts
Passed via `types.GenerateContentConfig(system_instruction="...")`:
```python
config = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant.",
)
```

### Tool Format
Tools are declared as `types.Tool` with `function_declarations`:
```python
function = types.FunctionDeclaration(
    name="get_weather",
    description="Get the current weather",
    parameters_json_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)
tool = types.Tool(function_declarations=[function])
```

Note: `parameters_json_schema` is used, which is essentially OpenAI's `parameters` schema.

### Function Calling Config
Gemini does **not** support OpenAI-style `tool_choice`. Instead it uses `types.ToolConfig` with `function_calling_config`:
```python
config = types.GenerateContentConfig(
    tools=[tool],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode="AUTO"   # or "ANY", "NONE"
        )
    ),
)
```
- `AUTO` — model decides.
- `ANY` — model must call a tool (similar to OpenAI `required`).
- `NONE` — disable tool calling.

### Streaming Tool Calls
In streaming, each `chunk` is a `GenerateContentResponse`.
- Text: `chunk.text` or `chunk.candidates[0].content.parts[0].text`
- Function calls: `chunk.function_calls` (list of `types.FunctionCall`) or inspect `chunk.candidates[0].content.parts` for parts where `part.function_call` is set.
- Unlike Anthropic, Gemini does **not** stream function call arguments incrementally; they appear as complete `FunctionCall` objects in a chunk.

### Capabilities
- **temperature:** Supported (`0.0` – `2.0` depending on model).
- **top_p:** Supported.
- **max_output_tokens:** `max_output_tokens` in `GenerateContentConfig` (NOT `max_tokens`).

### Error Types (from `google.genai.errors`)
Base class: `APIError`

| Exception | HTTP Range | Meaning |
|-----------|------------|---------|
| `ClientError` | 4xx | Client-side error (bad request, auth, etc.) |
| `ServerError` | 5xx | Server-side error |
| `APIError` | other | General error |

`APIError` exposes `.code`, `.status`, `.message`, and `.details`.

Common status codes from Gemini API:
- `400` — Invalid argument
- `401` — Unauthenticated
- `403` — Permission denied
- `429` — Rate limit
- `500/503` — Internal / unavailable

---

## 3. Comparison Table

| Aspect | OpenAI (reference) | Anthropic | Gemini |
|--------|-------------------|-----------|--------|
| **SDK package** | `openai` | `anthropic` | `google-genai` |
| **Async client** | `AsyncOpenAI(http_client=...)` | `AsyncAnthropic(http_client=...)` | `genai.Client(api_key=...).aio` |
| **Async streaming method** | `await client.chat.completions.create(..., stream=True)` | `await client.messages.create(..., stream=True)` | `await client.aio.models.generate_content_stream(...)` |
| **Message roles** | `system`, `user`, `assistant`, `tool` | `user`, `assistant` only | `user`, `model`, `tool` |
| **System prompt** | `messages[0] = {"role": "system", ...}` | Top-level `system` parameter | `GenerateContentConfig(system_instruction=...)` |
| **Tool format** | `{"type": "function", "function": {"name": ..., "parameters": ...}}` | `{"name": ..., "description": ..., "input_schema": ...}` | `types.Tool(function_declarations=[types.FunctionDeclaration(...)])` |
| **Tool call in stream** | `delta.tool_calls` with incremental `function.arguments` | `content_block_start` (tool_use) → `input_json_delta` (partial_json) → `content_block_stop` | Complete `FunctionCall` objects in `chunk.function_calls` |
| **Tool choice** | `tool_choice="auto" / "required" / "none"` | `tool_choice="auto" / "any" / "none"` | `ToolConfig(function_calling_config=FunctionCallingConfig(mode="AUTO"/"ANY"/"NONE"))` |
| **Temperature support** | `temperature` (0–2) | `temperature` (0–1) | `temperature` (0–2 model-dependent) |
| **Max tokens param name** | `max_tokens` / `max_completion_tokens` | `max_tokens` (required) | `max_output_tokens` |
| **Top_p support** | `top_p` | `top_p` | `top_p` |
| **Streaming usage** | `stream_options={"include_usage": True}` | `usage` included in final `message_stop` / `message_delta` event | `usage_metadata` on final chunk (model-dependent) |
| **Key error types** | `AuthenticationError`, `RateLimitError`, `APIStatusError`, `APITimeoutError`, `APIConnectionError` | `AuthenticationError`, `RateLimitError`, `BadRequestError`, `OverloadedError`, `APITimeoutError`, `APIConnectionError` | `ClientError` (4xx), `ServerError` (5xx), `APIError` |

---

## 4. Minimal Code Examples

### Anthropic — Async Streaming with Tools

```python
import json
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageStopEvent,
    InputJSONDelta,
    ToolUseBlock,
)

client = AsyncAnthropic(api_key="YOUR_API_KEY")

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

tools = [{
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}]

stream = await client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    temperature=0.7,
    stream=True,
)

text_buffer = ""
current_tool = None
tool_json_buffer = ""
tool_calls = []

async for event in stream:
    if event.type == "content_block_start":
        block = event.content_block
        if block.type == "tool_use":
            current_tool = {"id": block.id, "name": block.name, "input": ""}
            tool_json_buffer = ""
        elif block.type == "text":
            text_buffer += block.text
            print(block.text, end="")

    elif event.type == "content_block_delta":
        delta = event.delta
        if delta.type == "text_delta":
            text_buffer += delta.text
            print(delta.text, end="")
        elif delta.type == "input_json_delta":
            tool_json_buffer += delta.partial_json

    elif event.type == "content_block_stop":
        if current_tool is not None:
            current_tool["input"] = json.loads(tool_json_buffer) if tool_json_buffer else {}
            tool_calls.append(current_tool)
            current_tool = None
            tool_json_buffer = ""

    elif event.type == "message_stop":
        # Final metadata available on event.message if needed
        break

if tool_calls:
    print("\n[Tool calls]", tool_calls)
```

### Gemini — Async Streaming with Tools

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="YOUR_API_KEY")
aclient = client.aio

function = types.FunctionDeclaration(
    name="get_weather",
    description="Get weather for a location",
    parameters_json_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]},
)
tool = types.Tool(function_declarations=[function])

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What's the weather in Boston?")]
    )
]

config = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant.",
    temperature=0.7,
    max_output_tokens=1024,
    tools=[tool],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
    ),
)

stream = await aclient.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=contents,
    config=config,
)

text_buffer = ""
tool_calls = []

async for chunk in stream:
    # Text
    if chunk.text:
        text_buffer += chunk.text
        print(chunk.text, end="")

    # Function calls (complete, not streamed incrementally)
    if chunk.function_calls:
        for fc in chunk.function_calls:
            tool_calls.append({
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })

    # Alternative: inspect candidates[0].content.parts
    # for part in chunk.candidates[0].content.parts:
    #     if part.function_call:
    #         tool_calls.append({...})

if tool_calls:
    print("\n[Tool calls]", tool_calls)
```

---

## 5. Special Considerations for Home Assistant

### httpx Client Sharing
- **OpenAI SDK:** Supports passing `http_client=get_async_client(hass)` directly.
- **Anthropic SDK:** Also built on `httpx` and supports passing a custom `http_client` (an `httpx.AsyncClient`). You can share HA's async client.
- **Gemini SDK:** Uses `httpx` by default (or `aiohttp` with extra). It **does not expose a simple `http_client=` parameter** in the same way. You may need to let it create its own client, or dig into `HttpOptions` / internal transport if you want to share HA's instance. In practice, letting `google-genai` manage its own `httpx.AsyncClient` is acceptable for a custom component.

### Dependency Size
- `anthropic` — moderate size; depends on `httpx`, `pydantic`, `typing-extensions`, `anyio`. ~2-3 MB installed.
- `google-genai` — moderate size; depends on `httpx`, `pydantic`, `google-auth`, `requests`. ~3-5 MB installed.
- Both are much lighter than the legacy `google-generativeai` + `google-cloud-aiplatform` stack.

### Timeout Handling
- Anthropic SDK has a default 10-minute timeout for non-streaming and validates that requests won't exceed it. **Always set `timeout=` explicitly** (e.g., `60.0`) to match HA expectations.
- Gemini SDK does not enforce a long default; still recommend passing a sensible timeout via `HttpOptions` or keeping the default.

### Message Format Conversion
Both providers require **converting from OpenAI-style messages** to native format:

| OpenAI | Anthropic | Gemini |
|--------|-----------|--------|
| `{"role": "system", ...}` | Extract into top-level `system` param | `GenerateContentConfig(system_instruction=...)` |
| `{"role": "user", ...}` | Pass as-is | `types.Content(role="user", ...)` |
| `{"role": "assistant", ...}` | Pass as-is | `types.Content(role="model", ...)` |
| `{"role": "tool", ...}` | `{"type": "tool_result", "tool_use_id": ..., "content": ...}` | `types.Content(role="tool", parts=[types.Part.from_function_response(...)])` |
| Tool definitions | `input_schema` instead of `parameters` | `parameters_json_schema` (same shape as OpenAI) |

### Streaming Normalization
Your `BaseProvider.stream_chat` must yield normalized chunks (`{"role": ...}`, `{"content": ...}`, `{"tool_calls": [...]}`). You will need state machines inside each provider to:
1. **Anthropic:** Accumulate `partial_json` across `input_json_delta` events, then emit a single `{"tool_calls": [...]}` chunk when the block stops.
2. **Gemini:** Watch for `chunk.function_calls` (complete objects) and emit them immediately; also collect text deltas.

### Error Mapping
Map each provider's exceptions to `HomeAssistantError` with friendly strings, just like the OpenAI provider does today:
- Auth errors → `HomeAssistantError("invalid_auth")`
- Timeouts → `HomeAssistantError("timeout")`
- Connection errors → `HomeAssistantError("cannot_connect")`
- Rate limits / overloaded → can map to `cannot_connect` or a custom retry logic.

---

## 6. Recommended Dependency Versions for `manifest.json`

```json
{
  "requirements": [
    "openai~=2.21.0",
    "anthropic~=0.102.0",
    "google-genai~=1.28.0"
  ]
}
```

Use `~=` (compatible release) so Home Assistant pip will install the latest patch but not auto-upgrade to a potentially breaking minor version.

If you want to be conservative:
```json
{
  "requirements": [
    "openai>=2.21.0,<3.0",
    "anthropic>=0.102.0,<0.103.0",
    "google-genai>=1.28.0,<2.0"
  ]
}
```

---

## 7. Implementation Checklist for New Providers

### `providers/anthropic.py`
- [ ] Import `AsyncAnthropic`
- [ ] Accept `http_client=get_async_client(hass)` if possible; otherwise omit
- [ ] Map `system` messages to top-level `system` param
- [ ] Convert OpenAI tool schema `parameters` → `input_schema`
- [ ] Accumulate `input_json_delta` / `partial_json` into complete tool calls
- [ ] Handle `thinking` blocks (optional; can be discarded or emitted as `reasoning_content`)
- [ ] Catch `AuthenticationError`, `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `BadRequestError`
- [ ] `validate_connection()` can call `client.models.list()` with a short timeout

### `providers/gemini.py`
- [ ] Import `genai` and `types` from `google.genai`
- [ ] Use `client.aio` for async operations
- [ ] Map `system` messages to `GenerateContentConfig(system_instruction=...)`
- [ ] Convert OpenAI messages to `types.Content` list (user → `user`, assistant → `model`, tool → `tool`)
- [ ] Convert OpenAI tool schema to `types.FunctionDeclaration(parameters_json_schema=...)`
- [ ] Map `tool_choice` to `FunctionCallingConfig(mode=...)`:
  - `auto` → `AUTO`
  - `required` / `any` → `ANY`
  - `none` → `NONE`
- [ ] Watch `chunk.function_calls` for complete tool call objects
- [ ] Catch `APIError` (check `.code` for 401/403/429) and `ClientError`/`ServerError`
- [ ] `validate_connection()` can call `client.aio.models.list()` with a short timeout

### `providers/__init__.py`
- [ ] Update `ANTHROPIC_CAPABILITIES` if needed (already present and mostly correct)
- [ ] Update `GEMINI_CAPABILITIES` if needed (already present; `supports_tool_choice=False` is correct because Gemini uses `function_calling_config`, not OpenAI-style `tool_choice`)

### `helpers.py`
- [ ] Register `