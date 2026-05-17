# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.21] - 2026-05-17

### Fixed
- **Missing `kimi-k2p6` model capability override** — Fireworks model ID `accounts/fireworks/routers/kimi-k2p6-turbo` was not recognized as vision-capable because the existing override key was `"kimi-k2.6"` (with dot) while the model string contains `"kimi-k2p6"` (with "p"). Added a dedicated `MODEL_CAPABILITY_OVERRIDES` entry for `"kimi-k2p6"` with identical capabilities to `"kimi-k2.6"` (vision, tools, reasoning, `max_completion_tokens`). Also updated `_resolve_capabilities()` to check normalized model strings (stripping dots and dashes) so both dotted and undotted variants are caught
- **`analyze_image` service hardcoded low `max_tokens`** — Vision analysis tasks need significantly more tokens than conversational tasks (default agent config was 500). Added `max_tokens` as an optional service parameter (default 2000, range 1-8192) that overrides the agent's configured `CONF_MAX_TOKENS` when calling `async_analyze_images()`
- **`async_analyze_images()` returned empty string for reasoning-only models** — Fireworks/Kimi K2.6T streams image analysis in `delta.reasoning_content` chunks instead of `delta.content`. The existing method only accumulated `chunk["content"]`, building an empty string. Now collects both `content` and `reasoning_content` chunks unconditionally

### Testing
- **297 tests, 97% coverage** — Added 5 new tests for the three fixes:
  - `test_resolve_capabilities_kimi_k2p6` — verifies `"accounts/fireworks/routers/kimi-k2p6-turbo"` resolves to vision-capable capabilities
  - `test_resolve_capabilities_kimi_k2_6_dot_variant` — verifies dotted variant still matches
  - `test_kimi_k2p6_has_correct_capabilities` — verifies override entry capabilities
  - `test_analyze_image_service_with_custom_max_tokens` — verifies `max_tokens=4000` override is passed through
  - `test_async_analyze_images_reasoning_content` — verifies `reasoning_content` chunks are collected when no `content` is present

## [0.1.20] - 2026-05-17

### Added
- **`analyze_image` service** — One-off vision analysis without conversation state or tools. Supports camera snapshots (`media-source://camera/`), image entities (`media-source://image/`), generic media sources, and local file paths. Vision capability is validated before sending; non-vision models raise a clear `HomeAssistantError`

### Testing
- **292 tests, 97% coverage** — Direct unit tests added for `async_analyze_images`:
  - `test_async_analyze_images_happy_path` — local file path, streaming response, verifies `tool_choice=none` and `schema_strict=False`
  - `test_async_analyze_images_vision_disabled` — raises `HomeAssistantError` when `supports_vision=False`
  - `test_async_analyze_images_token_limit_warning` — warns and stops on `finish_reason==length`
  - `test_async_analyze_images_path_not_allowed` — raises `HomeAssistantError` for disallowed paths
  - `test_async_analyze_images_file_not_found` — raises `ServiceValidationError` for missing files
  - `test_async_analyze_images_camera_source` — resolves `media-source://camera/` URIs
  - `test_async_analyze_images_image_source` — resolves `media-source://image/` URIs
  - `test_async_analyze_images_generic_media_source` — resolves generic `media-source://` URIs via `async_resolve_media`

## [0.1.19] - 2026-05-16

### Added
- **Image & PDF attachment support** — Chat conversations can now include image and PDF attachments via `UserContent.attachments`:
  - `supports_vision` capability on `ProviderCapabilities` — vision-enabled models (GPT-4o, Claude, Gemini, Kimi K2.6) are flagged; non-vision models raise a clear `HomeAssistantError` when attachments are present
  - Automatic image resizing via Pillow (max 1568px) before base64 encoding to stay within API limits
  - Provider-specific multimodal formatting:
    - **OpenAI-compatible** — `image_url` data URIs for images; PDFs skipped with warning (Chat Completions API does not support inline PDFs)
    - **Anthropic** — `image` and `document` content blocks with base64 source
    - **Gemini** — `Part.from_bytes()` for both images and PDFs
  - `Pillow>=10.0` added to `manifest.json` requirements

### Testing
- **284 tests, 97.32% coverage** — 22 new tests for attachment support:
  - `test_convert_content_to_param_with_image` / `with_pdf` — base64 encoding in chat log conversion
  - `test_resize_image_if_needed_resizes_oversized` / `passthrough_non_image` / `passthrough_no_pillow` — image downsampling
  - `test_vision_disabled_raises` — conversation blocked when model lacks vision
  - Provider-specific attachment conversion tests in `test_openai_compatible_provider.py`, `test_anthropic_provider.py`, and `test_gemini_provider.py`

## [0.1.18] - 2026-05-16

### Fixed
- **`EVENT_CONVERSATION_FINISHED` not fired on dual failure** — Moved event firing before the early error return so the event is always emitted (even when both primary and fallback models fail), ensuring analytics consumers receive `outcome="failed"` and `error_type`
- **Async generator mock bug in test fixtures** — `mock_provider_stream_always_fail` and `mock_provider_stream_non_retryable` in `conftest.py` were bare `async def` functions with no `yield`, causing Python to return a coroutine object instead of an async generator. Added unreachable `yield {}` to force proper async generator behavior so retry/fallback logic is actually tested

### Testing
- **259 tests, 97.21% coverage** — Phase 2–4 coverage push completed:
  - `test_non_conversation_subentry_skipped` — verifies non-conversation subentries do not create conversation entities
  - `test_fallback_event_payload_on_dual_failure` — verifies event payload when both primary and fallback fail
  - `test_relative_working_directory` — skill directory resolution for relative `DEFAULT_WORKING_DIRECTORY`
  - `test_last_content_not_assistant` — empty speech when chat log ends without AssistantContent
  - `test_adjust_schema_non_dict` / `test_adjust_schema_no_properties` — edge cases in schema adjustment
  - `test_convert_content_to_param_empty_tool_calls_popped` — empty `tool_calls` list is removed from assistant messages
  - `test_transform_stream_reasoning_content_hidden` / `visible` — covers `hide_thinking=True` (collects to `reasoning_parts`) and `hide_thinking=False` (yields as content in token mode)
  - `test_structured_output_formatting` — `_async_handle_chat_log` passes `response_format` when a `vol.Schema` structure is provided
  - `test_function_not_found_raises` — `FunctionNotFound` raised when stream yields an unknown tool call
  - `test_fetch_skips_empty_model_id` — model fetch filters out models with empty `id`

## [0.1.17] - 2026-05-16

### Added
- **Native Anthropic (Claude) provider** — Direct Anthropic API integration with streaming, tool calls, thinking content, and usage metadata:
  - `providers/anthropic.py` — `AnthropicProvider` with `AsyncAnthropic` client
  - Thinking/reasoning content emitted as `reasoning_content` chunks (filtered by `hide_thinking` before TTS)
  - Tool call state machine handles `content_block_start` / `content_block_delta` / `content_block_stop` events
  - `validate_connection()` maps 401→`invalid_auth`, timeout→`timeout`, everything else→`cannot_connect`
- **Native Google Gemini provider** — Direct `google-genai` SDK integration with streaming and function calling:
  - `providers/gemini.py` — `GeminiProvider` with `genai.Client`
  - `generate_content_stream()` with `GenerateContentConfig` for system instructions and tool config
  - `FunctionCallingConfig(mode=ANY)` for tool choice mapping (`auto`→`AUTO`, `none`→`NONE`, anything else→`ANY`)
  - `validate_connection()` maps 401→`invalid_auth`, 5xx→`cannot_connect`
- **Provider factory dispatch** — `helpers.get_provider()` now routes to all three providers based on `provider_key` (`openai_compatible`, `anthropic`, `gemini`)
- **Capability-based provider presets** — `PRESET_TO_PROVIDER` maps UI presets to internal provider keys; Anthropic and Gemini presets now instantiate native providers instead of OpenAI-compatible proxies

### Changed
- Setup validation (`__init__.py`) now uses the user's configured `chat_model` instead of a hardcoded dummy model when calling `provider.validate_connection()`
- `manifest.json` requirements updated to include `anthropic>=0.102.0,<0.103.0` and `google-genai>=1.28.0,<2.0`

### Testing
- **226 tests, 93% coverage** — Full suite passes with no failures:
  - 8 new Anthropic provider tests (init, validation, streaming, message/tool conversion)
  - 11 new Gemini provider tests (init, validation, streaming, message/tool conversion, tool choice mapping)
  - 4 new `test_init.py` tests verifying setup dispatches the correct provider and uses configured `chat_model`
  - 3 new `test_helpers.py` tests for factory dispatch (`openai_compatible`, `anthropic`, `gemini`)

## [0.1.16] - 2026-05-16

### Added
- **Sentence-level TTS streaming** — New `tts_streaming_mode` option with two modes:
  - **`sentence` (default)** — Buffers LLM output to sentence boundaries (`. `, `? `, `! `, `…`, `。`, `？`, `！`, newline) before yielding to TTS. Gives the TTS engine full sentence context for natural prosody and intonation. Best for voice-first interactions.
  - **`token`** — Yields raw tokens/words as they arrive from the LLM. Lowest latency to first audio byte. Good for chat UI or impatient users.
- **Tool-call sentence flush** — When a tool call interrupts the assistant mid-sentence, the buffer is flushed first so TTS doesn't speak a half-sentence before the tool result
- **Unicode sentence boundaries** — Chinese/Japanese sentence terminators `。`, `？`, `！` are detected without requiring whitespace after them

### Changed
- `_transform_stream()` now supports `sentence_mode` via `_flush_sentence_buffer()` helper
- `DEFAULT_TTS_STREAMING_MODE` set to `"sentence"` for best voice quality out of the box

### Testing
- **183 tests, 96% coverage** — Added 6 new tests:
  - `test_transform_stream_sentence_mode_single` — single sentence yields immediately
  - `test_transform_stream_sentence_mode_multi` — multi-sentence chunk splits correctly
  - `test_transform_stream_sentence_mode_no_boundary` — unterminated text buffered until stream end
  - `test_transform_stream_sentence_mode_tool_call_flush` — tool calls force buffer flush
  - `test_transform_stream_sentence_mode_unicode` — CJK punctuation boundaries
  - `test_transform_stream_token_mode_unchanged` — token mode preserves raw chunk behavior

## [0.1.5] - 2026-05-15

### Fixed
- **Template rendering crash** — `get_exposed_entities` in `_build_system_prompt` was passed as a bare function reference instead of a lambda, causing `TypeError: missing 1 required positional argument: 'hass'` when the prompt template referenced `{{ universal_llm.exposed_entities }}`
- **Tool call ID shortening crash** — Local variable `shorten_tool_call_id` (boolean option) shadowed the imported `shorten_tool_call_id` helper function, causing `TypeError: 'bool' object is not callable` when the option was enabled

### Testing
- **153 tests, 94% coverage** — Full test suite from 42 tests at ~40% coverage
- Integration tests pass against HA 2026.5.1 runtime (13 integration + 140 unit tests)
- All provider streaming paths tested (content, reasoning, usage, tool calls, finish reasons)
- All function types tested (native, template, script, bash, composite, read_file)
- Config flow two-step navigation tested (provider + model selection, advanced options, AI task subentries)
- Skill load/download/reload lifecycle fully covered

## [0.1.1] - 2026-05-15

### Added
- **Provider preset dropdown** — Select from Fireworks AI, OpenAI, Ollama, OpenRouter, Azure OpenAI, Anthropic, Gemini, or Custom
- **Automatic base URL pre-fill** — Fireworks, OpenAI, Ollama, OpenRouter base URLs are pre-filled when selected
- **Model enumeration** — After provider selection, the component fetches available chat/completion models from the provider's `/v1/models` endpoint and presents them in a dropdown
- **Model filtering** — Non-chat models (embeddings, image, audio, TTS) are automatically excluded from the dropdown
- **Manual fallback** — If model fetch fails, users can still enter a model name manually
- **Fallback model in setup** — Configure a fallback model directly during initial setup

### Changed
- Config flow now has two steps: (1) Provider + API credentials, (2) Model selection
- `chat_model` and `fallback_model` are set during config entry creation and inherited by conversation subentries

## [0.1.0] - 2026-05-15

### Added
- Initial release of Universal LLM Conversation
- **Provider-agnostic architecture** — `BaseProvider` abstraction with capability-based parameter filtering
- **OpenAI Compatible provider** — Works with Fireworks, Firepass, Ollama, OpenRouter, LiteLLM proxy, Groq, vLLM, LocalAI, Azure OpenAI, and any OpenAI-compatible endpoint
- **Kimi K2.6 support** — Per-model capability overrides for `kimi-k2.6` and `kimi-k2.5`:
  - Sends `max_completion_tokens` instead of deprecated `max_tokens`
  - Omits `temperature` and `top_p` (which Kimi rejects)
  - Strips `reasoning_content` before it reaches TTS
- **Response sanitization** — `sanitize_for_speech()` removes leaked tool-call syntax (`execute_services(...)`) and thinking blocks from voice responses
- **Request timeouts** — Default 60s instead of OpenAI SDK's 600s hang time; configurable 10–300s per agent
- **Fallback model** — Configure a backup model; on any failure the agent rewinds the chat log and retries once before giving up
- **Schema strictness toggle** — Default `strict=False` for broad non-OpenAI compatibility; enable for providers that support it
- **Custom functions** — Template, script, bash, composite, and native HA service execution
- **Skills system** — Download and enable reusable AI capabilities from the repository
- **HACS compatible** — Can be installed via Home Assistant Community Store

### Testing
- 42 tests covering core logic: helpers (sanitization, capabilities), providers (parameter filtering, model overrides), functions (execute, template, bash, composite), config flow, and init logic
- Pure unit tests run without a full Home Assistant instance
- Full HA integration tests require the Home Assistant devcontainer or a running HA instance

### Known Limitations
- Native Anthropic (Claude) and Google Gemini providers are not yet implemented; use the OpenAI Compatible provider with their proxy endpoints (e.g., LiteLLM, OpenRouter) as a workaround
- `ai_task_data` subentry type is defined but not fully wired into the AI Task platform yet

## [0.1.6] - 2026-05-16

### Changed
- **3-step config flow** — Replaced two-step setup with a clearer three-step wizard:
  1. Select provider preset (e.g. Fireworks AI, OpenAI, Ollama)
  2. Enter API credentials (API key shown; `base_url`, `api_version`, and `organization` only appear for Custom/Azure providers)
  3. Choose chat model from fetched dropdown or enter manually
- **Conditional credential fields** — `base_url` is no longer shown when a preset already knows its URL (e.g. Fireworks, OpenRouter, Ollama), reducing confusion

## [0.1.7] - 2026-05-16

### Changed
- **3-step config flow implemented** — Actually restructured `config_flow.py` into a true three-step wizard:
  1. Select provider preset (e.g. Fireworks AI, OpenAI, Ollama)
  2. Enter API credentials (API key shown; `base_url`, `api_version`, and `organization` only appear for Custom/Azure providers)
  3. Choose chat model from fetched dropdown or enter manually
- **Conditional credential fields** — `base_url` is no longer shown when a preset already knows its URL (e.g. Fireworks, OpenRouter, Ollama), reducing confusion
- **Updated tests** — Config flow integration tests rewritten for 3-step navigation

## [0.1.8] - 2026-05-16

### Fixed
- **Live validation connection errors** — Removed `async_add_executor_job` wrapper from `validate_connection()` and `async_fetch_models()` that caused thread-pool/HTTP client conflicts
- **Specific error messages during setup** — Instead of a generic "cannot_connect", the config flow now shows:
  - `invalid_auth` — when API key is rejected
  - `timeout` — when provider is unreachable
  - `cannot_connect` — for general connection or HTTP errors
- **Proper exception propagation** — `validate_connection()` now raises `HomeAssistantError` with machine-readable keys instead of returning `False`, so the real failure reason surfaces in the UI
- **Model fetch error logging** — Failed model list fetches now log at ERROR level instead of silently returning empty lists

### Testing
- **156 tests, 93% coverage** — Added integration tests for `invalid_auth` and `timeout` error paths in config flow
- Updated provider unit tests to match new exception-based validation behavior

## [0.1.9] - 2026-05-16

### Fixed
- **Fire Pass / restricted API key tiers** — `validate_connection()` now treats HTTP 403 on `/v1/models` as "key valid but endpoint restricted" instead of a fatal error. Setup proceeds and the user enters a model manually
- **`model_list_restricted` error message** — When the provider's API tier blocks model enumeration, the UI shows: *"This API key tier does not support automatic model listing. Enter a model name manually."*
- **`async_fetch_models()` raises on 403** — Detects 403/forbidden responses and raises `HomeAssistantError("model_list_restricted")` so the config flow can show the appropriate message

### Testing
- **158 tests** — Added tests for 403 handling in `validate_connection()`, `async_fetch_models()`, and config flow model step

## [0.1.10] - 2026-05-16

### Added
- **Fireworks AI — Fire Pass preset** — Dedicated preset for Fire Pass ($49/month early access) subscribers:
  - **Single model**: `accounts/fireworks/routers/kimi-k2p6-turbo` shown as the only option with label *"Kimi 2.6 included with Fire Pass"*
  - **No live validation** — skips the `validate_connection()` call that would 403 on `/v1/models`
  - **No fallback model field** — Fire Pass only covers one model
  - **No red error banner** — clean setup flow without the `model_list_restricted` warning

### Testing
- **160 tests** — Added integration test for Fire Pass preset flow (skip validation, single model selection, subentry creation)

## [0.1.11] - 2026-05-16

### Fixed
- **Runtime `base_url` resolution for preset providers** — Config entries created with provider presets (e.g. Fireworks, OpenRouter, Ollama) previously stored `base_url` as `None` because the preset's known URL was only used during config flow but never persisted. At runtime, `async_setup_entry()` fell back to OpenAI's default endpoint (`https://api.openai.com/v1`), causing `401 Unauthorized` for non-OpenAI keys:
  - `config_flow.py` `async_step_model()` now resolves and injects `base_url` into `user_data` before `async_create_entry()`
  - `helpers.py` `_get_base_url_from_preset()` is now shared between config flow and runtime setup
  - `__init__.py` `async_setup_entry()` resolves `base_url` from preset at runtime as a fallback, ensuring old entries without persisted `base_url` still work
- **Fire Pass runtime validation skip** — `async_setup_entry()` now explicitly skips `validate_connection()` for the `fireworks_firepass` preset, preventing the 403 error that occurs when the restricted key hits `/v1/models`

### Testing
- **163 tests, 94% coverage** — Added tests for `base_url` persistence in config flow entry data and Fire Pass runtime validation skip

## [0.1.12] - 2026-05-16

### Fixed
- **Critical: Conversation runtime defaulted to OpenAI endpoint for ALL providers** — `entity.py` `_get_provider()` was passing `base_url=None` to the OpenAI SDK whenever the config entry did not explicitly store a `base_url` (which is the case for all preset providers: Fireworks, OpenRouter, Ollama, OpenAI, etc.). The OpenAI SDK then defaulted to `https://api.openai.com/v1`, causing `401 Unauthorized` for every non-OpenAI API key at conversation time. Setup validation appeared to work because `__init__.py` had the fix, but the actual chat request was always hitting OpenAI:
  - `entity.py` `_get_provider()` now calls `_get_base_url_from_preset()` as a fallback when `base_url` is not in entry data
  - This ensures Fireworks, Fire Pass, Ollama, OpenRouter, and all other preset providers hit their correct endpoints during conversation
  - Old entries without persisted `base_url` are automatically fixed at runtime

### Testing
- **165 tests, 94% coverage** — Added test for `_get_provider()` base_url resolution from preset at conversation runtime

## [0.1.13] - 2026-05-16

### Fixed
- **`provider` key mismatch — runtime code always defaulted to `openai_compatible`** — The config entry stores `provider_preset` (e.g. `"fireworks"`, `"anthropic"`), but `__init__.py`, `entity.py`, and `config_flow.py` were all reading `data.get("provider", "openai_compatible")`. Since `provider` was never stored in entry data, it always fell back to `"openai_compatible"`. This is currently harmless (all presets are OpenAI-compatible), but would silently break when native Anthropic/Gemini providers are added:
  - Added `PRESET_TO_PROVIDER` mapping in `const.py` that maps each preset to its internal provider key
  - `config_flow.py` `async_step_model()` now persists the resolved `provider` key in entry data
  - `__init__.py`, `entity.py`, and `config_flow.py` `validate_input()` now resolve `provider` from `provider_preset` via `PRESET_TO_PROVIDER` as a fallback
- **Code style consistency** — `validate_input()` in `config_flow.py` now uses the same `data.get(CONF_BASE_URL) or _get_base_url_from_preset(data)` pattern as `__init__.py` and `entity.py`

### Testing
- **166 tests, 94% coverage** — Added test for `_get_provider()` provider key resolution from preset at runtime

## [0.1.14] - 2026-05-16

### Removed
- **AI Task subentries** — Complete ghost feature removed. The UI could create "AI Task" subentries but `PLATFORMS = [CONVERSATION]` meant they were silently skipped with no runtime entity. Removed `UniversalLLMAITaskSubentryFlowHandler`, `DEFAULT_AI_TASK_NAME`, `DEFAULT_AI_TASK_OPTIONS`, and the `ai_task_data` subentry registration (~100 lines of dead code)
- **Dead constants** — `EVENT_AUTOMATION_REGISTERED` (never fired), `SERVICE_QUERY_IMAGE` + `CONF_PAYLOAD_TEMPLATE` (no service exists), `DEFAULT_ALLOWED_DIRS` (never referenced), duplicate `CONF_CHAT_MODEL` definition
- **Dead exception** — `ParseArgumentsFailed` (defined and imported, never raised or caught)
- **Dead code** — `_get_enabled_skills` in `entity.py` (shadowed by identical method in `conversation.py`), `tests/common.py` (imported but never used), `mock_exposed_entities` fixture (never consumed)
- **Unused imports** across `conversation.py`, `entity.py`, `__init__.py`, `helpers.py`, `functions.py`, `providers/openai_compatible.py`

### Changed
- **Context truncation strategy** — Removed the dropdown offering only "Clear All Messages". The strategy is now hard-coded to "clear" until a second strategy is actually implemented. Removes UI noise and schema complexity
- **advanced_options no longer persisted** — The toggle was stored in subentry data but never read by runtime. Now used only for flow control and discarded before entry creation
- **Tightened fallback error catch** — Removed blanket `Exception` from `_FALLBACK_ELIGIBLE_ERRORS`. Now only `TimeoutError` and `ConnectionError` trigger fallback retries, preventing real bugs from being masked
- **Code style consistency** — `validate_input()` now uses the same `data.get(CONF_BASE_URL) or _get_base_url_from_preset(data)` pattern as other files

### Testing
- **172 tests, 97% coverage** — Added tests for structured output formatting, token-length exceeded error, background execution scheduling, and provider validation error branches (401→invalid_auth, timeout, 500→cannot_connect)
- **Provider test coverage: 100%** — All validation error paths now tested

## [0.1.15] - 2026-05-16

### Added
- **Smarter fallback with error classification** — `_is_retryable_error()` classifies OpenAI SDK exceptions by HTTP status code:
  - **Retryable**: `TimeoutError`, `ConnectionError`, `APITimeoutError`, `APIConnectionError`, `429` (rate limit), `5xx` (server errors)
  - **Not retryable**: `400` (bad request), `401` (unauthorized), `403` (forbidden) — these now fail fast instead of burning the fallback attempt
  - Future non-OpenAI providers can extend this without touching the fallback logic
- **Token usage tracking** — `provider.stream_chat()` yields usage chunks that are now accumulated across tool iterations and surfaced in `EVENT_CONVERSATION_FINISHED`:
  - `usage.prompt_tokens` — input tokens consumed
  - `usage.completion_tokens` — output tokens generated
  - `usage.total_tokens` — combined total
- **Conversation outcome analytics** — `EVENT_CONVERSATION_FINISHED` now includes:
  - `outcome`: `"success"`, `"fallback_used"`, or `"failed"`
  - `error_type`: exception class name (e.g., `APIStatusError`, `ConnectionError`, `TokenLengthExceededError`) — `None` when successful
  - `usage`: token totals (omitted when both primary and fallback fail)

### Changed
- **`_async_handle_chat_log()` now returns `dict[str, int]`** with accumulated token usage instead of `None`
- **`_transform_stream()` accumulates usage** via a mutable `usage_accumulator` dict passed from the caller

### Testing
- **177 tests, 97% coverage** — Added:
  - Unit tests for `_is_retryable_error()` covering built-in exceptions, OpenAI SDK exceptions (429, 503, 401, 400), and the `ImportError` safety branch
  - Integration test verifying non-retryable errors (`ValueError`) skip fallback even when a fallback model is configured
  - Integration test verifying `EVENT_CONVERSATION_FINISHED` payload includes `outcome`, `error_type`, and `usage` on success
  - Integration test verifying `outcome="fallback_used"` and `error_type="ConnectionError"` when fallback succeeds after primary failure
  - Entity-level test for `_transform_stream` usage accumulator

## [Unreleased]

- Rename integration to `chathaus` (v1.0)
- Structured output / JSON schema validation UI
- Memory / context window management strategies beyond "clear"
