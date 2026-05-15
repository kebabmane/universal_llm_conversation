# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-05-15

### Fixed
- **Template rendering crash** — `get_exposed_entities` in `_build_system_prompt` was passed as a bare function reference instead of a lambda, causing `TypeError: missing 1 required positional argument: 'hass'` when the prompt template referenced `{{ universal_llm.exposed_entities }}`
- **Tool call ID shortening crash** — Local variable `shorten_tool_call_id` (boolean option) shadowed the imported `shorten_tool_call_id` helper function, causing `TypeError: 'bool' object is not callable` when the option was enabled

### Testing
- All 13 integration tests now pass (previously all failed due to `IntegrationNotFound`)
- Added dependency mocking for `ai_task`, `energy`, `history`, `recorder`, `rest`, `scrape` in test environment
- Added `PyTurboJPEG` to test requirements
- Fixed `hass.services.async_call` patching for HA 2026.5 read-only registry
- Updated config flow tests for v0.1.1 two-step flow with model fetch mocking
- Added coverage tests for `_convert_content_to_param`, `_execute_function_tool`, fallback error path, and agent skills property
- Overall test coverage improved from ~40% to ~60%

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

## [Unreleased]

- Native Anthropic provider
- Native Google Gemini provider
- Per-provider error classification for smarter fallback triggering (e.g., only retry on 429/5xx, not 401/400)
- Token usage tracking and cost estimation
- Streaming support for non-OpenAI-compatible providers
- Structured output / JSON schema validation
