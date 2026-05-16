# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

- Native Anthropic provider
- Native Google Gemini provider
- Per-provider error classification for smarter fallback triggering (e.g., only retry on 429/5xx, not 401/400)
- Token usage tracking and cost estimation
- Streaming support for non-OpenAI-compatible providers
- Structured output / JSON schema validation
