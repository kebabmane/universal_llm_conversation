# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
