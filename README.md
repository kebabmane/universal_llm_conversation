# Universal LLM Conversation

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/kebabmane/universal_llm_conversation)](https://github.com/kebabmane/universal_llm_conversation/releases)
[![License](https://img.shields.io/github/license/kebabmane/universal_llm_conversation)](LICENSE)

A lean, provider-agnostic conversation agent for Home Assistant. Works with any OpenAI-compatible LLM endpoint.

## Why This Exists

The popular [`extended_openai_conversation`](https://github.com/jekalmin/extended_openai_conversation) is tightly coupled to OpenAI APIs. It uses model-name regexes to guess parameter support, hardcodes `strict=True` on function schemas, has no request timeouts, no fallback models, and leaks tool-call syntax / reasoning content into TTS responses.

**Universal LLM Conversation** fixes all of that with a capability-based provider abstraction.

| Pain Point | extended_openai_conversation | Universal LLM Conversation |
|-----------|------------------------------|--------------------------|
| Provider support | OpenAI only (Azure with hacks) | Any OpenAI-compatible endpoint |
| Parameter guessing | Regex on model name | Capability-based filtering per model |
| Schema strictness | Hardcoded `strict=True` | Toggle, default `False` |
| Timeouts | None (600s SDK default) | 60s default, configurable |
| Fallback model | None | Retry once with backup model |
| Tool-call leaks to TTS | Yes | Stripped before speech |
| Reasoning in voice | Leaked | Hidden by default |

## Features

- **Broad provider support** — Fireworks, Fire Pass, Ollama, OpenRouter, LiteLLM, Groq, vLLM, LocalAI, Together, and any OpenAI-compatible endpoint.
- **Capability-based parameters** — No regex guessing. `temperature`, `top_p`, `max_tokens` vs `max_completion_tokens` are sent only if the provider supports them.
- **Hide thinking / reasoning** — Strips Kimi-style `reasoning_content` and other thinking blocks before they reach TTS.
- **Response sanitization** — Removes leaked tool-call syntax (`execute_services(...)`) from voice responses.
- **Request timeouts** — Default 60s instead of hanging for 10 minutes.
- **Fallback model** — Configure a backup model; on failure the agent rewinds the chat log and retries once.
- **Strict schema toggle** — Default `strict=False` for broad compatibility. Enable for providers that support it.
- **Skills** — Download and enable reusable AI capabilities.
- **Custom functions** — Template, script, bash, composite, and native HA service execution.
- **3-step setup wizard** — Select preset → enter credentials → choose model.
- **Provider presets** — Fireworks, Fire Pass, OpenAI, Ollama, OpenRouter, Azure, Anthropic, Gemini, Custom.

## Quick Start

### Installation (HACS)

1. Open HACS in Home Assistant.
2. Go to **Integrations** > **Custom Repositories**.
3. Add `https://github.com/kebabmane/universal_llm_conversation` as an **Integration** type.
4. Search for **Universal LLM Conversation** and install it.
5. Restart Home Assistant.

### Setup

1. Go to **Settings > Devices & Services > Add Integration**.
2. Search for **Universal LLM Conversation**.
3. Follow the 3-step wizard:
   1. **Select preset** — Choose your LLM provider (e.g. Fireworks AI, Ollama, OpenRouter)
   2. **Enter credentials** — API key (base_url is auto-filled for presets; editable for Ollama/Custom)
   3. **Choose model** — Pick from the fetched dropdown or enter manually
4. Go to **Settings > Voice Assistants**, edit your assistant, and select **Universal LLM Conversation** as the conversation agent.

## Provider Quick Starts

### Fireworks AI — Fire Pass (Kimi K2.6)

For Fire Pass ($49/month early access) subscribers:

| Setting | Value |
|---------|-------|
| Preset | **Fireworks AI — Fire Pass** |
| API Key | Your Fireworks API key |
| Chat Model | `accounts/fireworks/routers/kimi-k2p6-turbo` (pre-selected) |
| Schema Strict | OFF |
| Hide Thinking | ON |
| Request Timeout | 60s |
| Fallback Model | *(none — Fire Pass covers one model)* |

> **Note:** Fire Pass keys are restricted to the Kimi K2.6 Turbo model and cannot list other models. The preset handles this automatically.

### Fireworks AI (Standard)

| Setting | Value |
|---------|-------|
| Preset | **Fireworks AI** |
| API Key | Your Fireworks API key |
| Chat Model | Fetched from `/v1/models` (e.g. `accounts/fireworks/models/llama-v3p1-70b`) |
| Schema Strict | OFF |
| Hide Thinking | ON (for reasoning models) |

### Ollama (Local)

| Setting | Value |
|---------|-------|
| Preset | **Ollama (Local)** |
| API Key | *(optional — leave blank if no auth)* |
| Base URL | `http://localhost:11434/v1` (editable for remote hosts) |
| Chat Model | Fetched from `/v1/models` (e.g. `llama3.1`, `mistral`) |

### OpenRouter

| Setting | Value |
|---------|-------|
| Preset | **OpenRouter** |
| API Key | Your OpenRouter API key |
| Chat Model | Fetched from `/v1/models` |
| Schema Strict | OFF |

### Custom / LiteLLM / Groq / vLLM / LocalAI

| Setting | Value |
|---------|-------|
| Preset | **Custom / Other** |
| API Key | Your API key |
| Base URL | Your endpoint URL (e.g. `https://api.groq.com/openai/v1`) |
| Chat Model | Enter manually (e.g. `llama-3.1-70b-versatile`) |

## Architecture

See [`AGENTS.md`](custom_components/universal_llm_conversation/AGENTS.md) for detailed architecture notes, release process, and testing strategy.

## Known Limitations

- **No migration** from `extended_openai_conversation`. This is a clean break — re-create the integration and copy your function YAML manually.
- **Native Anthropic (Claude) and Google Gemini providers** are not yet implemented. Use the OpenRouter or LiteLLM proxy presets as a workaround.
- **Context truncation** only supports the "clear" strategy (delete to last user message).

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `401 Unauthorized` at runtime | `base_url` defaulted to OpenAI (`https://api.openai.com/v1`) | Verify preset selected in config; check entry data has `base_url` |
| `403` on `/v1/models` during setup | API tier restricts model listing (e.g. Fire Pass) | Use the dedicated Fire Pass preset, or enter model manually |
| Tool syntax in TTS output (e.g. "execute_services(...)") | Response sanitization not catching edge case | Report with the exact text; check `hide_thinking` is enabled |
| Reasoning blocks in voice | `reasoning_content` not stripped | Enable **Hide Thinking** in advanced options |
| Timeout errors | Provider slow or unreachable | Increase **Request Timeout** in advanced options (10–300s) |
| Fallback never triggers | Error is auth/bad-request (not network) | Fallback only retries on timeout/connection/rate-limit errors |

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
