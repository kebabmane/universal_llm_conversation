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
| Image & PDF support | No | Yes — camera, files, media sources |
| Vision analysis service | No | `analyze_image` service call |

## Features

- **Broad provider support** — Fireworks, Fire Pass, Ollama, OpenRouter, LiteLLM, Groq, vLLM, LocalAI, Together, Anthropic, Gemini, and any OpenAI-compatible endpoint.
- **Capability-based parameters** — No regex guessing. `temperature`, `top_p`, `max_tokens` vs `max_completion_tokens` are sent only if the provider supports them.
- **Hide thinking / reasoning** — Strips Kimi-style `reasoning_content` and other thinking blocks before they reach TTS.
- **Response sanitization** — Removes leaked tool-call syntax (`execute_services(...)`) from voice responses.
- **Request timeouts** — Default 60s instead of hanging for 10 minutes.
- **Fallback model** — Configure a backup model; on failure the agent rewinds the chat log and retries once.
- **Strict schema toggle** — Default `strict=False` for broad compatibility. Enable for providers that support it.
- **Image & PDF attachments in chat** — Send images and PDFs in conversation turns; automatically resized to 1568px and base64-encoded for vision-capable models (GPT-4o, Claude, Gemini, Kimi K2.6).
- **`analyze_image` service** — One-off vision analysis via Home Assistant service calls, supporting camera snapshots, image entities, media sources, and local files.
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

## Image & Vision Analysis

### In-Chat Attachments

When a conversation turn includes image or PDF attachments (via the Home Assistant Assist pipeline or frontend upload layer), the component:

1. Validates the model supports vision (`supports_vision=True`)
2. Resizes images to max 1568px via Pillow to stay within API limits
3. Base64-encodes attachments into provider-specific multimodal format
4. PDFs are supported by Anthropic and Gemini; skipped with a warning on OpenAI-compatible providers

### `analyze_image` Service

For one-off vision analysis without starting a conversation, call the `universal_llm_conversation.analyze_image` service:

```yaml
service: universal_llm_conversation.analyze_image
data:
  agent_id: "01ABCDEF..."
  images:
    - "media-source://camera/camera.front_door"
    - "/config/www/photo.jpg"
  prompt: "What do you see?"
  max_tokens: 4000
```

**Supported image sources:**
- `media-source://camera/<entity_id>` — Live camera snapshot
- `media-source://image/<entity_id>` — Image entity
- `media-source://...` — Generic media source (resolved via HA's media source system)
- Local file path — Must be in an allowed path (e.g. `/config/www/`)

**Parameters:**
| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `agent_id` | Yes | — | The Universal LLM agent entity ID |
| `images` | Yes | — | List of image sources (minimum 1) |
| `prompt` | No | "Describe this image." | Instructions for the LLM |
| `max_tokens` | No | 2000 | Max tokens for the response. Vision tasks need 1000–4000+ |

**Vision-capable models:**
- GPT-4o, GPT-4o-mini (OpenAI-compatible)
- Claude 3/4 series (Anthropic)
- Gemini 1.5/2.0 series (Google)
- Kimi K2.6, Kimi K2p6 (Moonshot via Fireworks)

Non-vision models raise a clear `HomeAssistantError` if images are provided.

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

> **Note:** Fire Pass keys are restricted to the Kimi K2.6 Turbo model and cannot list other models. The preset handles this automatically. Vision analysis is enabled for this model.

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
- **Context truncation** only supports the "clear" strategy (delete to last user message).
- **PDFs on OpenAI-compatible providers** are skipped with a warning (Chat Completions API does not support inline PDFs). Use Anthropic or Gemini native providers for PDF analysis.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `401 Unauthorized` at runtime | `base_url` defaulted to OpenAI (`https://api.openai.com/v1`) | Verify preset selected in config; check entry data has `base_url` |
| `403` on `/v1/models` during setup | API tier restricts model listing (e.g. Fire Pass) | Use the dedicated Fire Pass preset, or enter model manually |
| Tool syntax in TTS output (e.g. "execute_services(...)") | Response sanitization not catching edge case | Report with the exact text; check `hide_thinking` is enabled |
| Reasoning blocks in voice | `reasoning_content` not stripped | Enable **Hide Thinking** in advanced options |
| Timeout errors | Provider slow or unreachable | Increase **Request Timeout** in advanced options (10–300s) |
| Fallback never triggers | Error is auth/bad-request (not network) | Fallback only retries on timeout/connection/rate-limit errors |
| Image analysis returns empty string | Model streams analysis in `reasoning_content` chunks (Kimi K2.6T) | Fixed in v0.1.21+ — both `content` and `reasoning_content` are collected |
| Vision model rejects images | Model ID doesn't match a vision override | Check model ID (e.g. `kimi-k2p6`, `gpt-4o`, `claude-*`, `gemini-*`) |
| "Image analysis hit token limit" | Default `max_tokens` (2000) too low for complex vision tasks | Increase `max_tokens` service parameter (try 4000–8000) |

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
