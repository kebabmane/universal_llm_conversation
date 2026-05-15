# Universal LLM Conversation

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/kebabmane/universal_llm_conversation)](https://github.com/kebabmane/universal_llm_conversation/releases)
[![License](https://img.shields.io/github/license/kebabmane/universal_llm_conversation)](LICENSE)

A provider-agnostic Home Assistant custom component for LLM-powered voice and text conversation.

## Why this exists

The popular [`extended_openai_conversation`](https://github.com/jekalmin/extended_openai_conversation) is tightly coupled to OpenAI APIs. It uses model-name regexes to guess parameter support, hardcodes `strict=True` on function schemas, has no request timeouts, no fallback models, and leaks tool-call syntax / reasoning content into TTS responses.

**Universal LLM Conversation** fixes all of that with a capability-based provider abstraction.

## Features

- **Broad provider support** — Works with any OpenAI-compatible endpoint (Fireworks, Firepass, Ollama, OpenRouter, LiteLLM, Groq, vLLM, etc.). Native Anthropic and Gemini support planned.
- **Capability-based parameters** — No more regex guessing. Temperature, top_p, max_tokens vs max_completion_tokens are sent only if the provider supports them.
- **Hide thinking / reasoning** — Strips Kimi-style `reasoning_content` and other thinking blocks before they reach TTS.
- **Response sanitization** — Removes leaked tool-call syntax (`execute_services(...)`) from voice responses.
- **Request timeouts** — Default 60s instead of hanging for 10 minutes.
- **Fallback model** — Configure a backup model; on failure the agent retries once before giving up.
- **Strict schema toggle** — Default `strict=False` for broad compatibility. Enable for OpenAI models that support it.
- **Skills** — Download and enable reusable AI capabilities.
- **Custom functions** — Template, script, bash, composite, and native HA service execution.

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant.
2. Go to **Integrations** > **Custom Repositories**.
3. Add `https://github.com/kebabmane/universal_llm_conversation` as an **Integration** type.
4. Search for **Universal LLM Conversation** in HACS and install it.
5. Restart Home Assistant.

### Manual

1. Download the latest release from [GitHub](https://github.com/kebabmane/universal_llm_conversation/releases).
2. Copy `custom_components/universal_llm_conversation/` into your Home Assistant `config/custom_components/` directory.
3. Restart Home Assistant.

## Setup

1. Go to **Settings > Devices & Services > Add Integration**.
2. Search for **Universal LLM Conversation**.
3. Enter your API key and provider settings.
4. Go to **Settings > Voice Assistants**, edit your assistant, and select **Universal LLM Conversation** as the conversation agent.

## Kimi K2.6 / Firepass Quick Setup

- **Provider**: `OpenAI Compatible`
- **Base URL**: `https://api.firepass.ai/v1` (or your Firepass endpoint)
- **API Key**: Your Firepass API key
- **Chat Model**: `kimi-k2.6` (or `firepass/accounts/fireworks/routers/kimi-k2p6-turbo` depending on your routing)
- **Advanced Options**:
  - `Schema Strict`: OFF
  - `Hide Thinking`: ON
  - `Request Timeout`: 60s
  - `Fallback Model`: `gpt-4o-mini` (or any backup)

## Project Status

This is a clean-break rewrite. There is no migration from `extended_openai_conversation`. You must re-create the integration and copy your function YAML manually.

## Architecture

See [`AGENTS.md`](custom_components/universal_llm_conversation/AGENTS.md) for detailed architecture notes.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
