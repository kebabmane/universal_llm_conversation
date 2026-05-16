# Universal LLM Conversation

A lean, provider-agnostic LLM conversation agent for Home Assistant.

**Works with:** Any OpenAI-compatible endpoint — Fireworks, Fire Pass, Ollama, OpenRouter, Groq, vLLM, LiteLLM, LocalAI, Together, and more.

**Why this over extended_openai_conversation?**
- No regex-guessing model support — capability-based parameter filtering
- No tool-call syntax leaking into TTS — voice-friendly sanitization
- No 10-minute hangs — configurable timeouts
- No single point of failure — fallback model retries
- No OpenAI lock-in — 9 provider presets with auto-filled endpoints

**Setup:** 3-step wizard (preset → credentials → model) via **Settings > Devices & Services > Add Integration**.

[Full Documentation → README.md](https://github.com/kebabmane/universal_llm_conversation#readme)
