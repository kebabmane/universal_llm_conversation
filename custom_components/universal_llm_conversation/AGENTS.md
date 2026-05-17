# Universal LLM Conversation — Agent Notes

## Domain
`universal_llm_conversation`

## Purpose
A provider-agnostic Home Assistant conversation agent that replaces `extended_openai_conversation` with broad LLM support, robust error handling, and voice-friendly output sanitization.

## Provider Support
- **OpenAI Compatible** — Fireworks, Firepass, Ollama, OpenRouter, vLLM, LocalAI, LiteLLM proxy, Groq, Together, etc. (default)
- **Anthropic (Claude)** — Native API with streaming, tool calls, thinking content, and vision
- **Google Gemini** — Native API with streaming, function calling, and vision

## Key Architecture Decisions

### Capability-based parameter filtering
Instead of regex-guessing model support, each provider declares capabilities. Params are filtered before every API call. This fixes Kimi K2.6 rejecting `temperature`/`top_p`, and Anthropic/LiteLLM conflicts.

### Response Sanitization
`helpers.sanitize_for_speech()` strips leaked tool-call syntax and reasoning blocks before TTS. Fixes issue #434 from extended_openai_conversation.

### Request Timeouts
Default 60s per-request timeout (was 600s OpenAI SDK default). Configurable per agent.

### Fallback Model
On any error, if a fallback model is configured, the component rewinds the chat log and retries once with the fallback.

### Schema Strictness
Default `strict=False` for broader non-OpenAI compatibility. Toggle per agent for better tool accuracy on providers that support it.

### Thinking Content
Default `hide_thinking=True`. Kimi K2.6 `reasoning_content` is captured but not emitted to TTS. Can be disabled to pass thinking through (e.g., for UI display).

### Image & Vision Support
Images and PDFs flow through `UserContent.attachments` for chat conversations, and through the dedicated `analyze_image` service for one-off analysis. The `supports_vision` capability on `ProviderCapabilities` determines whether a model accepts multimodal input. Images are resized to 1568px max dimension via Pillow before base64 encoding. Provider-specific formatting: OpenAI-compatible uses `image_url` data URIs; Anthropic uses `image`/`document` content blocks; Gemini uses `Part.from_bytes()`. PDFs are skipped with a warning on OpenAI Chat Completions but supported by Anthropic and Gemini.

## File Map

| File | Role |
|------|------|
| `__init__.py` | HA bootstrap, provider validation on setup |
| `manifest.json` | Integration metadata, deps |
| `config_flow.py` | Config + subentry flows with provider-aware options |
| `const.py` | Constants, default prompt, default functions |
| `conversation.py` | Main agent entity, fallback logic, response sanitization |
| `entity.py` | Base LLM entity, streaming transformation, tool execution |
| `helpers.py` | Exposed entities, provider factory, speech sanitizer |
| `providers/base.py` | Abstract provider class |
| `providers/openai_compatible.py` | OpenAI/Azure client with capability filtering, image attachment conversion |
| `providers/anthropic.py` | Native Anthropic client with vision, thinking, and document support |
| `providers/gemini.py` | Native Gemini client with vision and function calling |
| `functions.py` | Native function registry (execute_service, template, script, composite, bash, read_file) |
| `skills.py` | Skill discovery, download, reload |
| `services.py` | HA services for skill management and `analyze_image` |
| `exceptions.py` | Custom exceptions |
| `strings.json` | UI labels |

## Testing Strategy
1. Install as custom component in HA.
2. Configure with Firepass base URL and Kimi K2.6 model.
3. Verify no `temperature`/`top_p` sent in request.
4. Verify `max_completion_tokens` sent instead of `max_tokens`.
5. Verify tool calls work with `schema_strict=False`.
6. Verify voice output has no leaked reasoning or tool syntax.
7. Set fallback model and block primary endpoint to test fallback.
8. Test `analyze_image` service with camera snapshots, local files, and media sources.
9. Verify vision models accept images; non-vision models reject with clear error.
10. Verify `max_tokens` override in `analyze_image` service is applied.

## Release Process

**Critical: HACS discovers GitHub Releases, not git tags.** Pushing a git tag alone will NOT make the update visible in Home Assistant.

### Step-by-step release checklist

1. **Update `manifest.json`** — Bump the `"version"` field.
2. **Update `CHANGELOG.md`** — Add a new section with the version number and date.
3. **Run tests** — `venv/bin/python -m pytest tests/ --timeout=30 -q` must pass.
4. **Commit** — `git add -A && git commit -m "vX.Y.Z: Description"`.
5. **Push** — `git push origin main`.
6. **Create git tag** — `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin --tags`.
7. **Create GitHub Release** — Use `gh` CLI (REQUIRED for HACS):
   ```bash
   gh release create vX.Y.Z \
     --title "vX.Y.Z — Short description" \
     --notes "Release notes from CHANGELOG..." \
     --latest
   ```
   Or create it manually at https://github.com/kebabmane/universal_llm_conversation/releases/new.
8. **Verify in HACS** — In Home Assistant, go to **HACS → Integrations → Universal LLM Conversation → Update information**. The new version should appear.

### Important notes
- Never skip step 7. HACS polls GitHub Releases API, not git tags.
- The release title and notes are what users see in HACS.
- Use `--latest` flag so GitHub marks it as the latest release.

## Migration
This is a clean break — no migration from `extended_openai_conversation`. Users must re-create integrations and copy their function YAML manually.
