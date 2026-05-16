"""Integration tests for conversation entity."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import yaml

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent

from custom_components.universal_llm_conversation.const import DOMAIN


@pytest.mark.usefixtures("mock_validate_connection")
async def test_conversation_agent_setup(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test conversation entity is created."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Agent is registered under the config entry id
    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent is not None
    assert agent.supported_languages == "*"


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream")
async def test_converse_returns_response(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test basic conversation returns a response."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "Hello from test" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_tool")
async def test_converse_executes_tool_call(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test conversation triggers tool execution."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Track service calls via class-level patch
    service_calls = []
    original_call = hass.services.async_call

    async def tracking_call(self, domain, service, service_data, **kwargs):
        service_calls.append((domain, service, service_data))
        return await original_call(domain, service, service_data, **kwargs)

    with patch("homeassistant.core.ServiceRegistry.async_call", tracking_call):
        result = await conversation.async_converse(
            hass,
            "turn on the light",
            None,
            Context(),
            agent_id=mock_config_entry.entry_id,
        )

    # Verify service was called
    assert any(
        call[0] == "light" and call[1] == "turn_on"
        for call in service_calls
    )


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_fallback")
async def test_fallback_model_retry(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test fallback model is used when primary fails."""
    # Configure fallback model
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "Fallback response" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_always_fail")
async def test_both_models_fail_returns_error(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test error response when primary and fallback both fail."""
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "problem talking to the LLM" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_non_retryable")
async def test_non_retryable_error_skips_fallback(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test fallback is NOT used for non-retryable errors (e.g., 400/401)."""
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    # Should error immediately without attempting fallback
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "problem talking to the LLM" in result.response.speech["plain"]["speech"]


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_fallback")
async def test_fallback_event_includes_outcome(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test EVENT_CONVERSATION_FINISHED includes outcome and usage on fallback."""
    from custom_components.universal_llm_conversation.const import EVENT_CONVERSATION_FINISHED

    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    events = []

    def event_listener(event):
        events.append(event)

    hass.bus.async_listen(EVENT_CONVERSATION_FINISHED, event_listener)

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    await hass.async_block_till_done()

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert len(events) == 1
    assert events[0].data["outcome"] == "fallback_used"
    assert events[0].data["error_type"] == "ConnectionError"


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_with_usage")
async def test_conversation_event_includes_usage(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test EVENT_CONVERSATION_FINISHED includes token usage."""
    from custom_components.universal_llm_conversation.const import EVENT_CONVERSATION_FINISHED

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    events = []

    def event_listener(event):
        events.append(event)

    hass.bus.async_listen(EVENT_CONVERSATION_FINISHED, event_listener)

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    await hass.async_block_till_done()

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert len(events) == 1
    assert events[0].data["outcome"] == "success"
    assert events[0].data["usage"]["prompt_tokens"] == 10
    assert events[0].data["usage"]["completion_tokens"] == 5
    assert events[0].data["usage"]["total_tokens"] == 15


@pytest.mark.usefixtures("mock_validate_connection")
async def test_agent_skills_empty_by_default(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test agent skills property returns empty list by default."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent.skills == []


@pytest.mark.usefixtures("mock_validate_connection")
async def test_sanitize_speech_with_function_names(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test speech sanitization strips known function names."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Configure function tools
    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "functions": yaml.dump([{"spec": {"name": "turn_on_light"}, "function": {"type": "native"}}])},
    )

    result = await conversation.async_converse(
        hass,
        "turn on the light",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    # Response should not contain leaked function syntax
    speech = result.response.speech["plain"]["speech"]
    assert "turn_on_light(" not in speech


@pytest.mark.usefixtures("mock_validate_connection")
async def test_async_added_to_hass_absolute_path(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test skill directory resolution when working dir is absolute."""
    from custom_components.universal_llm_conversation.const import DEFAULT_WORKING_DIRECTORY
    from pathlib import Path

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(
        mock_config_entry.entry_id
    )
    assert agent is not None
    # Verify skill_manager was initialized
    assert agent.skill_manager is not None
    # The skills_dir should be based on DEFAULT_WORKING_DIRECTORY
    assert "skills" in str(agent.skill_manager.user_skills_dir)


def test_is_retryable_error_classification() -> None:
    """Test _is_retryable_error correctly classifies exceptions."""
    from custom_components.universal_llm_conversation.conversation import _is_retryable_error
    import httpx

    # Built-in retryable
    assert _is_retryable_error(TimeoutError("timed out")) is True
    assert _is_retryable_error(ConnectionError("reset")) is True

    # Non-retryable built-in
    assert _is_retryable_error(ValueError("bad")) is False
    assert _is_retryable_error(RuntimeError("oops")) is False

    # OpenAI SDK errors
    from openai import APIStatusError, APITimeoutError, APIConnectionError

    req = httpx.Request("GET", "http://test")
    assert _is_retryable_error(APITimeoutError(req)) is True
    assert _is_retryable_error(APIConnectionError(request=req)) is True

    # 429 -> retryable
    resp429 = httpx.Response(status_code=429, request=req)
    assert _is_retryable_error(APIStatusError("rate limit", response=resp429, body=None)) is True

    # 502/503/504 -> retryable
    resp503 = httpx.Response(status_code=503, request=req)
    assert _is_retryable_error(APIStatusError("unavailable", response=resp503, body=None)) is True

    # 400/401/403 -> NOT retryable
    resp401 = httpx.Response(status_code=401, request=req)
    assert _is_retryable_error(APIStatusError("auth", response=resp401, body=None)) is False

    resp400 = httpx.Response(status_code=400, request=req)
    assert _is_retryable_error(APIStatusError("bad req", response=resp400, body=None)) is False

    # ImportError branch when openai SDK is not available
    with patch("builtins.__import__", side_effect=ImportError("no openai")):
        assert _is_retryable_error(ValueError("whatever")) is False


@pytest.mark.usefixtures("mock_validate_connection")
async def test_vision_disabled_raises_error(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test HomeAssistantError when non-vision model receives attachments."""
    from homeassistant.exceptions import HomeAssistantError
    from unittest.mock import MagicMock, patch

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

    # Mock provider without vision support
    mock_provider = MagicMock()
    mock_provider.model = "non-vision-model"
    mock_provider.capabilities.supports_vision = False
    mock_provider.filter_params = lambda p: p
    mock_provider.stream_chat = MagicMock(return_value=[])

    # Mock chat_log with attachment
    chat_log = MagicMock()
    att = MagicMock()
    att.mime_type = "image/jpeg"
    att.path = MagicMock()
    att.path.read_bytes = MagicMock(return_value=b"fake")
    chat_log.content = [
        conversation.SystemContent(content="sys"),
        conversation.UserContent(content="hi", attachments=[att]),
    ]

    with patch.object(agent, "_get_provider", return_value=mock_provider):
        with pytest.raises(HomeAssistantError) as exc_info:
            await agent._async_handle_chat_log(
                chat_log,
                function_tools=[],
                exposed_entities=[],
            )
    assert "does not support image or PDF attachments" in str(exc_info.value)


@pytest.mark.usefixtures("mock_validate_connection")
async def test_non_conversation_subentry_skipped(
    hass: HomeAssistant,
) -> None:
    """Test that non-conversation subentries do not create conversation entities."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from homeassistant.config_entries import ConfigSubentryData
    from custom_components.universal_llm_conversation.config_flow import DEFAULT_OPTIONS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test Universal LLM",
        data={
            "api_key": "test-api-key",
            "provider_preset": "custom",
            "base_url": "http://localhost:1234/v1",
            "api_version": None,
            "organization": None,
            "skip_authentication": False,
        },
        version=1,
        subentries_data=[
            ConfigSubentryData(
                data=dict(DEFAULT_OPTIONS),
                subentry_type="other_type",
                title="Other Subentry",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)

    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Only the built-in Home Assistant conversation entity should exist
    states = hass.states.async_all("conversation")
    assert all("home_assistant" in s.entity_id for s in states)


@pytest.mark.usefixtures("mock_validate_connection", "mock_provider_stream_always_fail")
async def test_fallback_event_payload_on_dual_failure(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test EVENT_CONVERSATION_FINISHED payload when both models fail."""
    from custom_components.universal_llm_conversation.const import EVENT_CONVERSATION_FINISHED

    subentry = next(iter(mock_config_entry.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "fallback_model": "gpt-4o-mini"},
    )

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    events = []

    def event_listener(event):
        events.append(event)

    hass.bus.async_listen(EVENT_CONVERSATION_FINISHED, event_listener)

    result = await conversation.async_converse(
        hass,
        "hello",
        None,
        Context(),
        agent_id=mock_config_entry.entry_id,
    )

    await hass.async_block_till_done()

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert len(events) == 1
    assert events[0].data["outcome"] == "failed"
    assert events[0].data["error_type"] == "fallback_ConnectionError"


@pytest.mark.usefixtures("mock_validate_connection")
async def test_relative_working_directory(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test skill directory resolution when working dir is relative."""
    from pathlib import Path
    from unittest.mock import patch

    with patch(
        "custom_components.universal_llm_conversation.conversation.DEFAULT_WORKING_DIRECTORY",
        "relative_dir",
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

        agent = conversation.get_agent_manager(hass).async_get_agent(
            mock_config_entry.entry_id
        )
        assert agent is not None
        expected = Path(hass.config.config_dir) / "relative_dir" / "skills"
        assert str(expected) in str(agent.skill_manager.user_skills_dir)


@pytest.mark.usefixtures("mock_validate_connection")
async def test_last_content_not_assistant(
    hass: HomeAssistant,
    mock_config_entry: object,
) -> None:
    """Test speech is empty when last content is not AssistantContent."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    agent = conversation.get_agent_manager(hass).async_get_agent(mock_config_entry.entry_id)

    with patch.object(agent, "_async_handle_chat_log", return_value={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}):
        result = await conversation.async_converse(
            hass,
            "hello",
            None,
            Context(),
            agent_id=mock_config_entry.entry_id,
        )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == ""
