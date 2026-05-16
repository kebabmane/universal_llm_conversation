"""Universal LLM Conversation agent entity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    async_get_chat_log,
)
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm, template
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import UniversalLLMConfigEntry
from .const import (
    CONF_FALLBACK_MODEL,
    CONF_PROMPT,
    CONF_SKILLS,
    DEFAULT_PROMPT,
    DEFAULT_WORKING_DIRECTORY,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .entity import UniversalLLMBaseEntity
from .helpers import get_exposed_entities, sanitize_for_speech
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)


def _is_retryable_error(err: Exception) -> bool:
    """Classify whether an exception warrants fallback retry.

    Retryable: network issues, timeouts, rate-limits (429), server errors (5xx).
    Not retryable: auth failures (401), bad requests (400), etc.
    """
    if isinstance(err, (TimeoutError, ConnectionError)):
        return True
    try:
        from openai import APIStatusError, APITimeoutError, APIConnectionError
    except ImportError:
        return False
    if isinstance(err, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(err, APIStatusError):
        return err.status_code == 429 or err.status_code >= 500
    return False


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: UniversalLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [UniversalLLMAgentEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class UniversalLLMAgentEntity(
    ConversationEntity,
    conversation.AbstractConversationAgent,
    UniversalLLMBaseEntity,
):
    """Universal LLM conversation agent."""

    _attr_supports_streaming = True
    _attr_supported_features = ConversationEntityFeature.CONTROL
    skill_manager: SkillManager

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    @property
    def skills(self) -> list[str]:
        return self.subentry.data.get(CONF_SKILLS, []) or []

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

        working_dir = DEFAULT_WORKING_DIRECTORY
        if Path(working_dir).is_absolute():
            skills_dir = Path(working_dir) / "skills"
        else:
            skills_dir = Path(self.hass.config.config_dir) / working_dir / "skills"

        self.skill_manager = await SkillManager.async_get_instance(
            self.hass, user_skills_dir=str(skills_dir)
        )

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        llm_context = user_input.as_llm_context(DOMAIN)
        exposed_entities = self._get_exposed_entities()
        function_tools = self._get_function_tools()
        system_prompt = self._build_system_prompt(exposed_entities, llm_context, user_input)
        chat_log.content[0] = conversation.SystemContent(content=system_prompt)

        fallback_model = self.subentry.data.get(CONF_FALLBACK_MODEL, "")
        last_error: Exception | None = None
        outcome = "success"
        error_type: str | None = None
        usage: dict[str, int] | None = None

        # Try primary model
        try:
            usage = await self._async_handle_chat_log(
                chat_log,
                function_tools=function_tools,
                exposed_entities=exposed_entities,
                llm_context=llm_context,
            )
        except Exception as err:
            last_error = err
            error_type = type(err).__name__
            _LOGGER.warning("Primary model failed: %s", err)

            # Try fallback only for retryable errors
            if _is_retryable_error(err) and fallback_model:
                try:
                    _LOGGER.info("Falling back to model: %s", fallback_model)
                    # Reset chat log to before the failed assistant turn
                    # Remove any partial assistant content added during failed stream
                    while (
                        len(chat_log.content) > 1
                        and isinstance(chat_log.content[-1], conversation.AssistantContent)
                    ):
                        chat_log.content.pop()

                    usage = await self._async_handle_chat_log(
                        chat_log,
                        function_tools=function_tools,
                        exposed_entities=exposed_entities,
                        llm_context=llm_context,
                        model_override=fallback_model,
                    )
                    outcome = "fallback_used"
                    last_error = None
                except Exception as fallback_err:
                    last_error = fallback_err
                    error_type = f"fallback_{type(fallback_err).__name__}"
                    _LOGGER.error("Fallback model also failed: %s", fallback_err)
                    outcome = "failed"
            else:
                outcome = "failed"

        # Build event payload
        event_payload: dict[str, Any] = {
            "user_input": {
                "text": user_input.text,
                "conversation_id": user_input.conversation_id,
                "language": user_input.language,
            },
            "messages": [c.as_dict() for c in chat_log.content],
            "agent_id": self.subentry.subentry_id,
            "outcome": outcome,
            "error_type": error_type,
        }
        if usage is not None:
            event_payload["usage"] = usage

        self.hass.bus.async_fire(EVENT_CONVERSATION_FINISHED, event_payload)

        if last_error:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to the LLM: {last_error}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        # Build response with sanitization
        intent_response = intent.IntentResponse(language=user_input.language)
        last_content = chat_log.content[-1]
        if isinstance(last_content, conversation.AssistantContent):
            speech_text = last_content.content or ""
            # Get function names for sanitizer
            func_names = [ft["spec"]["name"] for ft in function_tools if "spec" in ft]
            sanitized = sanitize_for_speech(speech_text, func_names)
            intent_response.async_set_speech(sanitized or "")
        else:
            intent_response.async_set_speech("")

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    def _build_system_prompt(
        self,
        exposed_entities: list[dict],
        llm_context: llm.LLMContext,
        user_input: ConversationInput,
    ) -> str:
        raw_prompt: str = self.subentry.data.get(CONF_PROMPT, DEFAULT_PROMPT)
        result = template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": llm_context.device_id,
                "user_input": user_input,
                "skills": self._get_enabled_skills(),
                "universal_llm": {
                    "working_directory": lambda: str(
                        Path(self.hass.config.config_dir) / DEFAULT_WORKING_DIRECTORY
                    ),
                    "exposed_entities": lambda: get_exposed_entities(self.hass),
                    "skill_dir": lambda name: str(
                        Path(self.hass.config.config_dir)
                        / DEFAULT_WORKING_DIRECTORY
                        / "skills"
                        / name
                    ),
                },
            },
            parse_result=False,
        )
        return str(result)

    def _get_enabled_skills(self) -> list[Any]:
        enabled_names = self.skills
        all_skills = self.skill_manager.get_all_skills()
        return [s for s in all_skills if s.name in enabled_names]
