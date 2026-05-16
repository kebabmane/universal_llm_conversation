"""Config flow for Universal LLM Conversation integration."""

from __future__ import annotations

import logging
import types
from typing import Any

import voluptuous as vol
import yaml

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    API_PROVIDERS,
    CONF_ADVANCED_OPTIONS,
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FALLBACK_MODEL,
    CONF_FUNCTION_TOOLS,
    CONF_HIDE_THINKING,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_PROVIDER_PRESET,
    CONF_REQUEST_TIMEOUT,
    CONF_SCHEMA_STRICT,
    CONF_SHORTEN_TOOL_CALL_ID,
    CONF_SKILLS,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ADVANCED_OPTIONS,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_AI_TASK_OPTIONS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_CONF_FUNCTION_TOOLS,
    DEFAULT_HIDE_THINKING,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_SCHEMA_STRICT,
    DEFAULT_SHORTEN_TOOL_CALL_ID,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    PROVIDER_PRESETS,
)
from .helpers import async_fetch_models, get_provider
from .skills import SkillManager

_LOGGER = logging.getLogger(__name__)

DEFAULT_CONF_FUNCTION_TOOLS_STR = yaml.dump(DEFAULT_CONF_FUNCTION_TOOLS, sort_keys=False)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTION_TOOLS: DEFAULT_CONF_FUNCTION_TOOLS_STR,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_SHORTEN_TOOL_CALL_ID: DEFAULT_SHORTEN_TOOL_CALL_ID,
        CONF_ADVANCED_OPTIONS: DEFAULT_ADVANCED_OPTIONS,
        CONF_SCHEMA_STRICT: DEFAULT_SCHEMA_STRICT,
        CONF_HIDE_THINKING: DEFAULT_HIDE_THINKING,
        CONF_REQUEST_TIMEOUT: DEFAULT_REQUEST_TIMEOUT,
        CONF_FALLBACK_MODEL: DEFAULT_FALLBACK_MODEL,
    }
)


def _get_base_url_from_preset(data: dict[str, Any]) -> str | None:
    """Resolve base URL from provider preset or manual input."""
    preset_key = data.get(CONF_PROVIDER_PRESET, "custom")
    preset = PROVIDER_PRESETS.get(preset_key, {})
    preset_base = preset.get("base_url", "")
    manual_base = data.get(CONF_BASE_URL, "")
    # Use manual override if provided, otherwise use preset default
    if manual_base:
        return manual_base if manual_base != preset_base else preset_base
    return preset_base or None


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate user input allows us to connect.

    Raises HomeAssistantError with a machine-readable error key:
    - invalid_auth
    - timeout
    - cannot_connect
    """
    api_key = data[CONF_API_KEY]
    base_url = _get_base_url_from_preset(data)
    api_version = data.get(CONF_API_VERSION)
    organization = data.get(CONF_ORGANIZATION)
    skip_auth = data.get(CONF_SKIP_AUTHENTICATION, False)
    provider_key = data.get("provider", "openai_compatible")

    if skip_auth:
        return

    provider = get_provider(
        hass=hass,
        provider_key=provider_key,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        organization=organization,
        model="gpt-4o-mini",
        timeout=10.0,
    )

    await provider.validate_connection()


class UniversalLLMConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the provider selection step."""
        if user_input is None:
            schema = vol.Schema(
                {
                    vol.Optional(CONF_NAME, default="Universal LLM"): str,
                    vol.Required(
                        CONF_PROVIDER_PRESET, default="fireworks"
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                SelectOptionDict(value=k, label=v["label"])
                                for k, v in PROVIDER_PRESETS.items()
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            )
            return self.async_show_form(step_id="user", data_schema=schema)

        # Store provider selection and proceed to credentials
        self.context["user_input"] = user_input
        return await self.async_step_credentials()

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the credentials step."""
        user_data: dict[str, Any] = self.context.get("user_input", {})
        preset_key = user_data.get(CONF_PROVIDER_PRESET, "custom")
        preset = PROVIDER_PRESETS.get(preset_key, {})

        if user_input is not None:
            # Merge credentials into stored user data
            merged = {**user_data, **user_input}
            self.context["user_input"] = merged

            errors = {}
            try:
                await validate_input(self.hass, merged)
            except HomeAssistantError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = str(err) if str(err) else "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected validation error")
                errors["base"] = "cannot_connect"
            else:
                return await self.async_step_model()

            # Show credentials form again with error
            return self.async_show_form(
                step_id="credentials",
                data_schema=self._build_credentials_schema(preset_key),
                errors=errors,
                description_placeholders={
                    "provider": preset.get("label", preset_key),
                },
            )

        return self.async_show_form(
            step_id="credentials",
            data_schema=self._build_credentials_schema(preset_key),
            description_placeholders={
                "provider": preset.get("label", preset_key),
            },
        )

    def _build_credentials_schema(self, preset_key: str) -> vol.Schema:
        """Build credentials schema based on provider preset."""
        schema: dict[Any, Any] = {
            vol.Required(CONF_API_KEY): str,
            vol.Optional(
                CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
            ): BooleanSelector(),
        }

        # Only show base_url, api_version, organization for custom or azure
        if preset_key in ("custom", "azure"):
            schema[vol.Optional(CONF_BASE_URL)] = str
            schema[vol.Optional(CONF_API_VERSION)] = str
            schema[vol.Optional(CONF_ORGANIZATION)] = str

        return vol.Schema(schema)

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle model selection step."""
        user_data: dict[str, Any] = self.context.get("user_input", {})
        preset_key = user_data.get(CONF_PROVIDER_PRESET, "custom")
        preset = PROVIDER_PRESETS.get(preset_key, {})
        base_url = _get_base_url_from_preset(user_data)
        api_key = user_data.get(CONF_API_KEY, "")
        supports_model_list = preset.get("supports_model_list", False)

        if user_input is not None:
            # Merge model selection into user data
            chat_model = user_input.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            fallback_model = user_input.get(CONF_FALLBACK_MODEL, "")
            # Update the subentry defaults with selected model
            options = dict(DEFAULT_OPTIONS)
            options[CONF_CHAT_MODEL] = chat_model
            options[CONF_FALLBACK_MODEL] = fallback_model

            return self.async_create_entry(
                title=user_data.get(CONF_NAME, DEFAULT_NAME),
                data=user_data,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "data": options,
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": "ai_task_data",
                        "data": dict(DEFAULT_AI_TASK_OPTIONS),
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": None,
                    },
                ],
            )

        # Build schema based on whether we can fetch models
        schema: dict[Any, Any] = {}
        models: list[str] = []

        if supports_model_list and base_url and api_key:
            models = await async_fetch_models(
                self.hass, api_key=api_key, base_url=base_url, timeout=10.0
            )

        if models:
            # Show dropdown with fetched models + custom option
            schema[
                vol.Required(CONF_CHAT_MODEL, default=models[0] if models else DEFAULT_CHAT_MODEL)
            ] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=m, label=m)
                        for m in models
                    ] + [SelectOptionDict(value="__custom__", label="Enter custom model manually...")],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
        else:
            # Show plain text input when fetch failed or not supported
            schema[
                vol.Required(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL)
            ] = str

        schema[
            vol.Optional(CONF_FALLBACK_MODEL, default=DEFAULT_FALLBACK_MODEL)
        ] = str

        errors = {}
        if supports_model_list and not models:
            errors["base"] = "model_fetch_failed"

        return self.async_show_form(
            step_id="model",
            data_schema=vol.Schema(schema),
            errors=errors,
            description_placeholders={
                "provider": preset.get("label", preset_key),
            },
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {
            "conversation": UniversalLLMSubentryFlowHandler,
            "ai_task_data": UniversalLLMAITaskSubentryFlowHandler,
        }


class UniversalLLMSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing conversation subentries."""

    options: dict[str, Any]
    _temp_data: dict[str, Any] | None = None
    _available_skills: list[dict[str, Any]] | None = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self.options = dict(DEFAULT_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if self._available_skills is None:
            self._available_skills = await self._async_get_skills()

        if user_input is not None:
            if user_input.get(CONF_ADVANCED_OPTIONS, False):
                self._temp_data = user_input
                return await self.async_step_advanced()

            if self._is_new:
                title = user_input.get(CONF_NAME, DEFAULT_NAME)
                user_input.pop(CONF_NAME, None)
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        schema = self._build_schema(self.options, self._available_skills)

        if self._is_new:
            schema = {vol.Optional(CONF_NAME, default=DEFAULT_NAME): str, **schema}

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            final_data = {**(self._temp_data or {}), **user_input}
            if self._is_new:
                title = final_data.get(CONF_NAME, DEFAULT_NAME)
                final_data.pop(CONF_NAME, None)
                return self.async_create_entry(title=title, data=final_data)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=final_data,
            )

        schema: dict[Any, Any] = {
            vol.Optional(
                CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_TOP_P, default=DEFAULT_TOP_P
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_REQUEST_TIMEOUT, default=DEFAULT_REQUEST_TIMEOUT
            ): NumberSelector(NumberSelectorConfig(min=10, max=300, step=5, unit_of_measurement="s")),
            vol.Optional(
                CONF_SHORTEN_TOOL_CALL_ID, default=DEFAULT_SHORTEN_TOOL_CALL_ID
            ): BooleanSelector(),
            vol.Optional(
                CONF_SCHEMA_STRICT, default=DEFAULT_SCHEMA_STRICT
            ): BooleanSelector(),
            vol.Optional(
                CONF_HIDE_THINKING, default=DEFAULT_HIDE_THINKING
            ): BooleanSelector(),
            vol.Optional(
                CONF_FALLBACK_MODEL, default=DEFAULT_FALLBACK_MODEL
            ): str,
        }

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def _async_get_skills(self) -> list[dict[str, Any]]:
        skill_manager = await SkillManager.async_get_instance(self.hass)
        return [
            {"name": skill.name, "description": skill.description}
            for skill in skill_manager.get_all_skills()
        ]

    def _build_schema(
        self, options: dict[str, Any], skills: list[dict[str, Any]] | None
    ) -> dict:
        default_skills: list[str] = []
        if self._is_new and CONF_SKILLS not in options and skills:
            default_skills = [s["name"] for s in skills]
        current_skills = options.get(CONF_SKILLS, default_skills)

        schema: dict = {
            vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): TemplateSelector(),
            vol.Optional(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL): str,
            vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): int,
            vol.Optional(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            ): int,
            vol.Optional(CONF_SKILLS, default=current_skills): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=s["name"], label=s["name"])
                        for s in (skills or [])
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_FUNCTION_TOOLS, default=DEFAULT_CONF_FUNCTION_TOOLS_STR
            ): TemplateSelector(),
            vol.Optional(CONF_CONTEXT_THRESHOLD, default=DEFAULT_CONTEXT_THRESHOLD): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY, default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=s["key"], label=s["label"])
                        for s in CONTEXT_TRUNCATE_STRATEGIES
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_ADVANCED_OPTIONS, default=DEFAULT_ADVANCED_OPTIONS
            ): BooleanSelector(),
        }

        if not skills:
            schema = {
                k: v
                for k, v in schema.items()
                if not (isinstance(k, vol.Optional) and k.schema == CONF_SKILLS)
            }

        return schema


class UniversalLLMAITaskSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for AI Task subentries."""

    options: dict[str, Any]
    _temp_data: dict[str, Any] | None = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self.options = dict(DEFAULT_AI_TASK_OPTIONS)
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if user_input.get(CONF_ADVANCED_OPTIONS, False):
                self._temp_data = user_input
                return await self.async_step_advanced()
            if self._is_new:
                title = user_input.get(CONF_NAME, DEFAULT_AI_TASK_NAME)
                user_input.pop(CONF_NAME, None)
                return self.async_create_entry(title=title, data=user_input)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=user_input,
            )

        schema: dict = {}
        if self._is_new:
            schema[vol.Optional(CONF_NAME, default=DEFAULT_AI_TASK_NAME)] = str
        schema.update(
            {
                vol.Optional(CONF_CHAT_MODEL, default=DEFAULT_CHAT_MODEL): str,
                vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): int,
                vol.Optional(
                    CONF_ADVANCED_OPTIONS, default=DEFAULT_ADVANCED_OPTIONS
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is not None:
            final_data = {**(self._temp_data or {}), **user_input}
            if self._is_new:
                title = final_data.get(CONF_NAME, DEFAULT_AI_TASK_NAME)
                final_data.pop(CONF_NAME, None)
                return self.async_create_entry(title=title, data=final_data)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=final_data,
            )

        schema: dict[Any, Any] = {
            vol.Optional(
                CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_TOP_P, default=DEFAULT_TOP_P
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_REQUEST_TIMEOUT, default=DEFAULT_REQUEST_TIMEOUT
            ): NumberSelector(NumberSelectorConfig(min=10, max=300, step=5, unit_of_measurement="s")),
            vol.Optional(
                CONF_FALLBACK_MODEL, default=DEFAULT_FALLBACK_MODEL
            ): str,
        }
        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), self.options
            ),
        )
