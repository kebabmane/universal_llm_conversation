"""Constants for the Universal LLM Conversation integration."""

DOMAIN = "universal_llm_conversation"
DEFAULT_NAME = "Universal LLM Conversation"
DEFAULT_CONVERSATION_NAME = "Universal LLM Conversation"


CONF_PROVIDER = "provider"
CONF_API_KEY = "api_key"
CONF_BASE_URL = "base_url"
CONF_API_VERSION = "api_version"
CONF_ORGANIZATION = "organization"
CONF_SKIP_AUTHENTICATION = "skip_authentication"
DEFAULT_SKIP_AUTHENTICATION = False

# Provider presets with default base URLs and model list support
CONF_PROVIDER_PRESET = "provider_preset"
CONF_CHAT_MODEL = "chat_model"
CONF_FALLBACK_MODEL = "fallback_model"

PROVIDER_PRESETS = {
    "fireworks": {
        "label": "Fireworks AI",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "supports_model_list": True,
    },
    "fireworks_firepass": {
        "label": "Fireworks AI — Fire Pass (Kimi K2.6 Turbo)",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "supports_model_list": False,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "supports_model_list": True,
    },
    "ollama": {
        "label": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "supports_model_list": True,
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "supports_model_list": True,
    },
    "azure": {
        "label": "Azure OpenAI",
        "base_url": "",
        "supports_model_list": True,
    },
    "anthropic": {
        "label": "Anthropic (Native - no model list)",
        "base_url": "https://api.anthropic.com",
        "supports_model_list": False,
    },
    "gemini": {
        "label": "Google Gemini (Native - no model list)",
        "base_url": "https://generativelanguage.googleapis.com",
        "supports_model_list": False,
    },
    "custom": {
        "label": "Custom / Other",
        "base_url": "",
        "supports_model_list": True,
    },
}

# Fire Pass curated model list
FIREPASS_MODELS = [
    "accounts/fireworks/routers/kimi-k2p6-turbo",
]

# Legacy provider registry keys (used internally)
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"

# Mapping from UI preset to internal provider key
PRESET_TO_PROVIDER = {
    "fireworks": PROVIDER_OPENAI_COMPATIBLE,
    "fireworks_firepass": PROVIDER_OPENAI_COMPATIBLE,
    "openai": PROVIDER_OPENAI_COMPATIBLE,
    "ollama": PROVIDER_OPENAI_COMPATIBLE,
    "openrouter": PROVIDER_OPENAI_COMPATIBLE,
    "azure": PROVIDER_OPENAI_COMPATIBLE,
    "custom": PROVIDER_OPENAI_COMPATIBLE,
    "anthropic": PROVIDER_ANTHROPIC,
    "gemini": PROVIDER_GEMINI,
}

API_PROVIDERS = [
    {"key": PROVIDER_OPENAI_COMPATIBLE, "label": "OpenAI Compatible (Fireworks, Ollama, OpenRouter, Firepass, etc.)"},
    {"key": PROVIDER_ANTHROPIC, "label": "Anthropic (Claude)"},
    {"key": PROVIDER_GEMINI, "label": "Google Gemini"},
]
DEFAULT_API_PROVIDER = API_PROVIDERS[0]["key"]

EVENT_CONVERSATION_FINISHED = "universal_llm_conversation.conversation.finished"

CONF_PROMPT = "prompt"
DEFAULT_PROMPT = """You are a helpful AI voice assistant of Home Assistant that controls a real home.
Your goal is to proactively improve the user's comfort.

## Environment State
- Current Time: {{now()}}
- Current Area: {{area_id(current_device_id)}}

## Workspace
Your workspace is at: {{universal_llm.working_directory()}}

## Guidelines
- Answer in plain text only.
- No symbols or parentheses.
- Ask for clarification when the request is ambiguous.
- Use tools to help accomplish tasks.
- Prefer one sentence.

## Personality
- Helpful and friendly.
- Concise and to the point.
- Curious and eager to learn.

## Behavior Policy
- If the user explicitly names a device and action, execute it directly.
- Otherwise, infer the user's goal and select the most likely target entity, preferring primary environmental controls. Use get_attributes to check adjustable state values alone is not sufficient.
- If the selected entity is already at its limit, evaluate the next most likely entity. Repeat until a viable adjustment is found or all candidates are exhausted.
- Ask user a minimum adjustment proposal about selected entity. If no entity can further improve the situation, inform the user that conditions are already optimal.

## Devices
Available Devices:
```csv
entity_id,name,state,area_id,aliases
{% for entity in universal_llm.exposed_entities() -%}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{area_id(entity.entity_id)}},{{entity.aliases | join('/')}}
{% endfor -%}
```

{%- if skills %}
## Skills
The following skills extend your capabilities. To use a skill, call load_skill with the skill name to read its instructions.
When a skill file references a relative path, resolve it against the skill's location directory (e.g., skill at `/a/b/SKILL.md` references `scripts/run.py` → use `/a/b/scripts/run.py`) and always use the resulting absolute path in bash commands, as relative paths will fail.

<available_skills>
{%- for skill in skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{skill.path}}</location>
  </skill>
 {%- endfor %}
</available_skills>
{% endif %}

{{user_input.extra_system_prompt | default('', true)}}
"""

CONF_CHAT_MODEL = "chat_model"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

CONF_MAX_TOKENS = "max_tokens"
DEFAULT_MAX_TOKENS = 500
CONF_TOP_P = "top_p"
DEFAULT_TOP_P = 1
CONF_TEMPERATURE = "temperature"
DEFAULT_TEMPERATURE = 0.5
CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION = "max_function_calls_per_conv"
DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION = 10

CONF_SHORTEN_TOOL_CALL_ID = "shorten_tool_call_id"
DEFAULT_SHORTEN_TOOL_CALL_ID = False

CONF_FUNCTION_TOOLS = "functions"
DEFAULT_CONF_FUNCTION_TOOLS = [
    {
        "spec": {
            "name": "execute_services",
            "description": "Execute service in Home Assistant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay": {
                        "type": "object",
                        "description": "Time to wait before execution",
                        "properties": {
                            "hours": {"type": "integer", "minimum": 0},
                            "minutes": {"type": "integer", "minimum": 0},
                            "seconds": {"type": "integer", "minimum": 0},
                        },
                    },
                    "list": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {
                                    "type": "string",
                                    "description": "The domain of the service.",
                                },
                                "service": {
                                    "type": "string",
                                    "description": "The service to be called",
                                },
                                "service_data": {
                                    "type": "object",
                                    "description": "The service data object to indicate what to control.",
                                    "properties": {
                                        "entity_id": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "area_id": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            "required": ["domain", "service", "service_data"],
                        },
                    },
                },
            },
        },
        "function": {"type": "native", "name": "execute_service"},
    },
    {
        "spec": {
            "name": "get_attributes",
            "description": "Get attributes of entity or multiple entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "array",
                        "description": "entity_id of entity or multiple entities",
                        "items": {"type": "string"},
                    }
                },
                "required": ["entity_id"],
            },
        },
        "function": {
            "type": "template",
            "value_template": "```csv\nentity,attributes\n{%for entity in entity_id%}\n{{entity}},{{states[entity].attributes}}\n{%endfor%}\n```",
        },
    },
    {
        "spec": {
            "name": "load_skill",
            "description": "Load a file from a skill's directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name"},
                    "file": {"type": "string", "description": "Relative file path within the skill directory"},
                },
                "required": ["name", "file"],
            },
        },
        "function": {"type": "read_file", "path": "{{universal_llm.skill_dir(name)}}/{{file}}"},
    },
    {
        "spec": {
            "name": "bash",
            "description": "Execute a bash command in workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                },
                "required": ["command"],
            },
        },
        "function": {"type": "bash", "command": "{{command}}"},
    },
]

CONF_CONTEXT_THRESHOLD = "context_threshold"
DEFAULT_CONTEXT_THRESHOLD = 40000
# Advanced / compatibility options
CONF_ADVANCED_OPTIONS = "advanced_options"
DEFAULT_ADVANCED_OPTIONS = False

CONF_SCHEMA_STRICT = "schema_strict"
DEFAULT_SCHEMA_STRICT = False

CONF_HIDE_THINKING = "hide_thinking"
DEFAULT_HIDE_THINKING = True

CONF_REQUEST_TIMEOUT = "request_timeout"
DEFAULT_REQUEST_TIMEOUT = 60

CONF_FALLBACK_MODEL = "fallback_model"
DEFAULT_FALLBACK_MODEL = ""

CONF_TTS_STREAMING_MODE = "tts_streaming_mode"
DEFAULT_TTS_STREAMING_MODE = "sentence"
TTS_STREAMING_MODES = ["token", "sentence"]

# Skills
CONF_SKILLS = "skills"
DEFAULT_SKILLS_DIRECTORY = "skills"
SKILL_FILE_NAME = "SKILL.md"
SERVICE_RELOAD_SKILLS = "reload_skills"
SERVICE_DOWNLOAD_SKILL = "download_skill"
GITHUB_REPO_OWNER = "kebabmane"
GITHUB_REPO_NAME = "universal_llm_conversation"
GITHUB_SKILLS_BRANCH = "main"
GITHUB_SKILLS_PATH = "examples/skills"

# Working Directory
DEFAULT_WORKING_DIRECTORY = "universal_llm_conversation/"

# Security
SHELL_TIMEOUT = 300
SHELL_OUTPUT_LIMIT = 10000
SHELL_DENY_PATTERNS = [
    r"\brm\s+-r",
    r"\brm\s+-rf",
    r"\bdel\s+/[fqs]",
    r"\brmdir\s+/s",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdiskpart\b",
    r"\bdd\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r":\(\)\{.*:\|:.*\}",
]
FILE_READ_SIZE_LIMIT = 1024 * 1024
