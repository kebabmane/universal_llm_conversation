"""Custom exceptions for Universal LLM Conversation."""

from homeassistant.exceptions import HomeAssistantError


class FunctionLoadFailed(HomeAssistantError):
    """Failed to load custom functions."""


class FunctionNotFound(HomeAssistantError):
    """Function not found in registry."""


class InvalidFunction(HomeAssistantError):
    """Invalid function configuration."""


class TokenLengthExceededError(HomeAssistantError):
    """Response exceeded token limit."""
