"""
Role-Based Model Configuration
==============================
Centralizes model selection for coordinator, executor, verifier, formatter,
and related runtime roles. Supports simple presets plus per-role overrides.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=False)

ROLE_MODEL_ENV: dict[str, str] = {
    "coordinator": "COORDINATOR_MODEL",
    "direct": "DIRECT_MODEL",
    "executor": "EXECUTOR_MODEL",
    "verifier": "VERIFIER_MODEL",
    "formatter": "FORMATTER_MODEL",
    "reflector": "REFLECTOR_MODEL",
}


def _role_env_prefix(role: str) -> str:
    return role.upper()


def _current_default_model() -> str:
    return os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def _current_model_profile() -> str:
    return os.getenv("MODEL_PROFILE", "custom").strip().lower()


def _profile_models() -> dict[str, dict[str, str]]:
    default_model = _current_default_model()
    return {
        "custom": {},
        "oss_debug": {
            "coordinator": default_model,
            "direct": default_model,
            "executor": default_model,
            "verifier": default_model,
            "formatter": default_model,
            "reflector": default_model,
        },
        "cheap": {
            "coordinator": "gpt-4o-mini",
            "direct": "gpt-4o-mini",
            "executor": default_model,
            "verifier": "gpt-4o-mini",
            "formatter": "gpt-4o-mini",
            "reflector": "gpt-4o-mini",
        },
        "balanced": {
            "coordinator": "gpt-4.1-mini",
            "direct": "gpt-4o-mini",
            "executor": "gpt-4o",
            "verifier": "gpt-4.1-mini",
            "formatter": "gpt-4o-mini",
            "reflector": "gpt-4o-mini",
        },
        "score_max": {
            "coordinator": "gpt-4.1-mini",
            "direct": "gpt-4.1-mini",
            "executor": "gpt-4.1",
            "verifier": "gpt-4.1-mini",
            "formatter": "gpt-4.1-mini",
            "reflector": "gpt-4.1-mini",
        },
    }


def get_model_name(role: str) -> str:
    """Resolve the model name for a runtime role."""
    env_name = ROLE_MODEL_ENV.get(role)
    if env_name:
        override = os.getenv(env_name)
        if override:
            return override

    profile_models = _profile_models().get(_current_model_profile(), {})
    if role in profile_models:
        return profile_models[role]

    return _current_default_model()


def get_client_kwargs(role: str) -> dict[str, Any]:
    """Resolve shared or role-specific OpenAI-compatible client settings."""
    prefix = _role_env_prefix(role)
    api_key = os.getenv(f"{prefix}_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(f"{prefix}_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def build_chat_model(role: str, **kwargs: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given runtime role."""
    return ChatOpenAI(
        model=get_model_name(role),
        **get_client_kwargs(role),
        **kwargs,
    )


def primary_runtime_model() -> str:
    """Main model label for run-level summaries and defaults."""
    return get_model_name("executor")
