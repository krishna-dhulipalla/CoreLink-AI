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
            "coordinator": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "direct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "executor": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "verifier": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "formatter": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "reflector": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        },
        "balanced": {
            "coordinator": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "direct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "executor": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "verifier": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "formatter": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "reflector": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        },
        "score_max": {
            "coordinator": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "direct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "executor": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "verifier": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "formatter": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "reflector": "meta-llama/Meta-Llama-3.1-70B-Instruct",
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
    profile = _current_model_profile()
    is_nebius_profile = profile in ("cheap", "balanced", "score_max")

    api_key = os.getenv(f"{prefix}_OPENAI_API_KEY")
    base_url = os.getenv(f"{prefix}_OPENAI_BASE_URL")

    if not api_key:
        if is_nebius_profile and os.getenv("NEBIUS_API_KEY"):
            api_key = os.getenv("NEBIUS_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")

    if not base_url:
        if is_nebius_profile and os.getenv("NEBIUS_BASE_URL"):
            base_url = os.getenv("NEBIUS_BASE_URL")
        elif is_nebius_profile:
            # Fallback to the known Nebius URL if using a Nebius profile but only the key was provided
            base_url = "https://api.studio.nebius.ai/v1/"
        else:
            base_url = os.getenv("OPENAI_BASE_URL")

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
