"""
Role-Based Model Configuration
==============================
Centralizes model selection for coordinator, executor, verifier, formatter,
and related runtime roles. Supports simple presets plus per-role overrides.
"""

from __future__ import annotations

import os
import json
from urllib.parse import urlparse
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage

load_dotenv(override=False)

ROLE_MODEL_ENV: dict[str, str] = {
    "coordinator": "COORDINATOR_MODEL",
    "direct": "DIRECT_MODEL",
    "executor": "EXECUTOR_MODEL",
    "verifier": "VERIFIER_MODEL",
    "formatter": "FORMATTER_MODEL",
    "reflector": "REFLECTOR_MODEL",
}

LOCAL_BACKEND_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}

# Nebius token factory and similar hosted vLLM backends that
# reject request-level chat templates / native tool calling
_PROMPT_TOOL_HOSTS = {"api.tokenfactory.nebius.com", "api.studio.nebius.ai"}


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
            "coordinator": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "direct": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "executor": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "verifier": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "formatter": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "reflector": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        },
        "balanced": {
            "coordinator": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "direct": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "executor": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "verifier": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "formatter": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "reflector": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        },
        "score_max": {
            "coordinator": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "direct": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "executor": "deepseek-ai/DeepSeek-V3.2",
            "verifier": "deepseek-ai/DeepSeek-V3.2",
            "formatter": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "reflector": "meta-llama/Llama-3.3-70B-Instruct-fast",
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


def _extract_json_payload(text: str) -> str:
    """Extract the outermost JSON object from a model response."""
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return content[start:end + 1]


def _structured_output_mode(role: str) -> str:
    """Choose how structured outputs should be requested for a role."""
    override = os.getenv("STRUCTURED_OUTPUT_MODE", "auto").strip().lower()
    if override in {"native", "local_json"}:
        return override

    base_url = get_client_kwargs(role).get("base_url", "")
    if not base_url:
        return "native"

    host = (urlparse(base_url).hostname or "").lower()
    if host in LOCAL_BACKEND_HOSTS:
        return "local_json"
    return "native"


def _tool_call_mode(role: str) -> str:
    """Choose how tool calling should work for a role.

    Returns:
        'native'  – use llm.bind_tools() (OpenAI-native)
        'prompt'  – inject tool descriptions into the system prompt
                    and parse tool calls from text output
    """
    override = os.getenv("TOOL_CALL_MODE", "auto").strip().lower()
    if override in {"native", "prompt"}:
        return override

    # Auto-detect: Nebius profiles always need prompt mode
    profile = _current_model_profile()
    if profile in ("cheap", "balanced", "score_max"):
        return "prompt"

    base_url = get_client_kwargs(role).get("base_url", "")
    if not base_url:
        return "native"

    host = (urlparse(base_url).hostname or "").lower()
    if host in LOCAL_BACKEND_HOSTS or host in _PROMPT_TOOL_HOSTS:
        return "prompt"
    return "native"


def _base_url_host(role: str) -> str:
    base_url = get_client_kwargs(role).get("base_url", "")
    return (urlparse(base_url).hostname or "").lower()


def startup_compatibility_warnings() -> list[str]:
    """Return startup warnings for risky local backend configurations."""
    warnings: list[str] = []

    executor_host = _base_url_host("executor")
    if executor_host in LOCAL_BACKEND_HOSTS:
        warnings.append(
            "Executor is configured to use a localhost OpenAI-compatible backend. "
            "If this is vLLM, the backend must support tool-calling requests and may need "
            "--trust-request-chat-template plus the appropriate tool-calling flags. "
            "If benchmark runs fail, move EXECUTOR_* to a reliable hosted model first."
        )

    for role in ("coordinator", "verifier"):
        host = _base_url_host(role)
        if host in LOCAL_BACKEND_HOSTS and _structured_output_mode(role) != "local_json":
            warnings.append(
                f"{role.capitalize()} is using a localhost backend with native structured output. "
                "Set STRUCTURED_OUTPUT_MODE=local_json unless that backend explicitly supports "
                "provider-native structured output."
            )

    return warnings


def invoke_structured_output(
    role: str,
    schema: type,
    messages: list[BaseMessage],
    **kwargs: Any,
):
    """Invoke a model and return a parsed Pydantic object.

    Native structured output is used when supported. For local vLLM-style
    endpoints, this falls back to explicit JSON prompting plus local parsing.
    """
    model_name = get_model_name(role)
    llm = ChatOpenAI(
        model=model_name,
        **get_client_kwargs(role),
        **kwargs,
    )

    if _structured_output_mode(role) == "native":
        parsed = llm.with_structured_output(schema).invoke(messages)
        return parsed, model_name

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
    json_instruction = (
        "Return ONLY valid JSON matching this schema. "
        "Do not include markdown fences or extra commentary.\n"
        f"JSON_SCHEMA={schema_json}"
    )
    fallback_messages = [SystemMessage(content=json_instruction)] + messages
    response = llm.invoke(fallback_messages)
    payload = _extract_json_payload(str(response.content or ""))
    parsed = schema.model_validate_json(payload)
    return parsed, model_name


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
