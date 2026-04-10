"""
Role-Based Model Configuration
==============================
Centralizes model selection for coordinator, executor, verifier, formatter,
and related runtime roles. Supports simple presets plus per-role overrides.
"""

from __future__ import annotations

import os
import json
import re
import time
from urllib.parse import urlparse
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI as _BaseChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage

from agent.langsmith_env import normalize_langsmith_env
from agent.cost import get_active_cost_tracker

load_dotenv(override=False)
normalize_langsmith_env()

_ROLE_ALIASES: dict[str, str] = {
    "coordinator": "profiler",
    "profiler": "profiler",
    "direct": "direct",
    "executor": "solver",
    "solver": "solver",
    "verifier": "reviewer",
    "reviewer": "reviewer",
    "formatter": "adapter",
    "adapter": "adapter",
    "reflector": "reflection",
    "reflection": "reflection",
}

ROLE_MODEL_ENV: dict[str, str] = {
    "profiler": "PROFILER_MODEL",
    "direct": "DIRECT_MODEL",
    "solver": "SOLVER_MODEL",
    "reviewer": "REVIEWER_MODEL",
    "adapter": "ADAPTER_MODEL",
    "reflection": "REFLECTION_MODEL",
}

_LEGACY_ROLE_MODEL_ENV: dict[str, str] = {
    "profiler": "COORDINATOR_MODEL",
    "solver": "EXECUTOR_MODEL",
    "reviewer": "VERIFIER_MODEL",
    "adapter": "FORMATTER_MODEL",
    "reflection": "REFLECTOR_MODEL",
}

_ROLE_CLIENT_PREFIX: dict[str, str] = {
    "profiler": "PROFILER",
    "direct": "DIRECT",
    "solver": "SOLVER",
    "reviewer": "REVIEWER",
    "adapter": "ADAPTER",
    "reflection": "REFLECTION",
}

_LEGACY_ROLE_CLIENT_PREFIX: dict[str, str] = {
    "profiler": "COORDINATOR",
    "solver": "EXECUTOR",
    "reviewer": "VERIFIER",
    "adapter": "FORMATTER",
    "reflection": "REFLECTOR",
}

LOCAL_BACKEND_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}

# Nebius token factory and similar hosted vLLM backends that
# reject request-level chat templates / native tool calling
_PROMPT_TOOL_HOSTS = {"api.tokenfactory.nebius.com", "api.studio.nebius.ai"}
_PROFILE_ALIASES: dict[str, str] = {
    "competition_gpt": "officeqa",
    "officeqa_gpt": "officeqa",
}


class ChatOpenAI(_BaseChatOpenAI):
    """ChatOpenAI wrapper with provider compatibility shims.

    Nebius Token Factory currently accepts `max_tokens` for all tested chat
    models, while at least `deepseek-ai/DeepSeek-V3-0324-fast` rejects
    `max_completion_tokens`. LangChain's OpenAI client rewrites `max_tokens`
    into `max_completion_tokens` by default, so we remap it back on Nebius
    hosts before the request is sent.
    """

    def _uses_legacy_max_tokens(self) -> bool:
        base = getattr(self, "base_url", None) or getattr(self, "openai_api_base", "")
        host = (urlparse(str(base)).hostname or "").lower()
        return host in _PROMPT_TOOL_HOSTS

    @property
    def _default_params(self) -> dict[str, Any]:
        params = super()._default_params
        if self._uses_legacy_max_tokens() and "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(
        self,
        input_,
        *,
        stop=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._uses_legacy_max_tokens() and "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


def _role_env_prefix(role: str) -> str:
    canonical = _canonical_role(role)
    return _ROLE_CLIENT_PREFIX.get(canonical, canonical.upper())


def _canonical_role(role: str) -> str:
    return _ROLE_ALIASES.get(role, role)


def _role_model_env_names(role: str) -> list[str]:
    canonical = _canonical_role(role)
    names: list[str] = []
    current = ROLE_MODEL_ENV.get(canonical)
    legacy = _LEGACY_ROLE_MODEL_ENV.get(canonical)
    if current:
        names.append(current)
    if legacy and legacy not in names:
        names.append(legacy)
    return names


def _role_client_prefixes(role: str) -> list[str]:
    canonical = _canonical_role(role)
    prefixes: list[str] = []
    current = _ROLE_CLIENT_PREFIX.get(canonical)
    legacy = _LEGACY_ROLE_CLIENT_PREFIX.get(canonical)
    if current:
        prefixes.append(current)
    if legacy and legacy not in prefixes:
        prefixes.append(legacy)
    return prefixes


def _current_default_model() -> str:
    return os.getenv("MODEL_NAME", "openai/gpt-oss-20b")


def _current_model_profile() -> str:
    raw = os.getenv("MODEL_PROFILE", "custom").strip().lower()
    return _PROFILE_ALIASES.get(raw, raw)


def _benchmark_name() -> str:
    return os.getenv("BENCHMARK_NAME", "").strip().lower()


def _effective_model_profile() -> str:
    profile = _current_model_profile()
    if profile == "custom" and _benchmark_name() == "officeqa":
        return "officeqa"
    return profile


def _profile_models() -> dict[str, dict[str, str]]:
    default_model = _current_default_model()
    return {
        "custom": {},
        "oss_debug": {
            "profiler": default_model,
            "direct": default_model,
            "solver": default_model,
            "reviewer": default_model,
            "adapter": default_model,
            "reflection": default_model,
        },
        "cheap": {
            "profiler": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "direct": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "solver": "Qwen/Qwen3-32B-fast",
            "reviewer": "Qwen/Qwen3-32B-fast",
            "adapter": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "reflection": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
        },
        "balanced": {
            "profiler": "Qwen/Qwen3-32B-fast",
            "direct": "Qwen/Qwen3-32B-fast",
            "solver": "deepseek-ai/DeepSeek-V3.2",
            "reviewer": "Qwen/Qwen3-32B-fast",
            "adapter": "meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
            "reflection": "Qwen/Qwen3-32B-fast",
        },
        "score_max": {
            "profiler": "Qwen/Qwen3-32B-fast",
            "direct": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "solver": "deepseek-ai/DeepSeek-V3.2",
            "reviewer": "meta-llama/Llama-3.3-70B-Instruct-fast",
            "adapter": "Qwen/Qwen3-32B-fast",
            "reflection": "Qwen/Qwen3-32B-fast",
        },
        "officeqa": {
            "profiler": "Qwen/Qwen3-32B-fast",
            "direct": "Qwen/Qwen3-32B-fast",
            "solver": "deepseek-ai/DeepSeek-V3.2",
            "reviewer": "Qwen/Qwen3-32B-fast",
            "adapter": "Qwen/Qwen3-32B-fast",
            "reflection": "Qwen/Qwen3-32B-fast",
        },
    }


def get_model_name(role: str) -> str:
    """Resolve the model name for a runtime role."""
    canonical = _canonical_role(role)
    for env_name in _role_model_env_names(canonical):
        override = os.getenv(env_name)
        if override:
            return override

    profile_models = _profile_models().get(_effective_model_profile(), {})
    if canonical in profile_models:
        return profile_models[canonical]

    return _current_default_model()


def get_model_name_for_task(
    role: str,
    *,
    execution_mode: str = "",
    task_family: str = "",
    prompt_tokens: int = 0,
    answer_mode: str = "",
    analysis_modes: list[str] | None = None,
) -> str:
    """Resolve a model name with optional task-aware long-context overrides."""
    canonical = _canonical_role(role)
    env_candidates: list[str] = []
    analysis_mode_set = {str(item).strip().lower() for item in analysis_modes or [] if str(item).strip()}

    if canonical == "solver":
        if execution_mode == "document_grounded_analysis":
            env_candidates.extend(["DOCUMENT_SOLVER_MODEL", "LONG_CONTEXT_SOLVER_MODEL"])
        elif execution_mode == "retrieval_augmented_analysis":
            env_candidates.extend(["RETRIEVAL_SOLVER_MODEL", "LONG_CONTEXT_SOLVER_MODEL"])
        elif task_family == "legal_transactional" and prompt_tokens >= 5000:
            env_candidates.append("LONG_CONTEXT_SOLVER_MODEL")
        if answer_mode in {"grounded_synthesis", "hybrid_grounded"}:
            env_candidates.extend(["SYNTHESIS_HEAVY_SOLVER_MODEL", "AMBIGUITY_SOLVER_MODEL"])
        if analysis_mode_set.intersection({"statistical_analysis", "time_series_forecasting", "risk_metric"}):
            env_candidates.extend(["FINANCIAL_REASONING_SOLVER_MODEL", "SYNTHESIS_HEAVY_SOLVER_MODEL", "AMBIGUITY_SOLVER_MODEL"])
    elif canonical == "reviewer":
        if execution_mode == "document_grounded_analysis":
            env_candidates.append("DOCUMENT_REVIEWER_MODEL")
        elif execution_mode == "retrieval_augmented_analysis":
            env_candidates.append("RETRIEVAL_REVIEWER_MODEL")
        if answer_mode in {"grounded_synthesis", "hybrid_grounded"} or analysis_mode_set.intersection({"statistical_analysis", "time_series_forecasting", "risk_metric"}):
            env_candidates.extend(["SYNTHESIS_HEAVY_REVIEWER_MODEL", "AMBIGUITY_REVIEWER_MODEL"])

    if prompt_tokens >= 6000:
        env_candidates.extend(
            [
                f"LONG_CONTEXT_{canonical.upper()}_MODEL",
                "LONG_CONTEXT_MODEL",
            ]
        )

    seen: set[str] = set()
    for env_name in env_candidates:
        if env_name in seen:
            continue
        seen.add(env_name)
        override = os.getenv(env_name, "").strip()
        if override:
            return override
    return get_model_name(role)


def get_model_name_for_officeqa_control(
    category: str,
    *,
    answer_mode: str = "",
    analysis_modes: list[str] | None = None,
) -> str:
    category_key = str(category or "").strip().lower()
    env_candidates = {
        "semantic_plan_llm": ["OFFICEQA_SEMANTIC_PLAN_MODEL", "SEMANTIC_PLAN_MODEL"],
        "retrieval_rerank_llm": ["OFFICEQA_RETRIEVAL_RERANK_MODEL", "RETRIEVAL_RERANK_MODEL"],
        "table_rerank_llm": ["OFFICEQA_TABLE_RERANK_MODEL", "TABLE_RERANK_MODEL"],
        "repair_llm": ["OFFICEQA_REPAIR_MODEL", "REPAIR_MODEL"],
    }.get(category_key, [])
    for env_name in env_candidates:
        override = os.getenv(env_name, "").strip()
        if override:
            return override

    if category_key == "semantic_plan_llm":
        if answer_mode in {"grounded_synthesis", "hybrid_grounded"}:
            return get_model_name_for_task("solver", answer_mode=answer_mode, analysis_modes=analysis_modes)
        if {str(item).strip().lower() for item in analysis_modes or [] if str(item).strip()}.intersection({"statistical_analysis", "time_series_forecasting", "risk_metric"}):
            return get_model_name_for_task("solver", answer_mode=answer_mode, analysis_modes=analysis_modes)
        return get_model_name("profiler")
    if category_key == "retrieval_rerank_llm":
        return get_model_name_for_task("direct", answer_mode=answer_mode, analysis_modes=analysis_modes)
    if category_key == "table_rerank_llm":
        return get_model_name_for_task("direct", answer_mode=answer_mode, analysis_modes=analysis_modes)
    if category_key == "repair_llm":
        return get_model_name_for_task("solver", answer_mode=answer_mode, analysis_modes=analysis_modes)
    return get_model_name("profiler")


def get_model_runtime_kwargs_for_officeqa_control(
    category: str,
    *,
    answer_mode: str = "",
    analysis_modes: list[str] | None = None,
) -> dict[str, Any]:
    category_key = str(category or "").strip().lower()
    if category_key in {"retrieval_rerank_llm", "table_rerank_llm"}:
        return get_model_runtime_kwargs(
            "direct",
            answer_mode=answer_mode,
            analysis_modes=analysis_modes,
        )
    if category_key == "repair_llm":
        return get_model_runtime_kwargs(
            "solver",
            answer_mode=answer_mode,
            analysis_modes=analysis_modes,
        )
    if category_key == "semantic_plan_llm":
        return get_model_runtime_kwargs(
            "solver" if answer_mode in {"grounded_synthesis", "hybrid_grounded"} else "profiler",
            answer_mode=answer_mode,
            analysis_modes=analysis_modes,
        )
    return get_model_runtime_kwargs("profiler", answer_mode=answer_mode, analysis_modes=analysis_modes)


def _role_reasoning_effort_env_names(role: str) -> list[str]:
    canonical = _canonical_role(role)
    names = [f"{canonical.upper()}_REASONING_EFFORT"]
    if canonical == "solver":
        names.append("EXECUTOR_REASONING_EFFORT")
    elif canonical == "reviewer":
        names.append("VERIFIER_REASONING_EFFORT")
    elif canonical == "adapter":
        names.append("FORMATTER_REASONING_EFFORT")
    elif canonical == "reflection":
        names.append("REFLECTOR_REASONING_EFFORT")
    elif canonical == "profiler":
        names.append("COORDINATOR_REASONING_EFFORT")
    names.append("REASONING_EFFORT")
    return names


def _profile_reasoning_efforts() -> dict[str, dict[str, str]]:
    return {
        "officeqa": {
            "profiler": "low",
            "direct": "medium",
            "solver": "high",
            "reviewer": "medium",
            "adapter": "low",
            "reflection": "medium",
        },
    }


def get_model_runtime_kwargs(
    role: str,
    *,
    execution_mode: str = "",
    task_family: str = "",
    prompt_tokens: int = 0,
    answer_mode: str = "",
    analysis_modes: list[str] | None = None,
) -> dict[str, Any]:
    canonical = _canonical_role(role)
    model_name = get_model_name_for_task(
        canonical,
        execution_mode=execution_mode,
        task_family=task_family,
        prompt_tokens=prompt_tokens,
        answer_mode=answer_mode,
        analysis_modes=analysis_modes,
    )
    if not model_name.startswith("gpt-5"):
        return {}

    for env_name in _role_reasoning_effort_env_names(canonical):
        override = os.getenv(env_name, "").strip().lower()
        if override:
            return {"reasoning_effort": override}

    profile = _effective_model_profile()
    effort = _profile_reasoning_efforts().get(profile, {}).get(canonical, "")
    if canonical == "solver":
        if execution_mode in {"document_grounded_analysis", "retrieval_augmented_analysis"} and prompt_tokens >= 5000:
            effort = "high"
        elif task_family in {"analytical_reasoning", "legal_transactional"}:
            effort = "high"
        elif answer_mode in {"grounded_synthesis", "hybrid_grounded"}:
            effort = "high"
        elif {str(item).strip().lower() for item in analysis_modes or [] if str(item).strip()}.intersection({"statistical_analysis", "time_series_forecasting", "risk_metric"}):
            effort = "high"
        elif not effort:
            effort = "medium"
    elif canonical == "reviewer":
        if answer_mode in {"grounded_synthesis", "hybrid_grounded"}:
            effort = "medium"
    elif not effort:
        effort = "low"

    return {"reasoning_effort": effort} if effort else {}


def get_client_kwargs(role: str) -> dict[str, Any]:
    """Resolve shared or role-specific OpenAI-compatible client settings."""
    prefixes = _role_client_prefixes(role)
    profile = _effective_model_profile()
    is_nebius_profile = profile in ("cheap", "balanced", "score_max", "officeqa")

    api_key = None
    base_url = None
    for prefix in prefixes:
        api_key = os.getenv(f"{prefix}_OPENAI_API_KEY")
        if api_key:
            break
    for prefix in prefixes:
        base_url = os.getenv(f"{prefix}_OPENAI_BASE_URL")
        if base_url:
            break

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
    """Extract the first balanced JSON object from a model response.

    Handles weak-model artifacts such as markdown fences and stray `<think>`
    tags before the JSON body.
    """
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            content = "\n".join(lines[1:-1]).strip()

    # Qwen-style reasoning markup can wrap or prefix the JSON body.
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE).strip()

    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start:idx + 1]

    raise ValueError("No JSON object found in model response.")


def _structured_output_mode(role: str) -> str:
    """Choose how structured outputs should be requested for a role."""
    override = os.getenv("STRUCTURED_OUTPUT_MODE", "auto").strip().lower()
    if override in {"native", "local_json"}:
        return override

    base_url = get_client_kwargs(role).get("base_url", "")
    if not base_url:
        return "native"

    host = (urlparse(base_url).hostname or "").lower()
    if host in LOCAL_BACKEND_HOSTS or host in _PROMPT_TOOL_HOSTS:
        return "local_json"
    return "native"


def _supports_json_object_mode(role: str) -> bool:
    host = _base_url_host(role)
    return host in _PROMPT_TOOL_HOSTS


def _merge_model_init_kwargs(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key in {"model_kwargs", "extra_body"} and isinstance(value, dict):
            current = merged.get(key, {})
            if isinstance(current, dict):
                merged[key] = {**current, **value}
            else:
                merged[key] = dict(value)
        else:
            merged[key] = value
    return merged


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
    profile = _effective_model_profile()
    if profile in ("cheap", "balanced", "score_max", "officeqa"):
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

    solver_host = _base_url_host("solver")
    if solver_host in LOCAL_BACKEND_HOSTS:
        warnings.append(
            "Solver is configured to use a localhost OpenAI-compatible backend. "
            "If this is vLLM, the backend must support tool-calling requests and may need "
            "--trust-request-chat-template plus the appropriate tool-calling flags. "
            "If benchmark runs fail, move SOLVER_* or EXECUTOR_* to a reliable hosted model first."
        )

    for role, label in (("profiler", "Profiler"), ("reviewer", "Reviewer")):
        host = _base_url_host(role)
        if host in LOCAL_BACKEND_HOSTS and _structured_output_mode(role) != "local_json":
            warnings.append(
                f"{label} is using a localhost backend with native structured output. "
                "Set STRUCTURED_OUTPUT_MODE=local_json unless that backend explicitly supports "
                "provider-native structured output."
            )

    return warnings


def startup_model_summary() -> dict[str, str]:
    """Return the active startup model map for logging and diagnostics."""
    benchmark = _benchmark_name() or "default"
    profile = _effective_model_profile()
    summary = {
        "benchmark": benchmark,
        "profile": profile,
        "profiler": get_model_name("profiler"),
        "direct": get_model_name("direct"),
        "solver": get_model_name("solver"),
        "document_solver": get_model_name_for_task(
            "solver",
            execution_mode="document_grounded_analysis",
            task_family="document_qa",
        ),
        "reviewer": get_model_name("reviewer"),
        "document_reviewer": get_model_name_for_task(
            "reviewer",
            execution_mode="document_grounded_analysis",
            task_family="document_qa",
        ),
        "adapter": get_model_name("adapter"),
        "reflection": get_model_name("reflection"),
    }
    return summary


def invoke_structured_output(
    role: str,
    schema: type,
    messages: list[BaseMessage],
    model_name_override: str | None = None,
    runtime_kwargs_override: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Invoke a model and return a parsed Pydantic object.

    Native structured output is used when supported. For local vLLM-style
    endpoints, this falls back to explicit JSON prompting plus local parsing.
    """
    model_name = str(model_name_override or get_model_name(role))
    model_init_kwargs: dict[str, Any] = {
        "model": model_name,
        **get_client_kwargs(role),
        **(runtime_kwargs_override or get_model_runtime_kwargs(role)),
        **kwargs,
    }

    tracker = get_active_cost_tracker()

    def _usage_from_response(response: Any) -> tuple[int, int]:
        usage_metadata = dict(getattr(response, "usage_metadata", {}) or {})
        if usage_metadata:
            return int(usage_metadata.get("input_tokens", 0) or 0), int(usage_metadata.get("output_tokens", 0) or 0)
        response_metadata = dict(getattr(response, "response_metadata", {}) or {})
        token_usage = dict(response_metadata.get("token_usage", {}) or {})
        if token_usage:
            return int(token_usage.get("prompt_tokens", 0) or 0), int(token_usage.get("completion_tokens", 0) or 0)
        return 0, 0

    if _structured_output_mode(role) == "native":
        llm = ChatOpenAI(**model_init_kwargs)
        t0 = time.monotonic()
        success = True
        try:
            parsed = llm.with_structured_output(schema).invoke(messages)
        except Exception:
            success = False
            raise
        finally:
            if tracker is not None:
                tracker.record(
                    operator=f"{role}_structured_output",
                    model_name=model_name,
                    tokens_in=0,
                    tokens_out=0,
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    success=success,
                )
        return parsed, model_name

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
    json_instruction = (
        "Return ONLY valid JSON matching this schema. "
        "Do not include markdown fences or extra commentary.\n"
        f"JSON_SCHEMA={schema_json}"
    )
    fallback_messages = [SystemMessage(content=json_instruction)] + messages
    if _supports_json_object_mode(role):
        model_init_kwargs = _merge_model_init_kwargs(
            model_init_kwargs,
            {
                "model_kwargs": {"response_format": {"type": "json_object"}},
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )
    llm = ChatOpenAI(**model_init_kwargs)
    t0 = time.monotonic()
    success = True
    try:
        response = llm.invoke(fallback_messages)
    except Exception:
        success = False
        raise
    finally:
        if tracker is not None and not success:
            tracker.record(
                operator=f"{role}_structured_output",
                model_name=model_name,
                tokens_in=0,
                tokens_out=0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                success=False,
            )
    payload = _extract_json_payload(str(response.content or ""))
    parsed = schema.model_validate_json(payload)
    if tracker is not None and success:
        tokens_in, tokens_out = _usage_from_response(response)
        tracker.record(
            operator=f"{role}_structured_output",
            model_name=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            success=True,
        )
    return parsed, model_name


def build_chat_model(
    role: str,
    *,
    execution_mode: str = "",
    task_family: str = "",
    prompt_tokens: int = 0,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given runtime role."""
    return ChatOpenAI(
        model=get_model_name_for_task(
            role,
            execution_mode=execution_mode,
            task_family=task_family,
            prompt_tokens=prompt_tokens,
        ),
        **get_client_kwargs(role),
        **get_model_runtime_kwargs(
            role,
            execution_mode=execution_mode,
            task_family=task_family,
            prompt_tokens=prompt_tokens,
        ),
        **kwargs,
    )


def primary_runtime_model() -> str:
    """Main model label for run-level summaries and defaults."""
    return get_model_name("solver")
