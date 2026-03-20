import importlib
import os
import sys
from pydantic import BaseModel

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _reload_model_config():
    import agent.model_config as model_config

    return importlib.reload(model_config)


class TestModelConfig:
    class _TinySchema(BaseModel):
        task_profile: str = "general"

    def test_custom_role_override_wins(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROFILE", "score_max")
        monkeypatch.setenv("SOLVER_MODEL", "custom-solver-model")
        model_config = _reload_model_config()

        assert model_config.get_model_name("solver") == "custom-solver-model"
        assert model_config.get_model_name("executor") == "custom-solver-model"

    def test_legacy_role_override_still_works(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROFILE", "score_max")
        monkeypatch.delenv("SOLVER_MODEL", raising=False)
        monkeypatch.setenv("EXECUTOR_MODEL", "legacy-executor-model")
        model_config = _reload_model_config()

        assert model_config.get_model_name("solver") == "legacy-executor-model"

    def test_profile_defaults_apply_without_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROFILE", "balanced")
        monkeypatch.delenv("COORDINATOR_MODEL", raising=False)
        monkeypatch.delenv("EXECUTOR_MODEL", raising=False)
        monkeypatch.delenv("PROFILER_MODEL", raising=False)
        monkeypatch.delenv("SOLVER_MODEL", raising=False)
        model_config = _reload_model_config()

        assert model_config.get_model_name("profiler")
        assert model_config.get_model_name("solver")
        assert model_config.get_model_name("profiler") != model_config._current_default_model()

    def test_role_specific_client_kwargs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "global-key")
        monkeypatch.delenv("REVIEWER_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VERIFIER_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("REVIEWER_OPENAI_BASE_URL", "https://example.test/v1")
        model_config = _reload_model_config()

        kwargs = model_config.get_client_kwargs("reviewer")
        assert kwargs["api_key"]
        assert kwargs["base_url"] == "https://example.test/v1"

    def test_solver_local_backend_warning(self, monkeypatch):
        monkeypatch.setenv("SOLVER_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        model_config = _reload_model_config()

        warnings = model_config.startup_compatibility_warnings()
        assert any("Solver is configured to use a localhost" in warning for warning in warnings)

    def test_localhost_structured_native_warning(self, monkeypatch):
        monkeypatch.setenv("PROFILER_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("STRUCTURED_OUTPUT_MODE", "native")
        model_config = _reload_model_config()

        warnings = model_config.startup_compatibility_warnings()
        assert any("Profiler is using a localhost backend" in warning for warning in warnings)

    def test_extract_json_payload_handles_unclosed_think_prefix(self, monkeypatch):
        model_config = _reload_model_config()
        payload = model_config._extract_json_payload(
            '<think>\nplanning...\n{"layers":["react_reason"],"task_type":"legal"}'
        )
        assert '"task_type":"legal"' in payload

    def test_invoke_structured_output_uses_json_mode_and_disables_thinking_for_nebius(self, monkeypatch):
        model_config = _reload_model_config()
        monkeypatch.setenv("PROFILER_OPENAI_BASE_URL", "https://api.studio.nebius.ai/v1/")

        captured = {}

        class _FakeResponse:
            content = '{"task_profile":"finance_options"}'

        class _FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def invoke(self, messages):
                captured["messages"] = messages
                return _FakeResponse()

            def with_structured_output(self, schema):
                raise AssertionError("native structured output should not be used for Nebius local_json mode")

        monkeypatch.setattr(model_config, "ChatOpenAI", _FakeChatOpenAI)

        parsed, _ = model_config.invoke_structured_output(
            "profiler",
            self._TinySchema,
            [],
            temperature=0,
            max_tokens=10,
        )

        assert parsed.task_profile == "finance_options"
        assert captured["kwargs"]["model_kwargs"]["response_format"] == {"type": "json_object"}
        assert captured["kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    def test_invoke_structured_output_keeps_plain_local_json_for_localhost(self, monkeypatch):
        model_config = _reload_model_config()
        monkeypatch.setenv("PROFILER_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

        captured = {}

        class _FakeResponse:
            content = '{"task_profile":"finance_quant"}'

        class _FakeChatOpenAI:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def invoke(self, messages):
                return _FakeResponse()

            def with_structured_output(self, schema):
                raise AssertionError("native structured output should not be used for localhost local_json mode")

        monkeypatch.setattr(model_config, "ChatOpenAI", _FakeChatOpenAI)

        parsed, _ = model_config.invoke_structured_output(
            "profiler",
            self._TinySchema,
            [],
            temperature=0,
            max_tokens=10,
        )

        assert parsed.task_profile == "finance_quant"
        assert "model_kwargs" not in captured["kwargs"]
        assert "extra_body" not in captured["kwargs"]
