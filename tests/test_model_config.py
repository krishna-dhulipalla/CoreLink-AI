import importlib
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _reload_model_config():
    import agent.model_config as model_config

    return importlib.reload(model_config)


class TestModelConfig:
    def test_custom_role_override_wins(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROFILE", "score_max")
        monkeypatch.setenv("EXECUTOR_MODEL", "custom-executor-model")
        model_config = _reload_model_config()

        assert model_config.get_model_name("executor") == "custom-executor-model"

    def test_profile_defaults_apply_without_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_PROFILE", "balanced")
        monkeypatch.delenv("COORDINATOR_MODEL", raising=False)
        monkeypatch.delenv("EXECUTOR_MODEL", raising=False)
        model_config = _reload_model_config()

        assert model_config.get_model_name("coordinator")
        assert model_config.get_model_name("executor")
        assert model_config.get_model_name("coordinator") != model_config._current_default_model()

    def test_role_specific_client_kwargs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "global-key")
        monkeypatch.delenv("VERIFIER_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("VERIFIER_OPENAI_BASE_URL", "https://example.test/v1")
        model_config = _reload_model_config()

        kwargs = model_config.get_client_kwargs("verifier")
        assert kwargs["api_key"]
        assert kwargs["base_url"] == "https://example.test/v1"

    def test_executor_local_backend_warning(self, monkeypatch):
        monkeypatch.setenv("EXECUTOR_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        model_config = _reload_model_config()

        warnings = model_config.startup_compatibility_warnings()
        assert any("Executor is configured to use a localhost" in warning for warning in warnings)

    def test_localhost_structured_native_warning(self, monkeypatch):
        monkeypatch.setenv("COORDINATOR_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("STRUCTURED_OUTPUT_MODE", "native")
        model_config = _reload_model_config()

        warnings = model_config.startup_compatibility_warnings()
        assert any("Coordinator is using a localhost backend" in warning for warning in warnings)

    def test_extract_json_payload_handles_unclosed_think_prefix(self, monkeypatch):
        model_config = _reload_model_config()
        payload = model_config._extract_json_payload(
            '<think>\nplanning...\n{"layers":["react_reason"],"task_type":"legal"}'
        )
        assert '"task_type":"legal"' in payload
