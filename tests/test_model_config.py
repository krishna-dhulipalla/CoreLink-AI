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
        model_config = _reload_model_config()

        assert model_config.get_model_name("coordinator") == "gpt-4.1-mini"
        assert model_config.get_model_name("executor") == "gpt-4o"

    def test_role_specific_client_kwargs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "global-key")
        monkeypatch.delenv("VERIFIER_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("VERIFIER_OPENAI_BASE_URL", "https://example.test/v1")
        model_config = _reload_model_config()

        kwargs = model_config.get_client_kwargs("verifier")
        assert kwargs["api_key"] == "global-key"
        assert kwargs["base_url"] == "https://example.test/v1"
