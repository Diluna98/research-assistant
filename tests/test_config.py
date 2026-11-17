
from config import load_config
import pytest
import config


@pytest.fixture(autouse=True)
def ignore_local_env_file(monkeypatch):
    monkeypatch.setattr(
        config,
        "load_dotenv",
        lambda *args, **kwargs: None
    )


def test_default_configuration(monkeypatch):

    monkeypatch.delenv("RA_MODEL", raising=False)
    monkeypatch.delenv("RA_OLLAMA_URL", raising=False)
    monkeypatch.delenv("RA_TIMEOUT", raising=False)

    config = load_config()

    assert config["model"] == "qwen3:4b-instruct"
    assert config["ollama_url"] == "http://127.0.0.1:11434"
    assert config["timeout"] == 180


def test_custom_model_configuration(monkeypatch):

    monkeypatch.setenv("RA_MODEL", "test-model")

    config = load_config()

    assert config["model"] == "test-model"