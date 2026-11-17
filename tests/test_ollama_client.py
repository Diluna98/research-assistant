
from ollama_client import post_chat
import pytest
import requests
from ollama_client import post_chat, chat_or_error

def test_chat_response_without_ollama(monkeypatch):

    class FakeResponse:

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": "This is a fake answer."
                }
            }

    def fake_post(url, json, timeout):
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json["model"] == "qwen3:4b-instruct"
        return FakeResponse()

    monkeypatch.setattr(
        "ollama_client.requests.post",
        fake_post
    )

    payload = {
        "model": "qwen3:4b-instruct",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    }

    result = post_chat(payload)

    assert result["message"]["content"] == "This is a fake answer."



def test_chat_handles_http_400(monkeypatch):

    class FakeErrorResponse:

        def raise_for_status(self):
            raise requests.HTTPError(
                "400 Client Error: Bad Request"
            )

    def fake_post(url, json, timeout):
        return FakeErrorResponse()

    monkeypatch.setattr(
        "ollama_client.requests.post",
        fake_post
    )

    payload = {
        "model": "qwen3:4b-instruct",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "stream": False
    }

    with pytest.raises(requests.HTTPError):
        post_chat(payload)


def test_error_message_is_displayed(monkeypatch, capsys):

    response = requests.Response()
    response.status_code = 400
    response._content = b'{"error": "Invalid context length"}'

    def fake_post_chat(payload):
        raise requests.HTTPError(
            "400 Client Error",
            response=response
        )

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({"model": "qwen3:4b-instruct"})

    output = capsys.readouterr().out

    assert result is None
    assert "HTTP status: 400" in output
    assert "Invalid context length" in output


def test_connection_failure(monkeypatch, capsys):

    def fake_post_chat(payload):
        raise requests.ConnectionError(
            "Connection refused"
        )

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({
        "model": "qwen3:4b-instruct"
    })

    output = capsys.readouterr().out

    assert result is None
    assert "Could not connect to Ollama" in output



def test_chat_timeout(monkeypatch, capsys):

    def fake_post_chat(payload):
        raise requests.ReadTimeout(
            "Ollama took too long to respond"
        )

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({
        "model": "qwen3:4b-instruct"
    })

    output = capsys.readouterr().out

    assert result is None
    assert "Ollama request timed out" in output


def test_retries_after_connection_timeout(monkeypatch):

    attempts = 0

    def fake_post_chat(payload):
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            raise requests.ConnectTimeout(
                "First connection attempt failed"
            )

        return {
            "message": {
                "content": "Connection successful"
            }
        }

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({
        "model": "qwen3:4b-instruct"
    })

    assert attempts == 2

    assert result["message"]["content"] == "Connection successful"



def test_stops_after_two_connection_timeouts(monkeypatch, capsys):

    attempts = 0

    def fake_post_chat(payload):
        nonlocal attempts

        attempts += 1

        raise requests.ConnectTimeout(
            "Ollama is unavailable"
        )

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({
        "model": "qwen3:4b-instruct"
    })

    output = capsys.readouterr().out

    assert attempts == 2
    assert result is None
    assert "Could not connect to Ollama" in output



def test_retry_creates_log(monkeypatch, caplog):

    attempts = 0

    def fake_post_chat(payload):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise requests.ConnectTimeout("Temporary failure")

        return {"message": {"content": "Success"}}

    monkeypatch.setattr(
        "ollama_client.post_chat",
        fake_post_chat
    )

    result = chat_or_error({
        "model": "qwen3:4b-instruct"
    })

    assert result["message"]["content"] == "Success"

    retry_logs = [
        record for record in caplog.records
        if getattr(record, "event", None) == "ollama_retry"
    ]

    assert len(retry_logs) == 1
    assert retry_logs[0].attempt == 1
    assert retry_logs[0].model == "qwen3:4b-instruct"



def test_chat_uses_custom_configuration(monkeypatch):

    monkeypatch.setenv(
        "RA_OLLAMA_URL",
        "http://127.0.0.1:9999"
    )

    monkeypatch.setenv(
        "RA_TIMEOUT",
        "30"
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "OK"}}

    def fake_post(url, json, timeout):
        assert url == "http://127.0.0.1:9999/api/chat"
        assert timeout == 30

        return FakeResponse()

    monkeypatch.setattr(
        "ollama_client.requests.post",
        fake_post
    )

    result = post_chat({
        "model": "qwen3:4b-instruct"
    })

    assert result["message"]["content"] == "OK"