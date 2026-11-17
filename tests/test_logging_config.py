
import json
import logging

from logging_config import JsonFormatter, configure_logging


def test_json_formatter():

    record = logging.LogRecord(
        name="research_assistant.ollama",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Retrying Ollama connection",
        args=(),
        exc_info=None
    )

    record.event = "ollama_retry"
    record.model = "qwen3:4b-instruct"
    record.attempt = 1
    record.reason = "connection_timeout"

    formatter = JsonFormatter()

    json_text = formatter.format(record)

    data = json.loads(json_text)

    assert data["level"] == "WARNING"
    assert data["event"] == "ollama_retry"
    assert data["model"] == "qwen3:4b-instruct"
    assert data["attempt"] == 1
    assert data["reason"] == "connection_timeout"
    assert "timestamp" in data



def test_logging_writes_json_file(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    logger = configure_logging()

    try:
        logger.info(
            "Research assistant started",
            extra={
                "event": "application_start",
                "model": "qwen3:4b-instruct"
            }
        )

        for handler in logger.handlers:
            handler.flush()

        log_file = tmp_path / "results" / "events.jsonl"

        assert log_file.exists()

        lines = log_file.read_text(
            encoding="utf-8"
        ).splitlines()

        data = json.loads(lines[-1])

        assert data["event"] == "application_start"
        assert data["model"] == "qwen3:4b-instruct"
        assert data["level"] == "INFO"

    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()