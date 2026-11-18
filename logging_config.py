
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):

    def format(self, record):
        data = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "event": getattr(record, "event", None),
            "model": getattr(record, "model", None),
            "attempt": getattr(record, "attempt", None),
            "reason": getattr(record, "reason", None)
        }

        metrics = getattr(record, "metrics", None)
        if metrics is not None:
            data["metrics"] = metrics
        return json.dumps(data)


def configure_logging():
    Path("results").mkdir(exist_ok=True)

    logger = logging.getLogger("research_assistant")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(
            "results/events.jsonl",
            encoding="utf-8"
        )

        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger
