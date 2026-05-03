from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.config import dictConfig
from typing import Any

_LOGGING_CONFIGURED = False

_LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


def _normalize_log_level(value: str | None) -> str:
    allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    level = str(value or "INFO").upper().strip()
    return level if level in allowed else "INFO"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_BUILTIN_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    normalized_level = _normalize_log_level(log_level)
    selected_formatter = "json" if json_logs else "standard"
    access_level = "DEBUG" if normalized_level == "DEBUG" else "INFO"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "app.core.logging.JsonFormatter",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": selected_formatter,
                    "level": normalized_level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": normalized_level,
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console"],
                    "level": normalized_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": normalized_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": access_level,
                    "propagate": False,
                },
            },
        }
    )

    _LOGGING_CONFIGURED = True
    logging.getLogger(__name__).info(
        "Application logging configured",
        extra={"log_level": normalized_level, "json_logs": json_logs},
    )
