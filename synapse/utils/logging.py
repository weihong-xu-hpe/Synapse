"""Logging setup for Synapse runtime components."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler

from synapse.config import SynapseConfig
from synapse.utils.runtime import RuntimePaths, get_runtime_paths


LOG_FILES = {
    "synapse.mcp-daemon": "mcp-daemon.log",
    "synapse.file-watcher": "file-watcher.log",
    "synapse.janitor": "janitor.log",
    "synapse.audit": "audit.log",
}


class JsonLineFormatter(logging.Formatter):
    """Simple JSON-line formatter for structured-ish logs."""

    _STANDARD_ATTRS = {
        "args",
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
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._STANDARD_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["data"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _reset_handlers(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(config: SynapseConfig, runtime_paths: RuntimePaths | None = None) -> dict[str, logging.Logger]:
    """Configure the named rotating loggers used across Synapse subsystems."""

    paths = runtime_paths or get_runtime_paths(config)
    max_bytes = config.logging.max_file_size_mb * 1024 * 1024
    formatter = JsonLineFormatter()
    configured: dict[str, logging.Logger] = {}

    for logger_name, file_name in LOG_FILES.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _reset_handlers(logger)

        handler = RotatingFileHandler(
            paths.logs / file_name,
            maxBytes=max_bytes,
            backupCount=config.logging.retention_days,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        configured[logger_name] = logger

    return configured
