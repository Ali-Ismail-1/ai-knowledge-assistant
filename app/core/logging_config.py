# app/core/logging_config.py
from logging.config import dictConfig
from pathlib import Path


def setup_logging(base_dir: str | None = None) -> None:
    LOG_DIR = Path(base_dir or "logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "json": {
                    "()": "app.core.json_logging.JsonFormatter",
                    "format": "%(message)s",  # message handled by JSONFormatter
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default"},
                "file_app": {
                    "class": "logging.FileHandler",
                    "formatter": "default",
                    "filename": str(LOG_DIR / "ai_knowledge_assistant.log"),
                    "mode": "a",
                    "encoding": "utf-8",
                },
                "file_feedback": {
                    "class": "logging.FileHandler",
                    "formatter": "json",
                    "filename": str(LOG_DIR / "feedback.log"),
                    "mode": "a",
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["console"], "level": "INFO"},
                "rag": {
                    "handlers": ["console", "file_app"],
                    "level": "INFO",
                    "propagate": True,
                },
                "feedback": {
                    "handlers": ["console", "file_feedback"],
                    "level": "INFO",
                    "propagate": True,
                },
            },
            "root": {"handlers": ["console", "file_app"], "level": "WARNING"},
        }
    )
