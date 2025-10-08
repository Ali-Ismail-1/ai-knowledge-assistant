# app/core/logging_config.py
import logging
from logging.config import dictConfig
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": { "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s" },
        },
        "handlers": {
            "console": { "class": "logging.StreamHandler", "formatter": "default" },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": LOG_DIR / "ai_knowledge_assistant.log",
                "mode": "a",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console"], "level": "INFO"},
            "rag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        },
        "root": {"handlers": ["console", "file"], "level": "WARNING"},
    })