# app/core/json_logging.py
import json
import logging
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON.
    If the original message is a dict, it is merged with level/logger.
    Otherwise, message becomes {"message": "<string>"}.
    """

    def format(self, record: logging.LogRecord) -> str:
        # If a dict was passed as the message, use it; else wrap it
        payload: Dict[str, Any]
        if isinstance(record.msg, dict):
            payload = dict(record.msg)
            # Ensure fallback text message still included
            if "message" not in payload:
                payload["message"] = super().format(record)
        else:
            payload = {"message": super().format(record)}

        payload["level"] = record.levelname
        payload["logger"] = record.name

        # Include exception details if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)
