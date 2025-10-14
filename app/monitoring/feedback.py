# app/monitoring/feedback.py
import logging
from time import time
from typing import Any, Dict, Optional

feedback_logger = logging.getLogger("feedback")
feedback_logger.setLevel(logging.INFO)
# Keep propagate=True so tests can capture JSON content without dictConfig
feedback_logger.propagate = True


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record structured user feedback as JSON.
    Writes to logs/feedback.log when logging_config.setup_logging() is active.
    """
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": int(rating),
        "timestamp": int(time()),
    }
    if extra:
        payload.update(extra)

    feedback_logger.info(payload)
