# tests/test_monitoring.py
from io import StringIO
import json
import logging
from pathlib import Path
from app.core.logging_config import setup_logging
from app.monitoring.feedback import record_feedback
from app.monitoring.logger import log_interaction


def test_logging_does_not_crash(caplog):
    """Verify log_interaction writes expected text to logger"""
    setup_logging()

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("rag")
    logger.addHandler(handler)
    
    log_interaction("test-session", "test-question", "test-answer")

    handler.flush()
    output = stream.getvalue()
    handler.close()
    logger.removeHandler(handler)
    
    assert "test-session" in output
    assert "test-question" in output
    assert "test-answer" in output


def test_feedback_logs_json_to_file(tmp_path, monkeypatch):
    # Redirect logs directory to a temporary directory
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    # Monkeypatch LOG_DIR path used by setup_logging
    # NOTE: We cannot reassign imported LOG_DIR directly in dictConfig.
    # Instead, temporarily chdir into tmp and let default LOG_DIR ("logs") be created there.
    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        setup_logging()
        record_feedback(
            "test-session", "test-question", "test-answer", 5, extra={"tag": "thing"}
        )

        feedback_path = Path("logs") / "feedback.log"
        assert feedback_path.exists(), "feedback.log should be created"

        # Read last line and verify JSON structure
        content = feedback_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(content) >= 1
        last = json.loads(content[-1])

        assert last["session_id"] == "test-session"
        assert last["rating"] == 5
        assert last["logger"] == "feedback"
        assert last["level"] == "INFO"
        assert last["tag"] == "thing"

    finally:  # Clean up
        os.chdir(cwd)
