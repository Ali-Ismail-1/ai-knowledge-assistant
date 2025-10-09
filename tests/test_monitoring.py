# tests/test_monitoring.py
from app.monitoring.logger import log_interaction

def test_logging_does_not_crash(tmp_path, caplog):
    log_interaction("test-session", "test-question", "test-answer")
    assert "test-session" in caplog.text
    assert "test-question" in caplog.text
    assert "test-answer" in caplog.text
