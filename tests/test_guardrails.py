# tests/test_guardrails.py
from app.guardrails.filters import strip_think, redact_pii, contains_profanity

def test_pii_redaction():
    assert "REDACTED" in redact_pii("123-45-6789")
    assert "REDACTED" in redact_pii("123-456-7890")
    assert "REDACTED" in redact_pii("john.doe@example.com")

def test_profanity_detection():
    assert contains_profanity("heck no") is True
    assert contains_profanity("hello") is False

def test_strip_think():
    text = "<think>I think this is a test</think>final answer"
    assert strip_think(text) == "final answer"
