# tests/test_rag_service.py
import pytest
from unittest.mock import Mock, patch
from app.services.rag_service import (
    RAGService,
    get_rag_service,
    reset_rag_service,
    _normalize_answer,
    _postprocess_answer,
    IDK_MSG,
    PROFANITY_REJECTION_MSG,
    FALLBACK_ERROR_MSG,
)


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", IDK_MSG),
        ("  ", IDK_MSG),
        ("   \n  ", IDK_MSG),
        ("unknown", IDK_MSG),
        ("Unknown", IDK_MSG),  # case insensitive
        ("not sure", IDK_MSG),
        ("n/a", IDK_MSG),
        ("I don't know", IDK_MSG),
        ("  valid answer  ", "valid answer"),
        ("This is a real answer", "This is a real answer"),
    ],
)
def test_normalize_answer_variants(raw, expected):
    """Test answer normalization handles empty and IDK variants."""
    assert _normalize_answer(raw) == expected


def test_postprocess_removes_think_tags():
    """Test that <think> tags are stripped from output."""
    raw = "Answer here <think>internal reasoning</think> more answer"
    result = _postprocess_answer(raw)
    assert "<think>" not in result
    assert "internal reasoning" not in result
    assert "Answer here" in result


def test_postprocess_redacts_pii():
    """Test that PII is redacted from output."""
    raw = "Contact John at john@email.com or 555-123-4567"
    result = _postprocess_answer(raw)
    # Assuming your redact_pii function catches these patterns
    assert "john@email.com" not in result or "[REDACTED]" in result


def test_postprocess_applies_filters_in_order():
    """Test that filters are applied in correct order: think -> PII."""
    calls = []

    with patch("app.services.rag_service.strip_think") as mock_think:
        with patch("app.services.rag_service.redact_pii") as mock_pii:
            mock_think.side_effect = lambda x: (calls.append("think"), x)[1]
            mock_pii.side_effect = lambda x: (calls.append("pii"), x)[1]

            _postprocess_answer("test")

            assert calls == ["think", "pii"], "Filters should run in order"


# ============================================================================
# Integration Tests for RAGService
# ============================================================================


@pytest.fixture
def mock_rag_components():
    """Mock all RAG dependencies to isolate RAGService logic."""
    with patch("app.services.rag_service.get_llm") as mock_llm, patch(
        "app.services.rag_service.get_retriever"
    ) as mock_retriever, patch(
        "app.services.rag_service.create_stuff_documents_chain"
    ), patch(
        "app.services.rag_service.create_retrieval_chain"
    ), patch(
        "app.services.rag_service.RunnableWithMessageHistory"
    ) as mock_runnable:

        yield {
            "llm": mock_llm,
            "retriever": mock_retriever,
            "runnable": mock_runnable,
        }


@pytest.fixture
def rag_service(mock_rag_components):
    """Create RAGService instance with mocked dependencies."""
    reset_rag_service()  # Clear singleton
    service = RAGService()
    return service


def test_rag_service_initialization(mock_rag_components):
    """Test RAGService initializes correctly."""
    service = RAGService()
    assert service is not None
    mock_rag_components["llm"].assert_called_once()
    mock_rag_components["retriever"].assert_called_once()


def test_ask_rejects_profanity(rag_service):
    """Test that profane questions are rejected before LLM call."""
    with patch("app.services.rag_service.contains_profanity", return_value=True):
        result = rag_service.ask("session-1", "profane question")

        assert result == PROFANITY_REJECTION_MSG
        # Verify LLM was not called
        rag_service._runnable.invoke.assert_not_called()


def test_ask_returns_answer_on_success(rag_service):
    """Test successful RAG flow returns sanitized answer."""
    # Mock the runnable to return a dict with answer
    rag_service._runnable.invoke = Mock(return_value={"answer": "This is the answer"})

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch("app.services.rag_service.log_interaction"), patch(
        "app.services.rag_service.get_summary_memory"
    ):

        result = rag_service.ask("session-1", "What is the return policy?")

        assert result == "This is the answer"
        rag_service._runnable.invoke.assert_called_once()


def test_ask_handles_llm_exception_gracefully(rag_service):
    """Test that LLM exceptions return user-friendly error message."""
    rag_service._runnable.invoke = Mock(side_effect=Exception("LLM failed"))

    with patch("app.services.rag_service.contains_profanity", return_value=False):
        result = rag_service.ask("session-1", "test question")

        assert result == FALLBACK_ERROR_MSG


def test_ask_normalizes_empty_llm_response(rag_service):
    """Test that empty LLM responses are normalized to IDK_MSG."""
    rag_service._runnable.invoke = Mock(return_value={"answer": "   "})

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch("app.services.rag_service.log_interaction"), patch(
        "app.services.rag_service.get_summary_memory"
    ):

        result = rag_service.ask("session-1", "question")

        assert result == IDK_MSG


def test_ask_applies_guardrails_to_output(rag_service):
    """Test that output guardrails (think removal, PII redaction) are applied."""
    raw_answer = "Answer <think>reasoning</think> with email@test.com"
    rag_service._runnable.invoke = Mock(return_value={"answer": raw_answer})

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch(
        "app.services.rag_service.strip_think",
        return_value="Answer  with email@test.com",
    ) as mock_think, patch(
        "app.services.rag_service.redact_pii", return_value="Answer with [REDACTED]"
    ) as mock_pii, patch(
        "app.services.rag_service.log_interaction"
    ), patch(
        "app.services.rag_service.get_summary_memory"
    ):

        result = rag_service.ask("session-1", "question")

        mock_think.assert_called_once()
        mock_pii.assert_called_once()
        assert "[REDACTED]" in result


def test_ask_logs_interaction(rag_service):
    """Test that successful interactions are logged."""
    rag_service._runnable.invoke = Mock(return_value={"answer": "test answer"})

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch("app.services.rag_service.log_interaction") as mock_log, patch(
        "app.services.rag_service.get_summary_memory"
    ):

        rag_service.ask("session-1", "test question")

        mock_log.assert_called_once()
        args = mock_log.call_args[0]
        assert args[0] == "session-1"
        assert args[1] == "test question"
        assert args[2] == "test answer"


def test_ask_updates_memory(rag_service):
    """Test that conversation memory is updated after each interaction."""
    rag_service._runnable.invoke = Mock(return_value={"answer": "test answer"})
    mock_memory = Mock()

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch("app.services.rag_service.log_interaction"), patch(
        "app.services.rag_service.get_summary_memory", return_value=mock_memory
    ):

        rag_service.ask("session-1", "test question")

        mock_memory.save_context.assert_called_once_with(
            {"input": "test question"}, {"output": "test answer"}
        )


def test_ask_suppresses_logging_errors(rag_service):
    """Test that logging failures don't crash the response."""
    rag_service._runnable.invoke = Mock(return_value={"answer": "answer"})

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch(
        "app.services.rag_service.log_interaction", side_effect=Exception("Log failed")
    ), patch(
        "app.services.rag_service.get_summary_memory"
    ):

        # Should not raise, should return answer
        result = rag_service.ask("session-1", "question")
        assert result == "answer"


def test_ask_suppresses_memory_errors(rag_service):
    """Test that memory save failures don't crash the response."""
    rag_service._runnable.invoke = Mock(return_value={"answer": "answer"})
    mock_memory = Mock()
    mock_memory.save_context.side_effect = Exception("Memory save failed")

    with patch(
        "app.services.rag_service.contains_profanity", return_value=False
    ), patch("app.services.rag_service.log_interaction"), patch(
        "app.services.rag_service.get_summary_memory", return_value=mock_memory
    ):

        # Should not raise, should return answer
        result = rag_service.ask("session-1", "question")
        assert result == "answer"


# ============================================================================
# Singleton Tests
# ============================================================================


def test_get_rag_service_returns_singleton(mock_rag_components):
    """Test that get_rag_service returns the same instance."""
    reset_rag_service()

    service1 = get_rag_service()
    service2 = get_rag_service()

    assert service1 is service2


def test_reset_rag_service_clears_singleton(mock_rag_components):
    """Test that reset_rag_service creates new instance."""
    service1 = get_rag_service()
    reset_rag_service()
    service2 = get_rag_service()

    assert service1 is not service2
