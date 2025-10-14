# app/services/rag_service.py
"""RAG Service orchestrating retrieval, generation, and guardrails."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Mapping, TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableWithMessageHistory

from app.core.config import settings
from app.guardrails.filters import contains_profanity, redact_pii, strip_think
from app.guardrails.prompts import BASE_PROMPT
from app.monitoring.logger import log_interaction
from app.services.llm_service import get_llm
from app.services.memory_service import get_history, get_summary_memory
from app.services.vectorstore_service import get_retriever

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

logger = logging.getLogger(__name__)

# Constants (Easy to mock)
FALLBACK_ERROR_MSG = "I couldn't process that just now. Please try again in a moment."
PROFANITY_REJECTION_MSG = (
    "Inappropriate question detected. Please rephrase your question."
)
IDK_MSG = "I don't know."
UNKNOWN_EQUIVALENTS = {"unknown", "not sure", "n/a", "i don't know"}


class _InvokeResult(TypedDict, total=False):
    """Shape returned by LangChain retrieval chains.

    Note: TypeDict with total=False allows flexible handling
    of chains that may return different structures.
    """

    answer: str
    context: list[Any]
    source_documents: list[Any]


def _postprocess_answer(text: str) -> str:
    """Apply output guardrails in deterministic order.

    Order matters:
    1. Strip hidden <think> blocks (potential prompt leakage)
    2. Redact PII (must happen after think removal to catch all text)

    Args:
        text: Raw LLM output

    Returns:
        Sanitized text safe for user consumption
    """
    return redact_pii(strip_think(text))


def _normalize_answer(raw_answer: str) -> str:
    """Normalize edge cases in LLM responses.

    Handles:
    - Empty/whitespace-only responses
    - Common "I don't know" variants that should be standardized

    Args:
        raw_answer: Raw text from LLM

    Returns:
        Normalized answer, defaulting to IDK_MSG for empty/ambiguous responses
    """
    normalized = raw_answer.strip()

    if not normalized or normalized.lower() in UNKNOWN_EQUIVALENTS:
        return IDK_MSG

    return normalized


class RAGService:
    """Retrieval-Augmented Generation orchestrator.

    Coordinates:
    - Input validation and guardrails (profanity filtering)
    - Documents retrieval via vector store
    - LLM generation with conversation history
    - Output sanitization (PII redaction, think-tag removal)
    - Structured observability logging
    - Graceful error handling

    Thread-safety: Not thread safe. Use one instance per thread or
    protect with locks if sharing across threads.
    """

    __slots__ = ("_runnable",)  # Memory optimization for service singletons

    def __init__(self) -> None:
        """Initialize RAG pipeline components.

        Raises:
            Various LangChain/LLM exceptions during initialization
            Caller should handle gracefully or fail-fast at startup
        """
        llm = get_llm()
        doc_chain = create_stuff_documents_chain(llm, BASE_PROMPT)
        retrieval_chain = create_retrieval_chain(get_retriever(), doc_chain)

        self._runnable = RunnableWithMessageHistory(
            runnable=retrieval_chain,
            get_session_history=get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        logger.info(
            "RAGService initialized successfully",
            extra={
                "llm_provider": settings.llm_provider,
                "retriever_k": settings.retriever_k,
            },
        )

    def ask(self, session_id: str, question: str) -> str:
        """Process user question through RAG pipeline with full guardrails.

        Flow:
        1. Input validation (profanity filtering)
        2. Retrieval & generation via LangChain
        3. Output sanitization (remove think, PII redaction)
        4. Response normalization
        5. Structured logging

        Args:
            session_id: Stable conversation identifier for memory/logging
            question: User's natural language question

        Returns:
            Sanitized answer safe for user display. Never raises;
            returns user-friendly error message on failure.
        """
        # Input guardrail: Profanity filtering
        if contains_profanity(question):
            return self._log_and_respond(
                session_id,
                question,
                PROFANITY_REJECTION_MSG,
                meta={"guardrails": "profanity"},
            )

        try:
            result = self._runnable.invoke(
                {"input": question}, config={"configurable": {"session_id": session_id}}
            )
        except Exception:
            logger.exception("RAG chain failed", extra={"session_id": session_id})
            return self._log_and_respond(
                session_id,
                question,
                FALLBACK_ERROR_MSG,
                meta={"error": "invoke_failed"},
            )

        # Extract, sanitize, and normalize answer
        raw = result.get("answer") if isinstance(result, dict) else str(result)
        answer = _normalize_answer(_postprocess_answer(raw))

        # Log and save to memory
        return self._log_and_respond(
            session_id,
            question,
            answer,
            meta={
                "provider": settings.llm_provider,
                "retriever_k": settings.retriever_k,
            },
        )

    def _log_and_respond(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Emit structured log, update memory, and return answer.

        Both logging and memory updates are best-effort (failures suppressed).

        Args:
            session_id: Session Identifier
            question: User Question
            answer: System response
            meta: Optional metadata for structured logging

        Returns:
            The answer unchanged.
        """
        # Log interaction
        with suppress(Exception):
            log_interaction(session_id, question, answer, meta=meta)

        # Update memory
        with suppress(Exception):
            memory = get_summary_memory(session_id)
            memory.save_context({"input": question}, {"output": answer})

        return answer


# Singleton factory
_instance: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or lazily create RAGService singleton.

    Returns:
        Shared RAGService instance.

    Note:
        Not thread-safe during initialization. Initialize eagerly
        at startup if called from multiple threads.
    """
    global _instance
    if _instance is None:
        _instance = RAGService()
    return _instance


def reset_rag_service() -> None:
    """Reset RAGService singleton for testing. Not for production use."""
    global _instance
    _instance = None
