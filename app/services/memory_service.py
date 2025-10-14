# app/services/memory_service.py
from typing import Dict
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.services.llm_service import get_llm

_store: Dict[str, ChatMessageHistory] = {}
_summary_store: Dict[str, ConversationSummaryBufferMemory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    """Return a lazily-initialized module-level chat message history."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

def get_summary_memory(session_id: str) -> ConversationSummaryBufferMemory:
    if session_id not in _summary_store:
        _summary_store[session_id] = ConversationSummaryBufferMemory(
            llm=get_llm(),
            max_token_limit=1000,
            return_messages=False
        )
    return _summary_store[session_id]
