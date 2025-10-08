from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory

_store: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    """Return a lazily-initialized module-level chat message history."""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]
