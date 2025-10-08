import logging

logger = logging.getLogger("rag")

def log_interaction(session_id: str, question: str, answer: str):
    logger.info("session=%s question=%s answer=%s", session_id, question, answer)
