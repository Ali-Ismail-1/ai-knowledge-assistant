# app/monitoring/logger.py
import logging

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.propagate = True # allow testing

def log_interaction(session_id: str, question: str, answer: str):
    logger.info("session=%s question=%s answer=%s", session_id, question, answer)
