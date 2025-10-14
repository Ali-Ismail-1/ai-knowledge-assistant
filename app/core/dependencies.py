# app/core/dependencies.py
from fastapi import Depends
from app.core.config import settings
from app.services.rag_service import RAGService, get_rag_service

def get_settings():
    """Expose global settings for FastAPI depency injection."""
    return settings

def get_rag(rag: RAGService = Depends(get_rag_service)):
    """Inject singleton RAGService instance"""
    return rag
