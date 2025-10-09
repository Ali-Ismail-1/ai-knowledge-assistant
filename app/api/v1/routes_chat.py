# app/api/v1/routes_chat.py
from fastapi import APIRouter, Depends

from app.api.v1.schemas import ChatRequest, ChatResponse
from app.core.dependencies import get_rag
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RAGService = Depends(get_rag)):
    answer = rag.ask(req.session_id, req.question)
    return ChatResponse(answer=answer)
