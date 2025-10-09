# app/api/v1/routes_chat.py
from fastapi import APIRouter, Depends

from app.api.v1.schemas import ChatRequest, ChatResponse
from app.core.dependencies import get_rag
from app.services.agent_service import AgentService, get_agent_service
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RAGService = Depends(get_rag)):
    answer = rag.ask(req.session_id, req.question)
    return ChatResponse(answer=answer)

@router.post("/agent", response_model=ChatResponse)
def agent(req: ChatRequest, agent: AgentService = Depends(get_agent_service)):
    answer = agent.run(req.question)
    return ChatResponse(answer=answer)
