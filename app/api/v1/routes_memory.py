# app/api/v1/routes_memory.py
from fastapi import APIRouter
from app.services.memory_service import get_summary_memory

router = APIRouter(prefix="/memory", tags=["memory"])

@router.get("/summary/{session_id}")
def summary(session_id: str):
    memory = get_summary_memory(session_id)
    return {"summary": memory.load_memory_variables({}).get("history", "")}
