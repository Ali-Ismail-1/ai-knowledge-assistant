# app/api/v1/schemas.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
