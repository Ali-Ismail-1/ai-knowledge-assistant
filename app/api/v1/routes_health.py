# app/api/v1/routes_health.py
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status": "healthy"}
