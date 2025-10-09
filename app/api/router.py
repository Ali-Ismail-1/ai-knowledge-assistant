# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1 import routes_health, routes_chat

router = APIRouter()
router.include_router(routes_health.router, prefix="/v1")
router.include_router(routes_chat.router, prefix="/v1")
