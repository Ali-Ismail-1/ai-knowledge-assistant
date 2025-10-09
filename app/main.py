from fastapi import FastAPI
from app.api.router import router
from app.core.logging_config import setup_logging

def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="AI Knowledge Assistant", version="1.0.0")
    app.include_router(router)
    return app

app = create_app()
