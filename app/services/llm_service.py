from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_llm():
    if settings.llm_provider == "openai":
        return ChatOpenAI(model=settings.openai_model, temperature=0.1, api_key=settings.openai_api_key)
    elif settings.llm_provider == "ollama":
        kwargs = {}
        if settings.ollama_base_url:
            kwargs["base_url"] = settings.ollama_base_url
        return ChatOllama(model=settings.ollama_model, temperature=0.1, **kwargs)
    raise ValueError(f"Invalid LLM provider: {settings.llm_provider}")