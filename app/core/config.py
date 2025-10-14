# app/core/config.py
from functools import cached_property
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

# Base directory (two levels up from this file)
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Configuration settings for the application.

    Lead settings from env variables, and .env files.
    Provides computed properties and strict validation for critical paths.
    """

    # Environment
    is_production: bool = False
    allow_reset: bool = True

    # Hardware Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LLM Provideres
    llm_provider: str = "ollama"  # "openai" or "ollama"

    # OpenAI
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None

    # Ollama
    ollama_model: str = "llama3:8b"
    ollama_base_url: str | None = "http://localhost:11434"

    # Embeddings & Vector Stores
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache: str | None = None
    embedding_dimension: int = 384

    # Vectorstore Settings
    vectorstore_provider: str = "pinecone"  # "chroma" or "pinecone"
    persist_directory: str = str(BASE_DIR / "data" / "chroma")
    chroma_url: str = "http://localhost:8000"
    doc_dir: str = str(BASE_DIR / "data" / "docs")

    # Pinecone Settings
    pinecone_api_key: str | None = None
    pinecone_index: str = "default-index"
    pinecone_environment: str = "us-east-1"

    # Retriever Settings
    retriever_k: int = 5

    # Orchestration
    use_langgraph: bool = False
    allow_multiple_providers: bool = False
    allow_multiple_embeddings: bool = False
    allow_multiple_chroma: bool = False

    # Streamlit
    streamlit_url: str = "http://localhost:8501"

    # Model config
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
    )

    @cached_property
    def chroma_path(self) -> Path:
        return Path(self.persist_directory).resolve()

    @cached_property
    def docs_path(self) -> Path:
        return Path(self.doc_dir).resolve()

    @cached_property
    def provider_name(self) -> str:
        return self.llm_provider.lower()


settings = Settings()
