# tests/test_core_config.py
from pathlib import Path
from app.core.config import settings


def test_settings_load():
    assert settings.llm_provider in {"ollama", "openai"}
    assert settings.doc_dir.endswith("docs")


def test_settings_loads_and_paths():
    assert settings.llm_provider in {"ollama", "openai"}
    assert isinstance(settings.docs_path, Path)
    assert settings.docs_path.exists() or "docs" in str(settings.docs_path)
    assert isinstance(settings.chroma_path, Path)
    assert settings.retriever_k > 0
    assert settings.embedding_dimension > 0


def test_provider_name_cached():
    name = settings.provider_name
    assert name == settings.llm_provider.lower()
