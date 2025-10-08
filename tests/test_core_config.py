from app.core.config import settings

def test_settings_load():
    assert settings.llm_provider in {"ollama", "openai"}
    assert settings.doc_dir.endswith("docs")
