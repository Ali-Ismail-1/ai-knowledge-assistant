from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthcheck():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint_smoke(monkeypatch):
    monkeypatch.setattr("app.services.rag_service.RAGService.ask", lambda self, sid, q: "test answer")
    response = client.post("/v1/chat", json={"session_id": "abc", "question": "hi"})
    assert response.status_code == 200
    assert "answer" in response.json()
