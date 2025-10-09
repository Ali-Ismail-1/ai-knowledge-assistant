
from app.services.agent_service import get_agent_service
from app.services.tools import make_tools


def test_make_tools_includes_retrieval_and_web_search():
    def fake_retrieve(q: str) -> str:
        return "retrieved answer"
    tools = make_tools(fake_retrieve)
    tools_names = [tool.name for tool in tools]
    assert "retrieval" in tools_names
    assert "web_search" in tools_names

def test_agent_service_runs_basic_query(monkeypatch):
    agent = get_agent_service()

    # Monkeypatch RAGService.ask to return predictable output
    monkeypatch.setattr(agent.rag, "ask", lambda session_id, question: "retrieved: " + question)

    response = agent.run("Who invented the Python?")
    assert isinstance(response, str)
    assert "Python" in response or "retrieved" in response