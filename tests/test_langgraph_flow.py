# tests/test_langgraph_flow.py
from types import SimpleNamespace
from app.services.graph_service import LangGraphService


class StubLLM:
    def invoke(self, prompt, *args, **kwargs):
        if "You are a planner" in prompt:
            return SimpleNamespace(content="TOOL=retrieval; WHY=test")
        # respond step
        return SimpleNamespace(content="Gravity is due to mass; answer: gravity")

def test_graph_executes_full_flow():
    graph = LangGraphService(llm=StubLLM())
    output = graph.run("Who discovered gravity?")
    assert isinstance(output, str)
    assert "who" in output.lower() or "gravity" in output.lower()

def test_plan_node_produces_valid_plan():
    graph = LangGraphService(llm=StubLLM())
    result = graph._plan({"question": "Test"})
    assert "plan" in result
    assert isinstance(result["plan"], str)
