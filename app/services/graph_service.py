# app/services/graph_service.py
from typing import Optional, TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from app.services.llm_service import get_llm
from app.services.rag_service import RAGService
from app.services.tools import make_tools


class GraphState(TypedDict):
    question: str
    plan: str
    evidence: str
    answer: str

class LangGraphService:
    def __init__(self, llm: Optional[BaseChatModel] = None, rag: Optional[RAGService] = None):
        self.llm = llm or get_llm()
        self.rag = rag or RAGService()
        def _retrieve(q: str) -> str:
            return self.rag.ask(session_id="graph-tool", question=q)
        self.tools = make_tools(_retrieve)

        graph = StateGraph(GraphState)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)
        graph.add_node("respond", self._respond)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", "respond")
        graph.add_edge("respond", END)
        self.app = graph.compile()

    # nodes
    def _plan(self, state: GraphState) -> GraphState:
        q = state["question"]
        prompt = (
            "You are a planner. Decide whether to use 'retrieval' or 'web_search' tools "
            "for the question. Reply with terse plan string: TOOL=<tool>; WHY=<reason>.\n"
            f"Question: {q}\n\n"
        )
        plan = self.llm.invoke(prompt).content
        return {**state, "plan": plan}

    def _act(self, state: GraphState) -> GraphState:
        plan = state["plan"].lower()
        tool_name = "retrieval" if "retrieval" in plan else "web_search"
        tool = {tool.name: tool for tool in self.tools}.get(tool_name)
        evidence = tool.run(state["question"]) if tool else ""
        return {**state, "evidence": evidence}

    def _respond(self, state: GraphState) -> GraphState:
        q, ev, plan = state["question"], state.get("evidence", ""), state.get("plan", "")
        prompt = (
            "Use the EVIDENCE to answer the QUESTION succinctly. If insufficient, say 'I don't know'.\n"
            f"PLAN: {plan}\nEVIDENCE: {ev}\nQUESTION: {q}\n\n"
        )
        answer = self.llm.invoke(prompt).content
        return {**state, "answer": answer}

    def run(self, question: str) -> str:
        result = self.app.invoke({"question": question, "plan": "", "evidence": "", "answer": ""})
        return result["answer"]
