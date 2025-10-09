# app/services/agent_service.py
from typing import Optional
from langchain.agents import initialize_agent, AgentType
from langchain_core.language_models.chat_models import BaseChatModel
from app.services.llm_service import get_llm
from app.services.tools import make_tools
from app.services.rag_service import RAGService


class AgentService:
    def __init__(self, llm: Optional[BaseChatModel] = None, rag: Optional[RAGService] = None):
        self.llm = llm or get_llm()
        self.rag = rag or RAGService()

        # Bridge to RAGService for the retrieval tool
        def _retrieve(q: str) -> str:
            return self.rag.ask(session_id="agent-tool", question=q)

        tools = make_tools(_retrieve)
        # Zero-shot ReAct is a crisp baseline for tool use
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False,
        )

    def run(self, query: str) -> str:
        return self.agent.run(query)

_instance: Optional[AgentService] = None

def get_agent_service() -> AgentService:
    global _instance
    if _instance is None:
        _instance = AgentService()
    return _instance
