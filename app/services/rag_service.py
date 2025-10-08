from typing import Optional
from langchain_core.runnables import RunnableWithMessageHistory
from app.guardrails.filters import contains_profanity, redact_pii, strip_think
from app.guardrails.prompts import BASE_PROMPT
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from app.services.llm_service import get_llm
from app.services.vectorstore_service import get_retriever
from app.services.memory_service import get_history


class RAGService:
    def __init__(self):
        llm = get_llm()
        doc_chain = create_stuff_documents_chain(llm, BASE_PROMPT)
        retrieval_chain = create_retrieval_chain(get_retriever(), doc_chain)
        self._runnable = RunnableWithMessageHistory(
            runnable=retrieval_chain,
            get_session_history=get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask(self, session_id: str, question: str) -> str:
        if contains_profanity(question):
            return "Inappropriate question detected. Please rephrase your question."
        result = self._runnable.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result.get("answer") or str(result)
        answer = strip_think(answer)
        answer = redact_pii(answer)
        return answer

_instance: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    global _instance
    if _instance is None:
        _instance = RAGService()
    return _instance
