from app.services.memory_service import get_history
from app.services.llm_service import get_llm
from app.services.vectorstore_service import get_retriever
from app.services.rag_service import get_rag_service


def test_memory_history_singleton():
    h1 = get_history("abc")
    h2 = get_history("abc")
    assert h1 is h2

def test_llm_creation():
    llm = get_llm()
    assert hasattr(llm, "invoke")

def test_retriever_setup():
    retriever = get_retriever()
    assert callable(retriever.get_relevant_documents)

def test_rag_service_answer():
    rag = get_rag_service()
    asnwer = rag.ask("demo-session", "What is the capital of France?")
    assert isinstance(asnwer, str)
