import os
from typing import Optional
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

_vectorstore: Optional[Chroma]= None
_retriever = None

def build_vectorstore(doc_dir: str, chroma_dir: str) -> Chroma:
    """Build and persist a Chroma vectorstore from documents on disk.

    This function recursively scans the provided `doc_dir` for supported
    file types, loads them into LangChain `Document` objects, splits them
    into overlapping chunks, computes embeddings, and persists a Chroma
    vector store at `chroma_dir`.

    Args:
        doc_dir: Root directory to recursively search for documents. Files
            ending with `.txt`, `.md`, and `.pdf` are supported.
        chroma_dir: Directory where the Chroma database will be created or
            updated and persisted.

    Returns:
        A Chroma vector store instance containing the embedded document chunks
        persisted at `chroma_dir`.
    """
    os.makedirs(chroma_dir, exist_ok=True)

    docs = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith((".txt", ".md")):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif file.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    splits = splitter.split_documents(docs) if docs else []

    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    db.add_documents(splits)
    db.persist()
    return db

def load_vectorstore(chroma_dir: str) -> Chroma:
    """Load an existing persisted Chroma vector store.

    Args:
        chroma_dir: Path to the Chroma persistence directory.

    Returns:
        A Chroma instance backed by the data at `chroma_dir`.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)


def get_vectorstore() -> Chroma:
    """Return a lazily-initialized module-level Chroma vector store.

    Uses `settings.persist_directory` as the storage location.
    Subsequent calls return the same instance.
    """
    global _vectorstore, _retriever
    if _vectorstore is None:
        _vectorstore = load_vectorstore(settings.persist_directory)
    return _vectorstore

def get_retriever():
    """Return a lazily-initialized retriever bound to the vector store.

    The retriever returns the top-k most similar chunks per query. The
    value of k is controlled by configuration.
    """
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(search_kwargs={"k": settings.retriever_k})
    return _retriever
