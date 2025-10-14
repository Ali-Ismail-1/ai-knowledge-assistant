# app/services/vectorstore_service.py
import os
import logging
import time
from typing import Optional, Union
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

_vectorstore: Optional[Union[Chroma, PineconeVectorStore]] = None
_retriever = None

logger = logging.getLogger(__name__)

def build_vectorstore(doc_dir: str, chroma_dir: str = None, pinecone_index: str = None) -> Union[Chroma, PineconeVectorStore]:
    """Build and persist a vectorstore from documents on disk.

    This function recursively scans the provided `doc_dir` for supported
    file types, loads them into LangChain `Document` objects, splits them
    into overlapping chunks, computes embeddings, and persists a Chroma
    vector store at `chroma_dir`.

    Args:
        doc_dir: Root directory to recursively search for documents. Files
            ending with `.txt`, `.md`, and `.pdf` are supported.
        chroma_dir: Directory where the Chroma database will be created or
            updated and persisted. Used if vectorstore_provider is "chroma".
        pinecone_index: Name of the Pinecone index. Used if vectorstore_provider
            is "pinecone".

    Returns:
        A vector store instance containing the embedded document chunks
        persisted at `chroma_dir` or `pinecone_index`.
    """    

    docs = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith((".txt", ".md")):
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                elif file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(path).load())
            except Exception as e:
                logger.error(f"Error loading document {path}: {e}")
                continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    
    if not docs:
        logger.error(f"No documents found in {doc_dir}")
        raise ValueError(f"No documents found in directory: {doc_dir}")

    splits = splitter.split_documents(docs)
    if not splits:
        logger.error("No chunks created after splitting documents")
        raise ValueError("Document splitting resulting in no chunks created")
        
    logger.info(f"Successfully loaded {len(docs)} documents and created {len(splits)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)

    # Build vectorstore based on provider
    if settings.vectorstore_provider == "pinecone":
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed. Please install it with 'pip install langchain-pinecone'.")

        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API key is not set. Please set it in the configuration.")

        index_name = pinecone_index or settings.pinecone_index

        # initialize Pinecone client
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # Create or connect to index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": settings.pinecone_environment}},
            )

        # create vectorstore and add documents
        db = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name,
        )

        return db
    else: # Default to Chroma    
        chroma_path = chroma_dir or settings.persist_directory
        os.makedirs(chroma_path, exist_ok=True)

        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        db.add_documents(splits)
        db.persist()
        return db


def load_vectorstore(chroma_dir: str = None, pinecone_index: str = None) -> Union[Chroma, PineconeVectorStore]:
    """Load an existing persisted Chroma vector store.

    Args:
        chroma_dir: Path to the Chroma persistence directory. Used if 
            vectorstore_provider is "chroma".
        pinecone_index: Name of the Pinecone index. Used if vectorstore_provider
            is "pinecone".

    Returns:
        A vector store instance backed by the configured provider.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)

    if settings.vectorstore_provider == "pinecone":
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed. Please install it with 'pip install langchain-pinecone'.")

        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API key is not set. Please set it in the configuration.")

        index_name = pinecone_index or settings.pinecone_index

        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )

    else: # Default to Chroma
        chroma_path = chroma_dir or settings.persist_directory
        return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


def get_vectorstore() -> Union[Chroma, PineconeVectorStore]:
    """Return a lazily-initialized module-level vector store.

    Uses settings to determine which provider to use and the appropriate
    configuration. Subsequent calls return the same instance.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
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
