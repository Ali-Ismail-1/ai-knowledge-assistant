# app/services/vectorstore_service.py
"""Vector store management for RAG applications.

This module provides functionality to build, load, and manage vector stores
using either Chroma or Pinecone as the backend.
"""

import os
import logging
import time
from typing import Optional, Union, List
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.debug(
        "Pinecone is not installed. Please install it with 'pip install langchain-pinecone'."
    )

if PINECONE_AVAILABLE:
    VectorStoreType = Union[Chroma, PineconeVectorStore]
else:
    VectorStoreType = Chroma

# =============================
# Helper Functions (Private)
# ==============================


def _validate_pinecone_config() -> None:
    """Validate Pinecone configuration.

    Raises:
        ImportError: If Pinecone is not installed.
        ValueError: If Pinecone settings are missing
    """
    if not PINECONE_AVAILABLE:
        raise ImportError(
            "Pinecone is not installed. Please install it with 'pip install langchain-pinecone'."
        )

    if not settings.pinecone_api_key:
        raise ValueError(
            "Pinecone API key is not set. Please set it in the configuration."
        )


def _load_documents(doc_dir: str) -> List[Document]:
    """Load documents from directory with error handling.

    Args:
        doc_dir: Directory containing documents to load.

    Returns:
        List of loaded LangChain Document objects.

    Raises:
        FileNotFoundError: If doc_dir doesn't exist.
        ValueError: If no documents are found.
    """
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"Directory {doc_dir} does not exist")

    docs = []
    supported_extensions = {".txt", ".md", ".pdf"}
    files_processed = 0

    logger.info(f"Scanning directory {doc_dir} for documents...")

    for root, _, files in os.walk(doc_dir):
        for file in files:
            file_lower = file.lower()

            if not any(file_lower.endswith(ext) for ext in supported_extensions):
                continue

            files_processed += 1
            path = os.path.join(root, file)

            try:
                if file_lower.endswith((".txt", ".md")):
                    logger.debug(f"Loading text file: {file}")
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                elif file_lower.endswith(".pdf"):
                    logger.debug(f"Loading PDF file: {file}")
                    docs.extend(PyPDFLoader(path).load())

            except Exception as e:
                logger.warning(f"Failed to load document {path}: {e}")
                continue

    if files_processed == 0:
        raise ValueError(f"No supported documents found in {doc_dir}")

    if not docs:
        raise ValueError(
            f"Found {files_processed} files but failed to load any documents from {doc_dir}"
        )

    logger.info(
        f"Successfully loaded {len(docs)} documents from {files_processed} files"
    )
    return docs


def _split_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into chunks.

    Args:
        docs: Documents to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of document chunks.

    Raises:
        ValueError: If no chunks are created.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("Document splitting resulting in no chunks created")

    logger.info(f"Created {len(splits)} chunks from {len(docs)} documents")
    return splits


def _ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    """Ensure Pinecone index exists and is ready.

    Args:
        pc: Pinecone client instance.
        index_name: Name of the index.
        dimension: Embedding dimension.

    Raises:
        TimeoutError: If index doesn't become ready in time.
    """
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index {index_name}")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec={
                "serverless": {"cloud": "aws", "region": settings.pinecone_environment}
            },
        )

        # Wait for index to be ready
        max_wait = 60
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                if pc.describe_index(index_name).status["ready"]:
                    logger.info(f"Pinecone index {index_name} is ready")
                    return
            except Exception as e:
                logger.debug(f"Error checking index status: {e}")

            logger.info(f"Waiting for Pinecone index {index_name} to be ready...")
            time.sleep(2)

        raise TimeoutError(f"Pinecone index {index_name} did not become ready in time")

    else:
        logger.info(f"Using existing Pinecone index {index_name}")


def _build_pincone_vectorstore(
    splits: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_name: str,
) -> PineconeVectorStore:
    """Build Pinecone vectorstore.

    Args:
        splits: Document chunks to embed.
        embeddings: Embedding function.
        index_name: Pinecone index name.

    Returns:
        Configured PineconeVectorStore.
    """
    _validate_pinecone_config()

    logger.info(f"Building Pinecone vectorstore for index {index_name}")

    # initialize Pinecone client
    pc = Pinecone(api_key=settings.pinecone_api_key)

    # ensure index exists and is ready
    _ensure_pinecone_index(pc, index_name, settings.embedding_dimension)

    # create vectorstore and add documents
    logger.info(f"Adding {len(splits)} documents to Pinecone index {index_name}")

    vectorstore = PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=settings.pinecone_api_key,
    )

    logger.info(f"Pinecone vectorstore for index {index_name} built successfully")
    return vectorstore


def _build_chroma_vectorstore(
    splits: List[Document],
    embeddings: HuggingFaceEmbeddings,
    chroma_dir: str,
) -> Chroma:
    """Build Chroma vectorstore.

    Args:
        splits: Document chunks to embed.
        embeddings: Embedding function.
        chroma_dir: Persistence directory.

    Returns:
        Configured Chroma instance.
    """
    os.makedirs(chroma_dir, exist_ok=True)

    logger.info(f"Building Chroma vectorstore in {chroma_dir}")

    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    logger.info(f"Adding {len(splits)} chunks to Chroma vectorstore in {chroma_dir}")
    db.add_documents(splits)

    logger.info(f"Chroma vectorstore in {chroma_dir} built successfully")
    return db


# ============================================================================
# Public API Functions
# ============================================================================


def build_pinecone_vectorstore(
    doc_dir: str, pinecone_index: str = None
) -> PineconeVectorStore:
    """Build and persist a Pinecone vectorstore from documents on disk."""
    logger.info(f"Building Pinecone vectorstore from {doc_dir}")
    print(f"Building Pinecone vectorstore from {doc_dir}")

    # Load documents
    docs = _load_documents(doc_dir)
    splits = _split_documents(docs)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)

    # Determine index name
    index_name = pinecone_index or settings.pinecone_index

    # Build Pinecone vectorstore
    vectorstore = _build_pincone_vectorstore(splits, embeddings, index_name)

    logger.info(f"Pinecone vectorstore for index {index_name} built successfully")
    print(f"✅ Pinecone vectorstore for index '{index_name}' built successfully")

    return vectorstore


def build_vectorstore(
    doc_dir: str, chroma_dir: str = None, pinecone_index: str = None
) -> VectorStoreType:
    """Build and persist a vectorstore from documents on disk.

    This function recursively scans the provided `doc_dir` for supported
    file types, loads them into LangChain `Document` objects, splits them
    into overlapping chunks, computes embeddings, and persists a Chroma
    vector store at `chroma_dir`.

    Args:
        doc_dir: Root directory to recursively search for documents.
        chroma_dir: Directory for Chroma database persistence. Only used when
            vectorstore_provider is "chroma". Defaults to settings.persist_directory.
        pinecone_index: Name of the Pinecone index. Only used when
            vectorstore_provider is "pinecone". Defaults to settings.pinecone_index.

    Returns:
        A vectorstore instance (Chroma or PineconeVectorStore) containing
        the embedded document chunks.

    Raises:
        ValueError: If no documents are found or no chunks are created.
        ImportError: If Pinecone provider is configured but not installed.
        FileNotFoundError: If doc_dir does not exist.

    Example:
        >>> vs = build_vectorstore("./data/docs")
        >>> retriever = vs.as_retriever()
    """

    logger.info(f"Building {settings.vectorstore_provider} vectorstore from {doc_dir}")

    # Load and split documents
    docs = _load_documents(doc_dir)
    splits = _split_documents(docs)

    # Initialize embeddings
    logger.info(f"Initializing embeddings model {settings.embeddings_model}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)

    # Build vectorstore based on provider
    logger.info(f"Building {settings.vectorstore_provider} vectorstore from {doc_dir}")
    print(f"Building {settings.vectorstore_provider} vectorstore from {doc_dir}")
    if settings.vectorstore_provider == "pinecone":
        index_name = pinecone_index or settings.pinecone_index
        return _build_pincone_vectorstore(splits, embeddings, index_name)
    else:
        chroma_dir = chroma_dir or settings.persist_directory
        return _build_chroma_vectorstore(splits, embeddings, chroma_dir)


def load_vectorstore(
    chroma_dir: str = None, pinecone_index: str = None
) -> VectorStoreType:
    """Load an existing persisted vectorstore.

    Args:
        chroma_dir: Path to the Chroma persistence directory. Only used when
            vectorstore_provider is "chroma". Defaults to settings.persist_directory.
        pinecone_index: Name of the Pinecone index. Only used when
            vectorstore_provider is "pinecone". Defaults to settings.pinecone_index.

    Returns:
        A vectorstore instance backed by the configured provider.

    Raises:
        ImportError: If Pinecone provider is configured but not installed.
        ValueError: If required configuration is missing.

    Example:
        >>> vs = load_vectorstore()
        >>> retriever = vs.as_retriever(search_kwargs={"k": 5})
    """
    logger.info(
        f"Loading {settings.vectorstore_provider} vectorstore from {chroma_dir if chroma_dir else pinecone_index}"
    )

    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model)

    if settings.vectorstore_provider == "pinecone":
        _validate_pinecone_config()

        index_name = pinecone_index or settings.pinecone_index
        logger.info(f"Loading Pinecone vectorstore from index {index_name}")

        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )

    else:  # Default to Chroma
        chroma_path = chroma_dir or settings.persist_directory

        if not os.path.exists(chroma_path):
            logger.warning(
                f"Chroma persistence directory {chroma_path} does not exist."
            )

        logger.info(f"Loading Chroma vectorstore from {chroma_path}")
        return Chroma(persist_directory=chroma_path, embedding_function=embeddings)


# ============================================================================
# Singleton Pattern (Global Cache)
# ============================================================================


_vectorstore: Optional[VectorStoreType] = None
_retriever = None


def get_vectorstore() -> VectorStoreType:
    """Return a lazily-initialized module-level vectorstore.

    Uses settings to determine which provider to use and the appropriate
    configuration. Subsequent calls return the same cached instance.

    Returns:
        Cached vectorstore instance.

    Example:
        >>> vs = get_vectorstore()
        >>> results = vs.similarity_search("my query")
    """
    global _vectorstore

    if _vectorstore is None:
        logger.debug("Initializing vectorstore cache")
        _vectorstore = load_vectorstore()

    return _vectorstore


def get_retriever():
    """Return a lazily-initialized retriever bound to the vectorstore.

    The retriever returns the top-k most similar chunks per query. The
    value of k is controlled by settings.retriever_k.

    Returns:
        Retriever instance configured with similarity search.

    Example:
        >>> retriever = get_retriever()
        >>> docs = retriever.get_relevant_documents("my query")
    """
    global _retriever

    if _retriever is None:
        logger.debug(f"Initializing retriever cache with k={settings.retriever_k}")
        _retriever = get_vectorstore().as_retriever(
            search_kwargs={"k": settings.retriever_k}
        )

    return _retriever


def reset_cache():
    """Reset the cached vectorstore and retriever instances.

    This is useful for:
    - Testing with different configurations
    - Switching between vectorstore providers
    - Reloading after index updates or configuration changes

    Example:
        >>> reset_cache()
        >>> vs = get_vectorstore()  # Will load fresh instance
    """
    global _vectorstore, _retriever

    logger.info("Resetting vectorstore and retriever cache")
    _vectorstore = None
    _retriever = None
