# scripts/build_vectorstore.py
from app.services.vectorstore_service import build_pinecone_vectorstore
from app.core.config import settings

if __name__ == "__main__":
    vectorstore = build_pinecone_vectorstore(
        doc_dir=settings.doc_dir,
        pinecone_index=settings.pinecone_index
    )

    if settings.vectorstore_provider == "pinecone":
        print(f"✅ Vectorstore uploaded to Pinecone index: {settings.pinecone_index}")
    else:
        print(f"✅ Vectorstore built locally at: {settings.persist_directory}")
