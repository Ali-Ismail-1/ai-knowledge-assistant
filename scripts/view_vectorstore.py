from pinecone import Pinecone
from app.core.config import settings

pc = Pinecone(api_key=settings.pinecone_api_key)
print(pc.list_indexes().names())
