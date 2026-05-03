from config import VECTOR_STORE_BACKEND

if VECTOR_STORE_BACKEND == "pinecone":
    from database.pinecone_store import init_store, close_store, get_store
else:
    from database.weaviate_store import init_store, close_store, get_store

__all__ = ["init_store", "close_store", "get_store"]
