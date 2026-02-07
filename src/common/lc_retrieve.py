import os
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def build_or_load_faiss(
    documents: List[Document],
    index_dir: str,
    api_key: str,
    model: str = "text-embedding-3-small",
) -> FAISS:
    """
    Build FAISS once, then reuse by loading from disk.
    This prevents embedding every request (rate limit fix).
    """
    os.makedirs(index_dir, exist_ok=True)

    index_file = os.path.join(index_dir, "index.faiss")
    pkl_file = os.path.join(index_dir, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(pkl_file):
        return FAISS.load_local(index_dir, embeddings=_embeddings(api_key, model), allow_dangerous_deserialization=True)

    store = FAISS.from_documents(documents, _embeddings(api_key, model))
    store.save_local(index_dir)
    return store


def search_faiss(store: FAISS, query: str, k: int = 8) -> List[Tuple[Document, float]]:
    """
    Returns docs + score (lower score can mean closer depending on metric).
    We'll still show top k.
    """
    return store.similarity_search_with_score(query, k=k)


def _embeddings(api_key: str, model: str) -> OpenAIEmbeddings:
    # IMPORTANT: pass a STRING key (not a function)
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
    )
