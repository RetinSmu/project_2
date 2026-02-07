from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def build_faiss(docs: List[str], api_key: str) -> FAISS:
    """
    Build a FAISS vector store from plain-text docs.
    """
    embeddings = OpenAIEmbeddings(api_key=api_key)
    documents = [Document(page_content=d) for d in docs]
    return FAISS.from_documents(documents, embeddings)


def lc_top_k_with_scores(query: str, store: FAISS, k: int = 6) -> List[Tuple[float, str]]:
    """
    Returns list of (score, text). Lower score = closer match for FAISS L2 distance.
    We'll still print it as 'score' for debugging.
    """
    results = store.similarity_search_with_score(query, k=k)
    out: List[Tuple[float, str]] = []
    for doc, score in results:
        out.append((float(score), doc.page_content))
    return out
