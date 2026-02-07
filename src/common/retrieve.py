from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def top_k_chunks(question: str, docs: List[str], k: int = 10) -> List[Tuple[int, float, str]]:
    if not docs:
        return []

    vec = TfidfVectorizer(stop_words="english", max_features=30000)
    X = vec.fit_transform(docs)
    q = vec.transform([question])

    sims = cosine_similarity(q, X).flatten()
    idxs = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i]), docs[int(i)]) for i in idxs]
