from src.common.ns_data import load_csv, df_to_row_docs
from src.common.llm_openai import chat_answer
from src.common.config import get_openai_key
from src.common.lc_retrieve import build_faiss, lc_top_k_with_scores


def vanilla_csv(question: str, csv_path: str, k_retrieve: int = 10, k_send: int = 8) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    api_key = get_openai_key()
    store = build_faiss(docs, api_key=api_key)

    hits = lc_top_k_with_scores(question, store, k=k_retrieve)

    print("\n=== VANILLA (LangChain) RETRIEVED ROWS ===")
    for r, (score, text) in enumerate(hits, 1):
        print(f"\n[{r}] score={score:.4f}")
        print(text[:450] + ("..." if len(text) > 450 else ""))

    # token filter
    context = "\n\n".join([f"Row {r}:\n{t[:700]}" for r, (_, t) in enumerate(hits[:k_send], 1)])

    system = (
        "You are a helpful assistant. "
        "Use ONLY the provided rows. "
        "If you cannot answer from rows, say what is missing."
    )
    user = f"Question: {question}\n\nRows:\n{context}\n\nAnswer:"
    return chat_answer(system, user)


if __name__ == "__main__":
    csv_path = "data/Surgical_Wait_Times_20260207.csv"
    q = "highest Surgery_90th values"
    print("\n=== VANILLA ANSWER ===\n", vanilla_csv(q, csv_path))
