from src.common.ns_data import load_csv, df_to_row_docs
from src.common.retrieve import top_k_chunks
from src.common.llm_openai import chat_answer


def vanilla_csv(question: str, csv_path: str, k_retrieve: int = 10, k_send: int = 8) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    hits = top_k_chunks(question, docs, k=k_retrieve)

    print("\n=== VANILLA RETRIEVED ROWS ===")
    for r, (i, s, c) in enumerate(hits, 1):
        print(f"\n[{r}] idx={i} score={s:.4f}")
        print(c[:450] + ("..." if len(c) > 450 else ""))

    # token filter: only send top k_send
    context = "\n\n".join([f"Row {i}:\n{c[:700]}" for i, _, c in hits[:k_send]])

    system = (
        "You are a helpful assistant. "
        "Use ONLY the provided rows. "
        "If you cannot answer from rows, say what is missing."
    )
    user = f"Question: {question}\n\nRows:\n{context}\n\nAnswer:"
    return chat_answer(system, user)


if __name__ == "__main__":
    csv_path = "data/Surgical_Wait_Times_20260207.csv"
    q = "Which procedures have the longest surgery wait times (Surgery_90th)?"
    print("\n=== VANILLA ANSWER ===\n", vanilla_csv(q, csv_path))
