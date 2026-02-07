from src.common.ns_data import load_csv, df_to_text
from src.common.chunk import chunk_text
from src.common.retrieve import top_k_chunks
from src.common.llm_openai import chat_answer

def answer_from_csv(question: str, csv_path: str) -> str:
    df = load_csv(csv_path)
    text = df_to_text(df, max_rows=300, max_cols=25)

    chunks = chunk_text(text, chunk_chars=1400, overlap=200)
    hits = top_k_chunks(question, chunks, k=5)

    print("\n=== RETRIEVED (NS CSV) ===")
    for r, (i, s, c) in enumerate(hits, 1):
        print(f"\n[{r}] idx={i} score={s:.4f}")
        print(c[:450] + ("..." if len(c) > 450 else ""))

    # token filter
    context = "\n\n".join([f"Chunk {i}:\n{c[:1200]}" for i, _, c in hits])

    system = "Use ONLY the context. If missing, say what is missing."
    user = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    return chat_answer(system, user)

if __name__ == "__main__":
    csv_path = "data/Surgical_Wait_Times_20260207.csv"

    q = "Which procedures have the longest wait times?"
    print("\n=== ANSWER ===\n", answer_from_csv(q, csv_path))
