from src.common.ns_data import load_csv, df_to_row_docs
from src.common.retrieve import top_k_chunks
from src.common.context_limit import build_context
from src.common.llm_openai import chat_answer


def vanilla_csv(question: str, csv_path: str, k: int = 8) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    hits = top_k_chunks(question, docs, k=k)  # [(idx, score, text)]
    context = build_context(hits, max_chars=3500, per_row_chars=280, max_rows=k)

    system = (
        "You answer using ONLY the provided rows.\n"
        "Return HTML only. No markdown.\n"
        "Do NOT output code blocks.\n"
        "Format exactly:\n"
        "<h3>Answer</h3><p>...</p>"
        "<h4>Key rows used</h4><ul><li>Row ...</li></ul>"
        "<h4>Missing</h4><p>...</p>\n"
        "If you cannot answer from rows, say what is missing."
    )

    user = f"Question: {question}\n\nRows:\n{context}\n\nAnswer:"
    out = chat_answer(system, user) or ""

    # tiny cleanup if model ever returns markdown anyway
    out = out.replace("```html", "").replace("```", "").strip()
    return out


if __name__ == "__main__":
    CSV_PATH = "data/Surgical_Wait_Times_20260207.csv"
    q = "highest Surgery_90th values"
    print(vanilla_csv(q, CSV_PATH))
