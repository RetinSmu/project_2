from typing import List, Tuple

from src.common.ns_data import load_csv, df_to_row_docs
from src.common.llm_openai import chat_answer
from src.common.config import get_openai_key
from src.common.lc_retrieve import build_faiss, lc_top_k_with_scores


ALLOWED_FIELDS = [
    "Period", "Year", "Quarter", "Specialty", "Procedure",
    "Provider", "Zone", "Facility", "Consult_Median",
    "Consult_90th", "Surgery_Median", "Surgery_90th",
]


def make_subqueries(question: str) -> List[str]:
    system = (
        "You create search subqueries for a CSV dataset about Nova Scotia surgical wait times.\n"
        "Rules:\n"
        "- Output EXACTLY 3 lines.\n"
        "- Each line must be plain English keywords (not SQL).\n"
        "- No code blocks, no backticks, no quotes.\n"
        "- Use field names when helpful: " + ", ".join(ALLOWED_FIELDS) + "\n"
        "- Keep each line under 10 words.\n"
    )

    out = chat_answer(system, f"User question: {question}")
    raw_lines = [l.strip().lstrip("-").strip() for l in out.splitlines() if l.strip()]

    clean: List[str] = []
    for l in raw_lines:
        low = l.lower()
        if "```" in low or "select" in low or "from " in low or "where " in low:
            continue
        if ";" in l or "(" in l or ")" in l:
            continue
        if len(l.split()) > 10:
            continue
        clean.append(l)

    if len(clean) < 3:
        fallback = [
            question,
            f"{question} Surgery_90th",
            "Surgery_90th highest values",
        ]
        for f in fallback:
            if f and f not in clean:
                clean.append(f)
            if len(clean) >= 3:
                break

    return clean[:3]


def dedupe_text_hits(hits: List[Tuple[float, str]], keep: int = 12) -> List[Tuple[float, str]]:
    seen = set()
    out: List[Tuple[float, str]] = []
    for score, text in sorted(hits, key=lambda x: x[0]):  # lower is better for FAISS L2
        key = text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((score, text))
        if len(out) >= keep:
            break
    return out


def agentic_csv(question: str, csv_path: str, per_sub_k: int = 6, k_send: int = 8) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    api_key = get_openai_key()
    store = build_faiss(docs, api_key=api_key)

    subqs = make_subqueries(question)

    print("\n=== AGENTIC SUBQUERIES ===")
    for s in subqs:
        print("-", s)

    all_hits: List[Tuple[float, str]] = []
    for sq in subqs:
        all_hits.extend(lc_top_k_with_scores(sq, store, k=per_sub_k))

    best = dedupe_text_hits(all_hits, keep=12)

    print("\n=== AGENTIC (LangChain) RETRIEVED ROWS ===")
    for r, (score, text) in enumerate(best, 1):
        print(f"\n[{r}] score={score:.4f}")
        print(text[:450] + ("..." if len(text) > 450 else ""))

    context = "\n\n".join([f"Row {r}:\n{t[:700]}" for r, (_, t) in enumerate(best[:k_send], 1)])

    system = (
        "You are a helpful assistant. "
        "Use ONLY the provided rows. "
        "If you cannot answer from rows, say what is missing."
    )
    user = (
        f"Question: {question}\n\n"
        f"Subqueries: {subqs}\n\n"
        f"Rows:\n{context}\n\n"
        "Answer:"
    )
    return chat_answer(system, user)


if __name__ == "__main__":
    csv_path = "data/Surgical_Wait_Times_20260207.csv"
    q = "highest Surgery_90th values"
    print("\n=== AGENTIC ANSWER ===\n", agentic_csv(q, csv_path))
