from typing import List, Tuple

from src.common.ns_data import load_csv, df_to_row_docs
from src.common.retrieve import top_k_chunks
from src.common.llm_openai import chat_answer


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
        "- Do not mention countries or outcomes.\n"
        "- Use field names when helpful: " + ", ".join(ALLOWED_FIELDS) + "\n"
        "- Keep each line under 10 words.\n"
        "Examples of good subqueries:\n"
        "Surgery_90th highest values\n"
        "Top Surgery_90th by procedure\n"
        "Maximum Surgery_90th rows\n"
    )

    out = chat_answer(system, f"User question: {question}")

    raw_lines = [l.strip().lstrip("-").strip() for l in out.splitlines() if l.strip()]

    clean: List[str] = []
    for l in raw_lines:
        low = l.lower()
        # Remove obvious code-ish / SQL-ish content
        if "```" in low or "select" in low or "from " in low or "where " in low:
            continue
        if ";" in l or "(" in l or ")" in l:
            # often shows SQL fragments; keep it strict
            continue
        # Keep it short
        if len(l.split()) > 10:
            continue
        clean.append(l)

    # Strong fallback: generate simple keyword variants ourselves
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


def dedupe_hits(hits: List[Tuple[int, float, str]], keep: int = 12) -> List[Tuple[int, float, str]]:
    seen = set()
    out: List[Tuple[int, float, str]] = []
    for i, s, c in sorted(hits, key=lambda x: x[1], reverse=True):
        if i in seen:
            continue
        seen.add(i)
        out.append((i, s, c))
        if len(out) >= keep:
            break
    return out


def agentic_csv(question: str, csv_path: str, per_sub_k: int = 6, k_send: int = 8) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    subqs = make_subqueries(question)

    print("\n=== AGENTIC SUBQUERIES ===")
    for s in subqs:
        print("-", s)

    all_hits: List[Tuple[int, float, str]] = []
    for sq in subqs:
        all_hits.extend(top_k_chunks(sq, docs, k=per_sub_k))

    best = dedupe_hits(all_hits, keep=12)

    print("\n=== AGENTIC RETRIEVED ROWS ===")
    for r, (i, s, c) in enumerate(best, 1):
        print(f"\n[{r}] idx={i} score={s:.4f}")
        print(c[:450] + ("..." if len(c) > 450 else ""))

    context = "\n\n".join([f"Row {i}:\n{c[:700]}" for i, _, c in best[:k_send]])

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
    q = "Compare surgery wait times by zone for Gastrointestinal Tract Surgery."
    print("\n=== AGENTIC ANSWER ===\n", agentic_csv(q, csv_path))
