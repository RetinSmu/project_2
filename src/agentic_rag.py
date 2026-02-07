from typing import List, Tuple

from src.common.ns_data import load_csv, df_to_row_docs
from src.common.retrieve import top_k_chunks
from src.common.context_limit import build_context
from src.common.llm_openai import chat_answer

ALLOWED_FIELDS = [
    "Period", "Year", "Quarter", "Specialty", "Procedure",
    "Provider", "Zone", "Facility", "Consult_Median",
    "Consult_90th", "Surgery_Median", "Surgery_90th",
]

Hit = Tuple[int, float, str]


def _clean_subquery_lines(text: str, fallback: str) -> List[str]:
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip().lstrip("-").strip()
        if not s:
            continue
        if s.startswith("```"):
            continue
        # block obvious SQL / irrelevant stuff
        upper = s.upper()
        if "SELECT " in upper or "FROM " in upper or "WHERE " in upper:
            continue
        if "COUNTRY" in upper or "OUTCOME" in upper:
            continue
        lines.append(s)

    if not lines:
        return [fallback, fallback, fallback]

    while len(lines) < 3:
        lines.append(fallback)

    return lines[:3]


def make_subqueries(question: str) -> List[str]:
    system = (
        "You generate dataset search subqueries for Nova Scotia surgical wait times CSV.\n"
        "Rules:\n"
        "- Stay ONLY in this dataset.\n"
        "- No SQL, no code blocks.\n"
        "- No countries, no outcomes, no external info.\n"
        "- Prefer these field names when helpful: "
        + ", ".join(ALLOWED_FIELDS)
        + "\n"
        "- Output EXACTLY 3 short subqueries, one per line."
    )
    out = chat_answer(system, f"User question: {question}")
    return _clean_subquery_lines(out, fallback=question)


def dedupe_hits(hits: List[Hit], keep: int = 30) -> List[Hit]:
    seen = set()
    out: List[Hit] = []
    for i, s, c in sorted(hits, key=lambda x: x[1], reverse=True):
        if i in seen:
            continue
        seen.add(i)
        out.append((i, s, c))
        if len(out) >= keep:
            break
    return out


def agentic_csv(question: str, csv_path: str, per_sub_k: int = 6, k_send: int = 10) -> str:
    df = load_csv(csv_path)
    docs = df_to_row_docs(df, max_rows=5000)

    subqs = make_subqueries(question)

    # retrieve
    all_hits: List[Hit] = []
    for sq in subqs:
        all_hits.extend(top_k_chunks(sq, docs, k=per_sub_k))

    best = dedupe_hits(all_hits, keep=40)

    # ✅ RED RULE: small, filtered context
    context = build_context(
        best,
        max_chars=3500,
        per_row_chars=280,
        max_rows=k_send,
    )

    system = (
        "You are a helpful assistant.\n"
        "Use ONLY the provided rows.\n"
        "Return HTML only (no markdown).\n"
        "Format exactly:\n"
        "<h3>Answer</h3><p>...</p>"
        "<h4>Key rows used</h4><ul><li>Row ...</li></ul>"
        "<h4>Missing</h4><p>...</p>\n"
        "If you cannot answer from rows, say what is missing."
    )

    user = (
        f"Question: {question}\n"
        f"Subqueries: {subqs}\n\n"
        f"Rows:\n{context}\n\n"
        "Answer:"
    )
    return chat_answer(system, user)


if __name__ == "__main__":
    csv_path = "data/Surgical_Wait_Times_20260207.csv"
    q = "Compare surgery wait times by zone for Gastric Surgery."
    print(agentic_csv(q, csv_path))
