from typing import List, Tuple

Hit = Tuple[int, float, str]

def build_context(
    hits: List[Hit],
    max_chars: int = 3500,
    per_row_chars: int = 280,
    max_rows: int = 10,
) -> str:
    """
    Build a tight context string.
    - Limits number of rows
    - Limits per-row snippet length
    - Limits total chars
    """
    parts = []
    used = 0

    for idx, score, row in hits[:max_rows]:
        snippet = (row or "").strip().replace("\n", " ")
        snippet = snippet[:per_row_chars]

        block = f"Row {idx} (score={score:.3f}): {snippet}"
        if used + len(block) + 1 > max_chars:
            break

        parts.append(block)
        used += len(block) + 1

    return "\n".join(parts)
