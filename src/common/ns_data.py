from typing import List
import pandas as pd

IMPORTANT_COLS = [
    "Period", "Year", "Quarter",
    "Specialty", "Procedure",
    "Provider", "Zone", "Facility",
    "Consult_Median", "Consult_90th",
    "Surgery_Median", "Surgery_90th",
]

def load_csv(path_or_url: str) -> pd.DataFrame:
    return pd.read_csv(path_or_url)

def df_to_row_docs(df: pd.DataFrame, max_rows: int = 5000) -> List[str]:
    df2 = df.copy()

    # keep only known columns that exist
    cols = [c for c in IMPORTANT_COLS if c in df2.columns]
    if not cols:
        cols = list(df2.columns)

    df2 = df2[cols].head(max_rows).fillna("")

    docs: List[str] = []
    for _, row in df2.iterrows():
        # row -> compact searchable sentence
        parts = []
        for c in cols:
            v = str(row[c]).strip()
            if v != "" and v.lower() != "nan":
                parts.append(f"{c}: {v}")
        docs.append(" | ".join(parts))
    return docs
