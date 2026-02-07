# true max feature

import pandas as pd
from src.common.ns_data import load_csv


def top_by_surgery_90th(csv_path: str, n: int = 10) -> str:
    # nicer printing in terminals
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    df = load_csv(csv_path).copy()
    df["Surgery_90th"] = pd.to_numeric(df.get("Surgery_90th"), errors="coerce")
    df["Surgery_Median"] = pd.to_numeric(df.get("Surgery_Median"), errors="coerce")

    df = df.dropna(subset=["Surgery_90th"])
    top = df.sort_values("Surgery_90th", ascending=False).head(n)

    cols = [c for c in [
        "Period", "Year", "Quarter", "Specialty", "Procedure",
        "Provider", "Zone", "Facility", "Surgery_Median", "Surgery_90th"
    ] if c in top.columns]

    # keep output readable
    return top[cols].fillna("").to_string(index=False, max_colwidth=35)


if __name__ == "__main__":
    print(top_by_surgery_90th("data/Surgical_Wait_Times_20260207.csv", n=10))
