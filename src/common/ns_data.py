import pandas as pd

def load_csv(path_or_url: str) -> pd.DataFrame:
    # works for local file paths and URLs
    return pd.read_csv(path_or_url)

def df_to_text(df: pd.DataFrame, max_rows: int = 300, max_cols: int = 25) -> str:
    use_cols = list(df.columns[:max_cols])
    small = df[use_cols].head(max_rows).copy()
    small = small.fillna("")
    return small.to_string(index=False)
