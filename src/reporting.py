from __future__ import annotations

import pandas as pd


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [len(df.columns)],
            "missing_pct": [float(df.isna().mean().mean() * 100)],
        }
    )
