from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_data
from src.features import (
    prepare_market_dataframe,
    build_residential_mask,
    clip_target_outliers,
)


def build_processed_dataset() -> Path:
    """Build the processed residential market dataset using the canonical pipeline.

    Uses the same functions as Notebook 01 to ensure reproducibility:
    prepare_market_dataframe → build_residential_mask → clip_target_outliers.
    """
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "rcn_lokale_polaczone_extra_osiedla.csv"
    out_dir = root / "data" / "processed"
    out_path = out_dir / "housing_residential_market.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    df = load_data(str(raw_path))
    df = prepare_market_dataframe(df)

    residential_mask = build_residential_mask(df)
    df_res = df[residential_mask].copy()
    df_res = clip_target_outliers(df_res, target_col="cena_za_m2", lo_q=0.01, hi_q=0.99)

    if df_res.empty:
        raise ValueError(
            "No rows remain after the residential filters. "
            "Verify the source file version before continuing."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path}")
    print(f"Rows: {len(df_res):,}")
    return out_path


if __name__ == "__main__":
    build_processed_dataset()

