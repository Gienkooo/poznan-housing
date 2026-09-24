#!/usr/bin/env python3
"""Generate a portfolio-ready artifact bundle for the Poznań housing project.

This script intentionally reuses the analysis logic already living in src/ and
scripts/ so that the notebook narrative and the public-facing artifact export stay
aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBRegressor

from src.data_io import load_data
from src.features import create_price_clusters, prepare_jst_dataframe, prepare_market_dataframe
from src.modeling import generate_shap_explanations
from src.viz import (
    plot_clusters,
    plot_corr_heatmap,
    plot_hexbin,
    plot_missingness,
    plot_price_over_time,
    plot_price_vs_distance,
    plot_target_distribution,
)
from jst_wizualizacje import plot_jst_summary


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_market_dataset() -> pd.DataFrame:
    raw = load_data("rcn_lokale_polaczone_extra_osiedla.csv")
    market = prepare_market_dataframe(raw)
    if "osiedle_cat" not in market.columns and "osiedle" in market.columns:
        market["osiedle_cat"] = market["osiedle"].astype(str)
    if "log_cena_za_m2" not in market.columns:
        market["log_cena_za_m2"] = np.log1p(market["cena_za_m2"])
    return market


def export_core_exploration(market_df: pd.DataFrame, out_dir: Path) -> None:
    plot_missingness(market_df, top_n=20, save_path=str(out_dir / "missingness_by_feature.png"))
    plot_target_distribution(market_df, target_col="cena_za_m2", save_path=str(out_dir / "target_distribution.png"))
    plot_price_over_time(market_df, year_col="rok", target_col="cena_za_m2", save_path=str(out_dir / "price_over_time.png"))
    plot_price_vs_distance(market_df, dist_col="odl_do_centrum", target_col="cena_za_m2", save_path=str(out_dir / "price_vs_distance.png"))

    if {"wsp_x", "wsp_y", "cena_za_m2"}.issubset(market_df.columns):
        plot_hexbin(
            market_df.dropna(subset=["wsp_x", "wsp_y", "cena_za_m2"]),
            x_col="wsp_x",
            y_col="wsp_y",
            value_col="cena_za_m2",
            save_path=str(out_dir / "hexbin_price_map.png"),
        )

    corr_cols = [
        col for col in [
            "cena_za_m2",
            "powierzchnia_final",
            "liczba_izb",
            "odl_do_centrum",
            "rok_miesiac_float",
            "srednie_wynagrodzenie",
            "wibor_3m",
            "metry_na_izbe",
            "is_pierwotny",
            "is_wielorodzinny",
        ] if col in market_df.columns
    ]
    corr_df = market_df[corr_cols].dropna()
    if corr_df.shape[1] > 1:
        plot_corr_heatmap(corr_df, numeric_only=True, save_path=str(out_dir / "correlation_heatmap.png"))

    kde_cols = [
        col for col in ["cena_za_m2", "powierzchnia_final", "odl_do_centrum", "liczba_izb"]
        if col in market_df.columns
    ]
    if len(kde_cols) >= 2:
        kde_df = market_df[kde_cols].dropna().sample(min(4000, len(market_df[kde_cols].dropna())), random_state=42)
        grid = sns.PairGrid(kde_df, diag_sharey=False, corner=True)
        grid.map_diag(sns.kdeplot, fill=True)
        grid.map_offdiag(sns.kdeplot, fill=True, alpha=0.35)
        grid.figure.tight_layout()
        grid.figure.savefig(out_dir / "kde_pairwise_density.png", dpi=180, bbox_inches="tight")
        grid.figure.clf()

    clustered = create_price_clusters(market_df.copy())
    if "price_cluster" in clustered.columns and {"wsp_x", "wsp_y"}.issubset(clustered.columns):
        plot_clusters(
            clustered.dropna(subset=["wsp_x", "wsp_y", "price_cluster"]),
            x_col="wsp_x",
            y_col="wsp_y",
            cluster_col="price_cluster",
            save_path=str(out_dir / "location_clusters.png"),
        )


def export_shap_artifact(market_df: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = [
        "powierzchnia_final",
        "liczba_izb",
        "odl_do_centrum",
        "is_pierwotny",
        "is_uzytkowanie_wieczyste",
        "is_wielorodzinny",
        "wibor_3m",
        "srednie_wynagrodzenie",
        "rok",
        "ma_pomieszczenia_przynalezne",
        "metry_na_izbe",
        "osiedle_cat",
    ]
    feature_cols = [col for col in feature_cols if col in market_df.columns]
    if "osiedle_cat" not in market_df.columns and "osiedle" in market_df.columns:
        market_df = market_df.copy()
        market_df["osiedle_cat"] = market_df["osiedle"].astype(str)
        feature_cols.append("osiedle_cat")

    X = market_df[feature_cols].copy()
    y = np.log1p(market_df["cena_za_m2"].dropna().copy())
    X = X.loc[y.index].copy()

    if X.empty or y.empty:
        return

    model = Pipeline([
        ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )),
    ])
    model.fit(X, y)
    generate_shap_explanations(model, X, sample_n=1500, save_path=str(out_dir / "shap_summary_xgb.png"))


def export_jst_artifacts(raw_df: pd.DataFrame, market_df: pd.DataFrame, out_dir: Path) -> None:
    jst_df = prepare_jst_dataframe(raw_df)
    plot_jst_summary(jst_df, market_df=market_df, save_dir=str(out_dir))


def main() -> None:
    out_dir = ensure_dir(ROOT / "plots")
    raw_df = load_data("rcn_lokale_polaczone_extra_osiedla.csv")
    market_df = prepare_market_dataframe(raw_df)

    export_core_exploration(market_df, out_dir)
    export_shap_artifact(market_df, out_dir)
    export_jst_artifacts(raw_df, market_df, out_dir)

    print(f"Portfolio artifacts exported to: {out_dir}")
    print("Generated files:")
    for path in sorted(out_dir.glob("*")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
