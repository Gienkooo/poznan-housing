from __future__ import annotations
from pathlib import Path
from typing import Optional

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import contextily as ctx
import geopandas as gpd
import warnings
from .config_styling import set_professional_style

warnings.filterwarnings('ignore')


def set_plot_style() -> None:
    """Use the central project-wide visual style configuration."""
    set_professional_style()


def plot_missingness(df: pd.DataFrame, top_n: int = 30, save_path: Optional[str] = None) -> None:
    na_pct = df.isna().mean().sort_values(ascending=False).head(top_n) * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(x=na_pct.index, y=na_pct.values, color="#4c72b0")
    plt.xticks(rotation=90)
    plt.ylabel("Missing values (%)")
    plt.title("Missing values by feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_target_distributions(df: pd.DataFrame, target_cols: list[str], save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 4))
    if len(target_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, target_cols):
        sns.histplot(df[col].dropna(), bins=60, kde=True, ax=ax, color="#55a868")
        ax.set_title(f"Rozkład: {col}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_price_over_time(df: pd.DataFrame, year_col: str, target_col: str, save_path: Optional[str] = None) -> None:
    by_year = df.groupby(year_col)[target_col].agg(["median", "mean"]).reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(by_year[year_col], by_year["median"], marker="o", label="Median")
    plt.plot(by_year[year_col], by_year["mean"], marker="s", label="Mean")
    plt.xlabel("Year")
    plt.ylabel(target_col)
    plt.title("Cena m2 w czasie")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_price_vs_distance(df: pd.DataFrame, dist_col: str, target_col: str, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(12, 5))

    # Scatter plot with transparency
    sns.scatterplot(
        data=df,
        x=dist_col,
        y=target_col,
        s=14,
        alpha=0.35,
        linewidth=0,
        color="#4c72b0",
    )
    plt.xlabel("Distance to center (km)")
    plt.ylabel(target_col)
    plt.title("Price per m2 vs distance to center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Binned median with SAME COLOR
    bins = pd.qcut(df[dist_col], q=10, duplicates="drop")
    binned = df.groupby(bins)[target_col].agg(['median', 'count']).reset_index()
    binned['bin_center'] = binned[dist_col].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(binned)), binned['median'], color="#4c72b0", alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(binned)))
    ax.set_xticklabels([f"{x:.1f}" for x in binned['bin_center']], rotation=45)
    ax.set_xlabel("Distance deciles (bin center, km)")
    ax.set_ylabel(f"Median {target_col}")
    ax.set_title("Median price per m2 by distance deciles (consistent color)")
    plt.tight_layout()
    if save_path:
        plt.savefig(str(Path(save_path).with_name(Path(save_path).stem + "_binned.png")), dpi=150, bbox_inches='tight')
    plt.show()


def plot_hexbin(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, save_path: Optional[str] = None) -> None:
    vmin = df[value_col].quantile(0.05)
    vmax = df[value_col].quantile(0.95)
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(
        df[x_col],
        df[y_col],
        C=df[value_col],
        reduce_C_function=np.median,
        gridsize=45,
        cmap="viridis",
        mincnt=5,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(hb, label=f"Median {value_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Przestrzenny hexbin: mediana ceny za m2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_corr_heatmap(df: pd.DataFrame, numeric_only: bool = True, save_path: Optional[str] = None) -> None:
    corr = df.corr(numeric_only=numeric_only)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Much larger figure for readability
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9}  # Smaller numbers, still readable
    )
    plt.title("Mapa ciepła korelacji (dolny trójkąt)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.show()


def plot_clusters(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=cluster_col,
        palette="tab20",
        s=16,
        alpha=0.7,
        linewidth=0,
    )
    plt.title("Location clusters")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def add_clean_basemap(
    ax: plt.Axes,
    zoom: str | int = "auto",
    alpha: float = 1.0,
    zorder: int | None = None,
) -> None:
    """Adds a clean, watermark-free basemap with robust fallbacks.
    
    Avoids CartoDB (which requires an API key and renders watermarks).
    Prioritizes Esri.WorldGrayCanvas and Esri.WorldStreetMap, followed by
    raw tile URLs, OpenStreetMap, and an offline styled background.
    """
    kwargs: dict = {}
    if zorder is not None:
        kwargs["zorder"] = zorder
    if alpha != 1.0:
        kwargs["alpha"] = alpha

    # 1. Esri WorldGrayCanvas (clean, light neutral basemap without API key requirement)
    try:
        if hasattr(ctx.providers, "Esri") and hasattr(ctx.providers.Esri, "WorldGrayCanvas"):
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, attribution_size=6, zoom=zoom, **kwargs)
            return
    except Exception:
        pass

    try:
        esri_gray_url = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ctx.add_basemap(ax, source=esri_gray_url, attribution="Tiles © Esri", attribution_size=6, zoom=zoom, **kwargs)
        return
    except Exception:
        pass

    # 2. Esri WorldStreetMap (standard street map without API key)
    try:
        if hasattr(ctx.providers, "Esri") and hasattr(ctx.providers.Esri, "WorldStreetMap"):
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldStreetMap, attribution_size=6, zoom=zoom, **kwargs)
            return
    except Exception:
        pass

    try:
        esri_street_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ctx.add_basemap(ax, source=esri_street_url, attribution="Tiles © Esri", attribution_size=6, zoom=zoom, **kwargs)
        return
    except Exception:
        pass

    # 3. OpenStreetMap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=6, zoom=zoom, **kwargs)
        return
    except Exception:
        pass

    # 4. Offline fallback: neutral canvas
    ax.set_facecolor("#f4f5f7")


def plot_clusters_on_map(
    df: pd.DataFrame,
    x_col: str,        # wsp_x (northing EPSG:2177)
    y_col: str,        # wsp_y (easting EPSG:2177)
    cluster_col: str,
    crs_input: str = "EPSG:2177",
    title: str = "Location clusters",
    save_path: Optional[str] = None
) -> None:
    sample = df.dropna(subset=[x_col, y_col])
    gdf = gpd.GeoDataFrame(
        sample,
        geometry=gpd.points_from_xy(sample[x_col], sample[y_col]),
        crs=crs_input,
    ).to_crs("EPSG:3857")  # contextily wymaga Web Mercator

    fig, ax = plt.subplots(figsize=(10, 10))

    clusters = gdf[cluster_col].unique()
    palette = sns.color_palette("Dark2", len(clusters))
    color_map = dict(zip(sorted(clusters), palette))

    for cluster, group in gdf.groupby(cluster_col):
        group.plot(
            ax=ax,
            color=color_map[cluster],
            markersize=6,
            alpha=0.6,
            label=str(cluster),
        )

    # Basemap with fallback
    add_clean_basemap(ax)

    ax.set_axis_off()
    ax.set_title(title, fontsize=14, pad=12)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[c], markersize=8, label=str(c))
        for c in sorted(clusters)
    ]
    ax.legend(handles=legend_elements, title=cluster_col,
              bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def silhouette_analysis_hdbscan(
    X: pd.DataFrame, 
    cluster_labels: pd.Series,
    min_cluster_size_range: list = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """Analyze silhouette scores for HDBSCAN with different min_cluster_size values."""
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import StandardScaler
    
    if min_cluster_size_range is None:
        min_cluster_size_range = [20, 50, 100, 200, 300]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    for min_size in min_cluster_size_range:
        hdb = HDBSCAN(min_cluster_size=min_size, min_samples=15)
        labels = hdb.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1:
            # Silhouette only on clustered points (exclude noise -1)
            mask = labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(X_scaled[mask], labels[mask])
            else:
                sil_score = np.nan
        else:
            sil_score = np.nan
        
        results.append({
            'min_cluster_size': min_size,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_pct': 100 * n_noise / len(labels),
            'silhouette_score': sil_score
        })
    
    df_results = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(df_results['min_cluster_size'], df_results['n_clusters'], 
                 marker='o', label='# Clusters', linewidth=2, markersize=8)
    axes[0].axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Suggested: 100')
    axes[0].set_xlabel('min_cluster_size')
    axes[0].set_ylabel('Number of clusters')
    axes[0].set_title('HDBSCAN: Cluster count vs min_cluster_size')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(df_results['min_cluster_size'], df_results['silhouette_score'], 
                 marker='s', label='Silhouette Score', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('min_cluster_size')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('HDBSCAN: Silhouette score quality')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return df_results


def cluster_profiling(df: pd.DataFrame, cluster_col: str, target_col: str = 'cena_za_m2', save_path: Optional[str] = None) -> pd.DataFrame:
    """Profile clusters by size, mean price, mean distance, etc."""
    
    profile = df.groupby(cluster_col, observed=True).agg({
        target_col: ['mean', 'median', 'std', 'count'],
        'odl_do_centrum': ['mean', 'min', 'max'],
        'wsp_x': 'mean',
        'wsp_y': 'mean',
    }).round(2)
    
    profile.columns = ['_'.join(col).strip() for col in profile.columns.values]
    profile = profile.rename(columns={
        f'{target_col}_mean': 'avg_price_m2',
        f'{target_col}_median': 'median_price_m2',
        f'{target_col}_std': 'std_price_m2',
        f'{target_col}_count': 'n_properties',
        'odl_do_centrum_mean': 'avg_dist_km',
        'odl_do_centrum_min': 'min_dist_km',
        'odl_do_centrum_max': 'max_dist_km',
        'wsp_x_mean': 'center_x',
        'wsp_y_mean': 'center_y',
    })
    
    profile = profile.sort_values('n_properties', ascending=False)
    
    print("=== Cluster Profile ===")
    print(profile.to_string())
    
    return profile


def plot_target_distribution(df: pd.DataFrame, target_col: str = 'cena_za_m2', save_path: Optional[str] = None) -> None:
    """Plots original and log-transformed target distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[target_col].dropna(), kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Rozkład {target_col}')
    axes[0].set_xlabel('Price (PLN)')

    sns.histplot(np.log1p(df[target_col].dropna()), kde=True, ax=axes[1], color='salmon')
    axes[1].set_title(f'Log Rozkład {target_col}')
    axes[1].set_xlabel('Log(Price)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plots model comparison with error bars (std deviation)."""
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(
        x=range(len(results_df)),
        y=results_df['Mean_CV_R2'],
        yerr=results_df['Std_CV_R2'],
        fmt='o',
        capsize=5,
        capthick=2,
        ecolor='black',
        markerfacecolor='royalblue',
        markersize=10
    )
    
    plt.xticks(range(len(results_df)), results_df['Model'], rotation=15, ha='right')
    plt.title('Model Comparison: CV R² Score with Standard Deviation', pad=20)
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> None:
    """Plots residuals for model evaluation."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: predicted vs residuals
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_title(f'{model_name}: Residuals vs Predicted')
    axes[0].set_xlabel('Predicted Price (log-space)')
    axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram: residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_title(f'{model_name}: Distribution of Residuals')
    axes[1].set_xlabel('Residual Error')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_yearly_performance_vs_wibor(
    yearly_df: pd.DataFrame,
    wibor_map: dict | None = None,
    save_path: str | None = None,
) -> None:
    """Dual-axis chart: per-year R² bars + WIBOR 3M overlay with regime bands.

    Args:
        yearly_df: Output of the expanding-window loop (columns: year, r2)
        wibor_map: Dict {year: wibor_rate}. Defaults to Polish WIBOR 2013–2025.
        save_path: Optional path to save PNG (e.g. 'output/wibor_chart.png')
    """
    if wibor_map is None:
        wibor_map = {
            2013: 3.0, 2014: 2.5, 2015: 1.7, 2016: 1.7, 2017: 1.7,
            2018: 1.7, 2019: 1.7, 2020: 0.7, 2021: 0.5,
            2022: 6.0, 2023: 6.5, 2024: 5.8, 2025: 5.8,
        }
    wibor_s = pd.Series(wibor_map)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.axvspan(2013, 2015,  alpha=0.06, color='green')
    ax1.axvspan(2015, 2021,  alpha=0.06, color='steelblue')
    ax1.axvspan(2021, 2025.5, alpha=0.06, color='red')

    colors = ['#4C78A8' if r >= 0 else '#d95f02' for r in yearly_df['r2']]
    ax1.bar(yearly_df['year'], yearly_df['r2'],
            color=colors, alpha=0.8, width=0.5, label='Out-of-time R²')
    ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax1.set_ylabel('R² (out-of-time)', fontsize=11)

    r2_min = float(yearly_df['r2'].min())
    r2_max = float(yearly_df['r2'].max())
    lower = min(0.0, r2_min)
    upper = max(0.0, r2_max)
    pad = max(0.25, (upper - lower) * 0.35)
    ax1.set_ylim(lower - pad, upper + pad)
    ax1.set_xlabel('Rok testowy', fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(wibor_s.index, wibor_s.values,
             color='crimson', linewidth=2.5, marker='s',
             markersize=6, linestyle='--', label='WIBOR 3M (%)')
    ax2.set_ylabel('WIBOR 3M (%)', fontsize=11, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.set_ylim(0, 10)

    ax1.text(2013.5, 0.50, 'Początek odczytów',       fontsize=9, color='green',     alpha=0.8)
    ax1.text(2016.5, 0.50, 'Stabilna sytuacja', fontsize=9, color='steelblue', alpha=0.8)
    ax1.text(2022.2, 0.50, 'Obszar po szoku',    fontsize=9, color='red',       alpha=0.8)

    r2_2020 = yearly_df.loc[yearly_df['year'] == 2020, 'r2'].values
    r2_2022 = yearly_df.loc[yearly_df['year'] == 2022, 'r2'].values
    if len(r2_2020):
        ax1.annotate('COVID\nlockdown', xy=(2020, r2_2020[0]),
                     xytext=(2019.3, 0.43), fontsize=8, color='orange',
                     arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    if len(r2_2022):
        ax1.annotate('skok\nWIBOR', xy=(2022, r2_2022[0]),
                     xytext=(2022.3, 0.40), fontsize=8, color='crimson',
                     arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.set_title(
        'XGBoost rozszerzające się okno R² vs WIBOR 3M\n'
        'Dane treningowe: wszystkie dane do roku Y  |  Dane testowe: tylko rok Y',
        fontsize=12, fontweight='bold',
    )
    ax1.set_xticks(yearly_df['year'])
    ax1.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_mae_trend(
    yearly_df: pd.DataFrame,
    stable_years: tuple[int, int] = (2016, 2021),
    shock_year_start: int = 2022,
    save_path: str | None = None,
) -> dict:
    """MAE over time with stable-era vs post-shock reference lines.

    Args:
        yearly_df: Output of expanding-window loop (columns: year, mae_pln)
        stable_years: Inclusive (start, end) for stable-era reference band
        shock_year_start: First year of post-shock period
        save_path: Optional PNG save path

    Returns:
        Dict with stable_mae, shock_mae, delta_pct for report citation
    """
    stable_mae = yearly_df.loc[
        yearly_df['year'].between(*stable_years), 'mae_pln'
    ].mean()
    shock_mae = yearly_df.loc[
        yearly_df['year'] >= shock_year_start, 'mae_pln'
    ].mean()
    delta_pct = (shock_mae - stable_mae) / stable_mae * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(yearly_df['year'], yearly_df['mae_pln'],
                    alpha=0.2, color='steelblue')
    ax.plot(yearly_df['year'], yearly_df['mae_pln'],
            marker='o', linewidth=2.5, markersize=8,
            color='steelblue', label='MAE (PLN/m²)')
    ax.axhline(stable_mae, color='green', linestyle='--', linewidth=1.5,
               label=f'Stabilna sytuacja ({stable_years[0]}–{stable_years[1]}): {stable_mae:,.0f} PLN/m²')
    ax.axhline(shock_mae,  color='red',   linestyle='--', linewidth=1.5,
               label=f'Obszar po szoku ({shock_year_start}+): {shock_mae:,.0f} PLN/m²')

    ax.set_ylabel('MAE (PLN/m²)', fontsize=11)
    ax.set_xlabel('Rok testowy', fontsize=11)
    ax.set_title('Model MAE w czasie — stabilna sytuacja vs pogorszenie po szoku',
                 fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.legend(fontsize=10)
    ax.set_xticks(yearly_df['year'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {'stable_mae': stable_mae, 'shock_mae': shock_mae, 'delta_pct': delta_pct}


def plot_mutual_information(
    mi_scores: pd.Series,
    threshold: float = 0.05,
    save_path: Optional[str] = None
) -> None:
    """Bar chart of mutual information scores with selection threshold.

    Args:
        mi_scores: Series of MI scores indexed by feature name (sorted desc)
        threshold: MI value line marking the selection cutoff
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ['#2196a8' if v > threshold else '#b0c4ce' for v in mi_scores.values]
    mi_scores.plot(kind='bar', ax=ax, color=colors, edgecolor='none')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.2,
               label=f'MI = {threshold} threshold')
    ax.set_title(
        'Mutual Information — nieliniowy sygnał cech\n'
        'Niebieski: ponad obszarem obcięcia | Szary: graniczny / wykluczony',
        fontsize=11,
    )
    ax.set_ylabel('MI Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_distributions(df: pd.DataFrame, cols: Optional[dict] = None, clip_quantile: float = 0.99, save_path: Optional[str] = None):
    """
    Violin + boxplot hybrid with quantile markers.
    cols: list of (col_name, label) tuples
    """
    if cols is None:
        cols = {
            'cena_za_m2':         'Cena za m² [PLN]',
            'powierzchnia_final': 'Powierzchnia [m²]',
            'metry_na_izbe':      'Metry na izbę [m²]',
            'liczba_izb':         'Liczba izb',
            'nr_kondygnacji':     'Nr kondygnacji',
            'odl_do_centrum':     'Odl do centrum [km]',
        }

    n = len(cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    i = -1
    for i, (col, label) in enumerate(cols.items()):
        ax = axes[i]
        data = df[col].dropna()

        # Clip to clip_quantile to avoid violins being crushed by outliers
        cap = data.quantile(clip_quantile)
        plot_data = data[data <= cap]

        # Violin
        parts = ax.violinplot(plot_data, vert=True, showmedians=False,
                               showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.4)

        # Overlay boxplot (IQR box + median line)
        bp = ax.boxplot(plot_data, vert=True, patch_artist=True,
                        widths=0.15, manage_ticks=False,
                        medianprops=dict(color='white', linewidth=2),
                        boxprops=dict(facecolor='steelblue', alpha=0.8),
                        whiskerprops=dict(color='steelblue'),
                        capprops=dict(color='steelblue'),
                        flierprops=dict(marker='.', markersize=1,
                                        alpha=0.3, color='steelblue'))

        # Quantile markers on the right side
        for q, c, ls in [(0.01, 'red', '--'), (0.05, 'orange', '--'),
                          (0.95, 'orange', ':'),  (0.99, 'red', ':')]:
            val = data.quantile(q)  # use unclipped data for true quantiles
            ax.axhline(val, color=c, linestyle=ls, alpha=0.7, linewidth=2,
                       label=f'p{int(q*100)}={val:.0f}')

        ax.set_title(col, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks([])
        ax.legend(fontsize=7, loc='upper right')

        # Annotate clipping if data was capped
        if data.max() > cap:
            n_clipped = (data > cap).sum()
            ax.text(0.02, 0.97, f'{n_clipped} outlierów odrzucono',
                    transform=ax.transAxes, fontsize=7,
                    color='gray', va='top')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Rozkłady zmiennych', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    
def plot_correlation_matrix(df: pd.DataFrame, target_col: str = 'cena_za_m2', top_n: int = 15, save_path: Optional[str] = None) -> None:
    """Plots a professional correlation matrix for the top correlated features."""
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    if target_col not in corr.columns:
        print(f"Target '{target_col}' not found.")
        return
    
    top_cols = corr[target_col].abs().nlargest(top_n).index
    top_corr = df[top_cols].corr(numeric_only=True)
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    
    sns.heatmap(top_corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5, 
                cbar_kws={"shrink": .7})
    plt.title(f'Top {top_n} Macierz korelacji cech (Pearson)', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_target_relationships(df: pd.DataFrame, target_col: str, features: list[str], save_path: Optional[str] = None) -> None:
    """Plots scatter plots with regression lines to visually find higher-order relationships."""
    num_features = len(features)
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.regplot(data=df, x=feature, y=target_col, ax=axes[i], 
                    scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'red'})
        axes[i].set_title(f'{feature} vs {target_col}')
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_model_comparison_cv(cv_results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plots boxplots of Cross-Validation folds to show variance and stability.
    
    If Fold_R2_Scores is available, renders a fold boxplot.
    Otherwise, falls back to an errorbar plot (Mean_CV_R2 ± Std_CV_R2).
    """
    plt.figure(figsize=(10, 6))

    if 'Fold_R2_Scores' in cv_results_df.columns:
        melted_df = pd.melt(cv_results_df, id_vars=['Model'], value_vars=['Fold_R2_Scores'],
                            var_name='Metric', value_name='R2 Score')
        melted_df = melted_df.explode('R2 Score')
        melted_df['R2 Score'] = melted_df['R2 Score'].astype(float)
        
        sns.boxplot(data=melted_df, x='Model', y='R2 Score', showmeans=True, 
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        plt.title('Walidacja krzyżowa R²: Rozkład pomiędzy foldami\n(Biały punkt = Średnia)', fontsize=12)
        plt.ylabel('R² Score', fontsize=11)
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3, axis='y')
    elif 'Mean_CV_R2' in cv_results_df.columns and 'Std_CV_R2' in cv_results_df.columns:
        plt.errorbar(
            x=range(len(cv_results_df)),
            y=cv_results_df['Mean_CV_R2'],
            yerr=cv_results_df['Std_CV_R2'],
            fmt='o',
            capsize=5,
            capthick=2,
            ecolor='black',
            markerfacecolor='royalblue',
            markersize=10
        )
        plt.xticks(range(len(cv_results_df)), cv_results_df['Model'], rotation=15, ha='right')
        plt.title('Porównanie modeli: Walidacja krzyżowa R² (Średnia ± Odch. std.)', pad=20, fontsize=12)
        plt.ylabel('R² Score', fontsize=11)
        plt.grid(True, alpha=0.3)
    else:
        raise KeyError(
            f"Expected 'Fold_R2_Scores' or ('Mean_CV_R2', 'Std_CV_R2') in dataframe columns: {list(cv_results_df.columns)}"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def plot_kmeans_silhouette_analysis(df_results: pd.DataFrame, title_prefix: str = "KMeans", save_path: Optional[str] = None) -> None:
    """Plots inertia and silhouette scores side-by-side to justify optimal k."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df_results.index, df_results['inertia'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Liczba skupisk (k)')
    axes[0].set_ylabel('Bezwładność (Suma kwadratów odległości)')
    axes[0].set_title(f'{title_prefix}: Metoda łokciowa')
    axes[0].grid(True, alpha=0.3)
    
    best_k = df_results['silhouette'].idxmax()
    axes[1].plot(df_results.index, df_results['silhouette'], marker='s', 
                 linewidth=2, markersize=8, color='green')
    axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.5, 
                    label=f'Max Silhouette: k={best_k}')
    axes[1].set_xlabel('Liczba skupisk (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f'{title_prefix}: Silhouette Analysis')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_bivariate_density(df: pd.DataFrame, features: list[str], sample_n: int = 5000, save_path: Optional[str] = None) -> None:
    """Plots a PairGrid with density lower triangle for clean viewing of relationships."""
    sample_df = df[features].sample(n=min(sample_n, len(df)), random_state=42)
    g = sns.PairGrid(sample_df, diag_sharey=False, corner=True, height=2.8)
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues", alpha=0.8, thresh=0.05)
    g.map_diag(sns.histplot, kde=True, color="steelblue", element="step")
    plt.suptitle(f"Zależności dwuwymiarowe (widok gęstości dla {sample_n} próbek)", y=1.02, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_property_segments(df: pd.DataFrame, cluster_col: str = 'property_segment', crs_input: str = 'EPSG:2177', save_path: Optional[str] = None) -> None:
    """Plots target distribution, price vs area, and separate spatial maps with basemaps."""
    
    fig_stats, axes_stats = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(data=df, x=cluster_col, y='cena_za_m2', ax=axes_stats[0], palette='Set2')
    axes_stats[0].set_title('Rozkład ceny za m² według segmentów nieruchomości')
    axes_stats[0].set_xlabel('Segment nieruchomości')
    axes_stats[0].set_ylabel('Cena za m2 (PLN)')

    sns.scatterplot(data=df, 
                    x='powierzchnia_final', y='cena_za_m2', hue=cluster_col, 
                    palette='Set2', alpha=0.5, ax=axes_stats[1], edgecolor=None)
    axes_stats[1].set_title('Cena vs Powierzchnia z podziałem na segmenty nieruchomości')
    axes_stats[1].set_xlabel('Powierzchnia (m2)')
    
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        plt.savefig(str(p.with_name(p.stem + "_stats" + p.suffix)), dpi=150, bbox_inches='tight')

    plt.show()

    sample_geo = df.dropna(subset=['wsp_x', 'wsp_y'])
    
    gdf = gpd.GeoDataFrame(
        sample_geo,
        geometry=gpd.points_from_xy(sample_geo['wsp_y'], sample_geo['wsp_x']),
        crs=crs_input,
    ).to_crs("EPSG:3857")

    unique_segments = sorted(gdf[cluster_col].unique())
    n_segments = len(unique_segments)

    palette = sns.color_palette("Dark2", n_segments)
    color_map = dict(zip(unique_segments, palette))

    minx, miny, maxx, maxy = gdf.total_bounds

    fig_maps, axes_maps = plt.subplots(1, n_segments, figsize=(5 * n_segments, 6))

    if n_segments == 1:
        axes_maps = [axes_maps]

    for ax, segment in zip(axes_maps, unique_segments):
        subset = gdf[gdf[cluster_col] == segment]
        
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color_map[segment],
                markersize=15,
                alpha=0.7,
                edgecolor='none'
            )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        add_clean_basemap(ax)

        ax.set_axis_off()
        ax.set_title(f'Segment: {segment}', fontsize=14, pad=12)

    fig_maps.suptitle('Rozkład geograficzny nieruchomości z podziałem na segmenty', fontsize=16, y=1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_temporal_segments(df: pd.DataFrame, cluster_col: str = 'temporal_segment', crs_input: str = 'EPSG:2177', save_path: Optional[str] = None) -> None:
    """Plots price distribution, market eras over time, and geographical shifts on separate maps."""

    fig_stats, axes_stats = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x=cluster_col, y='cena_za_m2', palette='viridis', ax=axes_stats[0])
    axes_stats[0].set_title('Rozkład cen w różnych reżimach rynkowych')
    axes_stats[0].set_xlabel('Segment czasowy')
    axes_stats[0].set_ylabel('Cena za m2 (PLN)')

    sns.scatterplot(data=df, x='rok_miesiac_float', y='cena_za_m2', hue=cluster_col, 
                    palette='viridis', alpha=0.4, ax=axes_stats[1], edgecolor=None)
    axes_stats[1].set_title('Okresy rynkowe wykryte przez klasteryzację')
    axes_stats[1].set_xlabel('Czas (rok.miesiąc)')
    axes_stats[1].set_ylabel('Cena za m2 (PLN)')
    axes_stats[1].legend(title='Segment czasowy', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        plt.savefig(str(p.with_name(p.stem + "_stats" + p.suffix)), dpi=150, bbox_inches='tight')

    plt.show()

    sample_geo = df.dropna(subset=['wsp_x', 'wsp_y'])

    gdf = gpd.GeoDataFrame(
        sample_geo,
        geometry=gpd.points_from_xy(sample_geo['wsp_y'], sample_geo['wsp_x']),
        crs=crs_input,
    ).to_crs("EPSG:3857")

    unique_segments = sorted(gdf[cluster_col].unique())
    n_segments = len(unique_segments)
    
    palette = sns.color_palette("viridis", n_segments)
    color_map = dict(zip(unique_segments, palette))

    minx, miny, maxx, maxy = gdf.total_bounds

    fig_maps, axes_maps = plt.subplots(1, n_segments, figsize=(10 * n_segments, 12))

    if n_segments == 1:
        axes_maps = [axes_maps]

    for ax, segment in zip(axes_maps, unique_segments):
        subset = gdf[gdf[cluster_col] == segment]
        
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color_map[segment],
                markersize=4, 
                alpha=0.1,
                edgecolor='none'
            )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        add_clean_basemap(ax)

        ax.set_axis_off()
        ax.set_title(f'Epoka: {segment}', fontsize=14, pad=12)

    fig_maps.suptitle('Geograficzna zmiana transakcji w poszczególnych epokach', fontsize=16, y=1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
