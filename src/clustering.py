from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def fit_spatial_clustering(
    X: pd.DataFrame,
    min_cluster_size: int = 100,
) -> tuple[pd.Series, HDBSCAN, StandardScaler]:
    """HDBSCAN clustering on spatial coordinates.

    METHODOLOGY (from silhouette analysis):
    - min_cluster_size=100: optimal balance between granularity and separation
    - Reduces fragmentation from 624 clusters (min_size=50) to ~30-50 clusters
    - Noise points (label=-1) represent geographic outliers, kept for analysis

    Args:
        X: DataFrame with spatial columns (wsp_x, wsp_y)
        min_cluster_size: HDBSCAN minimum cluster size (default=100)

    Returns:
        (cluster_labels, model, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_cluster_size // 2)
    labels = model.fit_predict(X_scaled)
    return pd.Series(labels, index=X.index, name="cluster"), model, scaler


def fit_property_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: list | None = None,
) -> tuple[pd.Series, KMeans, StandardScaler]:
    """KMeans clustering on physical property attributes.

    Segments properties by what they are (size, price, location, floor).
    Must be run on a pre-filtered residential DataFrame (df_res) to avoid
    commercial/warehouse records dominating cluster centroids.

    Args:
        df: Residential-filtered DataFrame (output of build_residential_mask)
        n_clusters: Number of property segments
        features: Override default feature list

    Returns:
        (labels, kmeans_model, scaler)
    """
    if features is None:
        features = [
            'powierzchnia_final',
            'metry_na_izbe',
            'odl_do_centrum',
            'nr_kondygnacji',
        ]
    df_cl = df[features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cl)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = pd.Series(kmeans.fit_predict(X), index=df_cl.index, name='property_segment')
    return labels, kmeans, scaler


def fit_temporal_clustering(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: list | None = None,
) -> tuple[pd.Series, KMeans, StandardScaler]:
    """KMeans market regime clustering on macro features — no target leakage.

    Identifies structural market regimes (post-GFC, pre-boom, COVID QE,
    rate-hike transition, post-hike premium) using only temporal/macro
    features. cena_za_m2 is deliberately excluded to prevent circular
    feature engineering.

    Args:
        df: Residential-filtered DataFrame (output of build_residential_mask)
        n_clusters: Number of market regimes
        features: Override default feature list

    Returns:
        (labels, kmeans_model, scaler)
    """
    if features is None:
        features = ['rok_miesiac_float', 'wibor_3m', 'srednie_wynagrodzenie']
    df_cl = df[features].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cl)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = pd.Series(kmeans.fit_predict(X), index=df_cl.index, name='temporal_segment')
    return labels, kmeans, scaler


def profile_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    target_col: str = 'cena_za_m2',
    extra_cols: list | None = None,
    feature_cols: list | None = None,   # ← NOWY parametr
) -> pd.DataFrame:
    agg_cols = {
        target_col: ['mean', 'median', 'std', 'count'],
        'odl_do_centrum': ['mean', 'min', 'max'],
        'wsp_x': 'mean',
        'wsp_y': 'mean',
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}

    profile = df.groupby(cluster_col, observed=True).agg(agg_cols).round(2)
    profile.columns = ['_'.join(c).strip() for c in profile.columns.values]

    rename = {
        f'{target_col}_mean':   'avg_price_m2',
        f'{target_col}_median': 'median_price_m2',
        f'{target_col}_std':    'std_price_m2',
        f'{target_col}_count':  'n_properties',
        'odl_do_centrum_mean':  'avg_dist_km',
        'odl_do_centrum_min':   'min_dist_km',
        'odl_do_centrum_max':   'max_dist_km',
        'wsp_x_mean':           'center_x',
        'wsp_y_mean':           'center_y',
    }
    profile = profile.rename(columns={k: v for k, v in rename.items() if k in profile.columns})

    # Średnie wartości cech użytych do klastrowania
    if feature_cols:
        feat_means = df.groupby(cluster_col, observed=True)[
            [c for c in feature_cols if c in df.columns]
        ].mean().round(2)
        feat_means.columns = [f'avg_{c}' for c in feat_means.columns]
        profile = profile.join(feat_means)

    if extra_cols:
        extra = df.groupby(cluster_col, observed=True)[
            [c for c in extra_cols if c in df.columns]
        ].mean().round(2)
        profile = profile.join(extra)

    return profile.sort_values('median_price_m2', ascending=False)


def elbow_analysis(
    X: pd.DataFrame,
    k_range: range = range(2, 12),
) -> pd.DataFrame:
    """Elbow method + silhouette scores for KMeans k selection.

    Useful for justifying the choice of n_clusters before running
    fit_property_clustering or fit_temporal_clustering.

    Args:
        X: Feature DataFrame (will be scaled internally)
        k_range: Range of k values to evaluate

    Returns:
        DataFrame with inertia and silhouette score per k
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.dropna())

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})

    return pd.DataFrame(results).set_index('k')