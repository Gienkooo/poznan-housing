from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import AREA_COL, BASE_YEAR, CENTRUM_POZNAN, CPI_YOY, PRICE_COLS

DEDUP_MAP = {
    'powierzchnia_final': [
        # 'calkowita_powierzchnia',
        'powierzchnia_uzytkowa_lokalu',
        'powierzchnia_pomieszczen_przynaleznych',
    ],
    'cena_za_m2': [
        'cena_za_m2_nominal', 'cena_za_m2_raw', 'cena_za_m2_normalized', 
        'cena_final', 'cena_final_base', 'cena_final_udzial_norm',
        'cena_lokalu_brutto', 'cena_nieruchomosci_brutto',
        'cena_transakcji_brutto',
    ],
    # 'rok_miesiac_float': ['rok'], 
}

# Columns that are IDs / admin / never model features
DROP_NON_FEATURES = [
    # admin / ids
    'okres', 'lokal_id', 'nieruchomosc_id', 'transakcja_id',
    'id_lokalu_ewidencyjny', 'notariusz',
    'log_target', 'inflation_factor_to_base',
    # building/land registry — only populated for grunt/grunt_z_budynkiem
    'bud_rodzaj_kod', 'bud_footprint_m2', 'bud_centroid_x', 'bud_centroid_y',
    'bud_liczba_budynkow', 'bud_is_residential',
    'dzialka_pow_m2', 'dzialka_cena_ewidencyjna', 'dzialka_footprint_m2',
    'n_pow_gruntu_m2',
    't_sprzedajacy',    # ← zakodowany już jako sprzedajacy_firma/jst/osoba
    't_kupujacy',       # ← nigdy nie był feature, teraz użyty tylko do filtra
    't_rodzaj_transakcji',  # ← po filtrze pre-raw zawsze 'sprzedaz'
    'rodzaj_rynku',     # ← zakodowany jako is_pierwotny
    'n_prawo_nazwa',    # ← zakodowany jako is_uzytkowanie_wieczyste
    'bud_rodzaj_nazwa', # ← zakodowany jako is_wielorodzinny
    'funkcja_lokalu',   # ← po filtrze pre-raw zawsze 1.0
    'n_rodzaj_nazwa',   # ← po filtrze pre-raw zawsze 'lokal'
    'n_udzial_float',   # ← po filtrze pre-raw zawsze 1.0
    # 'osiedle'
]

def drop_redundant_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step: deduplicate engineered columns.
    Drops all aliases in DEDUP_MAP (keeps the canonical).
    Drops non-feature admin columns.
    Logs what was removed.
    """
    to_drop = []

    for canonical, aliases in DEDUP_MAP.items():
        for col in aliases:
            if col in df.columns:
                to_drop.append(col)
                if verbose:
                    print(f"  drop '{col}'  →  kept '{canonical}'")

    for col in DROP_NON_FEATURES:
        if col in df.columns:
            to_drop.append(col)
            if verbose:
                print(f"  drop '{col}'  (non-feature)")

    dropped = list(dict.fromkeys(to_drop))  # preserve order, dedupe list itself
    df = df.drop(columns=dropped)
    if verbose:
        print(f"\n✓ Dropped {len(dropped)} redundant/admin columns. "
              f"Remaining: {df.shape[1]}")
    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # rodzaj_rynku, nie t_rynek
    if 'rodzaj_rynku' in out.columns:
        out['is_pierwotny'] = (out['rodzaj_rynku'] == 'pierwotny').astype(int)
    else:
        out['is_pierwotny'] = 0

    if 'n_prawo_nazwa' in out.columns:
        out['is_uzytkowanie_wieczyste'] = (
            out['n_prawo_nazwa'] == 'uzytkowanie_wieczyste'
        ).astype(int)
    else:
        out['is_uzytkowanie_wieczyste'] = 0

    if 't_sprzedajacy' in out.columns:
        out['sprzedajacy_firma'] = out['t_sprzedajacy'].isin([
            'spolka_prawa_handlowego', 'inny_podmiot_prawny'
        ]).astype(int)

        out['sprzedajacy_jst']    = (out['t_sprzedajacy'] == 'jednostka_samorzadu').astype(int)
        out['sprzedajacy_osoba']  = (out['t_sprzedajacy'] == 'osoba_fizyczna').astype(int)
    else:
        out['sprzedajacy_firma'] = 0
        out['sprzedajacy_jst'] = 0
        out['sprzedajacy_osoba'] = 0

    if 'bud_rodzaj_nazwa' in out.columns:
        out['is_wielorodzinny'] = (
            out['bud_rodzaj_nazwa'] == 'wielorodzinny'
        ).astype(int)
    else:
        out['is_wielorodzinny'] = 0

    return out

def encode_osiedle(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'osiedle' in out.columns:
        out['osiedle_cat'] = out['osiedle'].astype('category').cat.codes
        # -1 = NaN (poza granicami)
    return out

def add_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price_series = out[PRICE_COLS[0]].copy()
    for col in PRICE_COLS[1:]:
        price_series = price_series.fillna(out[col])
    out["cena_final"] = price_series
    out["powierzchnia_final"] = out[AREA_COL] if AREA_COL in out.columns else np.nan
    if "powierzchnia_pomieszczen_przynaleznych" in out.columns:
        out["powierzchnia_pomieszczen_przynaleznych"] = out[
            "powierzchnia_pomieszczen_przynaleznych"
        ].fillna(0)
    else:
        out["powierzchnia_pomieszczen_przynaleznych"] = 0.0
    out["ma_pomieszczenia_przynalezne"] = (
        out["powierzchnia_pomieszczen_przynaleznych"] > 0
    ).astype(int)
    out["calkowita_powierzchnia"] = (
        out["powierzchnia_final"] + out["powierzchnia_pomieszczen_przynaleznych"]
    )
    out["cena_za_m2_nominal"] = out["cena_final"] / out["powierzchnia_final"]
    return out


def add_time_features(df: pd.DataFrame, date_col: str = "data_dokumentu") -> pd.DataFrame:
    """Adds continuous and discrete time features."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["rok"] = out[date_col].dt.year
    out["rok_miesiac_float"] = out[date_col].dt.year + (
        out[date_col].dt.month - 1
    ) / 12.0
    return out

# Monthly WIBOR 3M: (year, month) → rate %
# Sources: Eurostat via YCharts (2022–2026), GPW Benchmark archives (2011–2021)
WIBOR_3M_MONTHLY = {
    # 2011
    (2011, 1): 3.95, (2011, 2): 4.00, (2011, 3): 4.13, (2011, 4): 4.31,
    (2011, 5): 4.65, (2011, 6): 4.84, (2011, 7): 4.93, (2011, 8): 4.93,
    (2011, 9): 4.93, (2011, 10): 4.93, (2011, 11): 4.93, (2011, 12): 4.99,
    # 2012
    (2012, 1): 4.99, (2012, 2): 4.99, (2012, 3): 4.99, (2012, 4): 4.99,
    (2012, 5): 4.93, (2012, 6): 4.93, (2012, 7): 4.93, (2012, 8): 4.93,
    (2012, 9): 4.93, (2012, 10): 4.87, (2012, 11): 4.60, (2012, 12): 4.24,
    # 2013 — aggressive cut cycle
    (2013, 1): 4.06, (2013, 2): 3.82, (2013, 3): 3.62, (2013, 4): 3.38,
    (2013, 5): 3.14, (2013, 6): 2.91, (2013, 7): 2.72, (2013, 8): 2.71,
    (2013, 9): 2.69, (2013, 10): 2.67, (2013, 11): 2.67, (2013, 12): 2.67,
    # 2014
    (2014, 1): 2.67, (2014, 2): 2.67, (2014, 3): 2.67, (2014, 4): 2.67,
    (2014, 5): 2.67, (2014, 6): 2.67, (2014, 7): 2.67, (2014, 8): 2.67,
    (2014, 9): 2.60, (2014, 10): 2.06, (2014, 11): 2.06, (2014, 12): 2.06,
    # 2015–2019: NBP rate frozen at 1.5%, WIBOR 3M flat ~1.72%
    (2015, 1): 2.06, (2015, 2): 1.65, (2015, 3): 1.65, (2015, 4): 1.65,
    (2015, 5): 1.65, (2015, 6): 1.65, (2015, 7): 1.72, (2015, 8): 1.72,
    (2015, 9): 1.72, (2015, 10): 1.72, (2015, 11): 1.72, (2015, 12): 1.72,
    **{(y, m): 1.72 for y in [2016, 2017, 2018, 2019] for m in range(1, 13)},
    # 2020 — COVID cuts (Mar–May)
    (2020, 1): 1.71, (2020, 2): 1.71, (2020, 3): 1.20, (2020, 4): 0.64,
    (2020, 5): 0.27, (2020, 6): 0.26, (2020, 7): 0.25, (2020, 8): 0.24,
    (2020, 9): 0.23, (2020, 10): 0.23, (2020, 11): 0.23, (2020, 12): 0.23,
    # 2021 — flat then hike cycle begins Oct
    (2021, 1): 0.21, (2021, 2): 0.21, (2021, 3): 0.21, (2021, 4): 0.21,
    (2021, 5): 0.21, (2021, 6): 0.21, (2021, 7): 0.21, (2021, 8): 0.21,
    (2021, 9): 0.21, (2021, 10): 0.63, (2021, 11): 1.21, (2021, 12): 2.35,
    # 2022 — rapid hike cycle
    (2022, 1): 2.79, (2022, 2): 3.33, (2022, 3): 4.27, (2022, 4): 5.48,
    (2022, 5): 6.42, (2022, 6): 6.85, (2022, 7): 7.03, (2022, 8): 7.04,
    (2022, 9): 7.16, (2022, 10): 7.34, (2022, 11): 7.43, (2022, 12): 7.11,
    # 2023 — plateau then cuts begin Sep
    (2023, 1): 6.95, (2023, 2): 6.93, (2023, 3): 6.92, (2023, 4): 6.90,
    (2023, 5): 6.90, (2023, 6): 6.90, (2023, 7): 6.81, (2023, 8): 6.69,
    (2023, 9): 5.99, (2023, 10): 5.68, (2023, 11): 5.77, (2023, 12): 5.85,
    # 2024 — flat
    (2024, 1): 5.87, (2024, 2): 5.86, (2024, 3): 5.86, (2024, 4): 5.86,
    (2024, 5): 5.85, (2024, 6): 5.85, (2024, 7): 5.86, (2024, 8): 5.85,
    (2024, 9): 5.85, (2024, 10): 5.85, (2024, 11): 5.85, (2024, 12): 5.85,
    # 2025
    (2025, 1): 5.85, (2025, 2): 5.87, (2025, 3): 5.85, (2025, 4): 5.59,
    (2025, 5): 5.26, (2025, 6): 5.22, (2025, 7): 5.02, (2025, 8): 4.88,
    (2025, 9): 4.75, (2025, 10): 4.56, (2025, 11): 4.28, (2025, 12): 4.06,
    # 2026
    (2026, 1): 3.93, (2026, 2): 3.86, (2026, 3): 3.86, (2026, 4): 3.86,
    (2026, 5): 3.86,
}

# Quarterly average salary (GUS national economy): (year, quarter) → PLN gross
SALARY_QUARTERLY = {
    (2011, 1): 3311, (2011, 2): 3373, (2011, 3): 3395, (2011, 4): 3621,
    (2012, 1): 3526, (2012, 2): 3577, (2012, 3): 3593, (2012, 4): 3822,
    (2013, 1): 3740, (2013, 2): 3770, (2013, 3): 3795, (2013, 4): 4025,
    (2014, 1): 3942, (2014, 2): 3979, (2014, 3): 3999, (2014, 4): 4218,
    (2015, 1): 4054, (2015, 2): 4099, (2015, 3): 4150, (2015, 4): 4357,
    (2016, 1): 4181, (2016, 2): 4215, (2016, 3): 4247, (2016, 4): 4532,
    (2017, 1): 4353, (2017, 2): 4395, (2017, 3): 4441, (2017, 4): 4762,
    (2018, 1): 4622, (2018, 2): 4703, (2018, 3): 4765, (2018, 4): 5104,
    (2019, 1): 4950, (2019, 2): 5025, (2019, 3): 5096, (2019, 4): 5439,
    (2020, 1): 5331, (2020, 2): 5024, (2020, 3): 5168, (2020, 4): 5457,
    (2021, 1): 5672, (2021, 2): 5748, (2021, 3): 5773, (2021, 4): 6177,
    (2022, 1): 6335, (2022, 2): 6523, (2022, 3): 6556, (2022, 4): 7102,
    (2023, 1): 7124, (2023, 2): 7404, (2023, 3): 7452, (2023, 4): 8038,
    (2024, 1): 8147, (2024, 2): 8387, (2024, 3): 8402, (2024, 4): 9191,
    (2025, 1): 8962, (2025, 2): 9310, (2025, 3): 9350, (2025, 4): 9800,
}

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds macroeconomic variables at monthly WIBOR and quarterly salary resolution.

    WIBOR 3M: monthly averages from Eurostat/GPW Benchmark (2011–2026).
    Salary: GUS quarterly average gross wage in national economy.
    Both joined on (rok, miesiac) to avoid intra-year flattening.
    """
    out = df.copy()

    date_col = "data_dokumentu"
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out["_month"] = out[date_col].dt.month
    elif "rok_miesiac_float" in out.columns:
        out["_month"] = (
            (out["rok_miesiac_float"] - out["rok_miesiac_float"].astype(int)) * 12
        ).round().astype(int).clip(1, 12)
    else:
        out["_month"] = 1  # fallback

    out["wibor_3m"] = out.apply(
        lambda r: WIBOR_3M_MONTHLY.get((int(r["rok"]), int(r["_month"])), float("nan")),
        axis=1,
    )

    out["_quarter"] = ((out["_month"] - 1) // 3 + 1).astype(int)
    out["srednie_wynagrodzenie"] = out.apply(
        lambda r: SALARY_QUARTERLY.get((int(r["rok"]), int(r["_quarter"])), float("nan")),
        axis=1,
    )

    return out.drop(columns=["_month", "_quarter"], errors="ignore")


def add_inflation_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = sorted(CPI_YOY.keys())
    price_index = {BASE_YEAR: 1.0}
    for y in range(BASE_YEAR - 1, min(years) - 1, -1):
        price_index[y] = price_index[y + 1] / CPI_YOY[y + 1]
    for y in range(BASE_YEAR + 1, max(years) + 1):
        price_index[y] = price_index[y - 1] * CPI_YOY[y]

    out["inflation_factor_to_base"] = out["rok"].map(
        lambda y: price_index[BASE_YEAR] / price_index[y] if y in price_index else np.nan
    )
    out["cena_final_base"] = out["cena_final"] * out["inflation_factor_to_base"]
    out["cena_za_m2"] = out["cena_final_base"] / out["powierzchnia_final"]
    return out


def add_distance_to_centrum(
    df: pd.DataFrame, northing_col: str = "wsp_x", easting_col: str = "wsp_y"
) -> pd.DataFrame:
    out = df.copy()
    out[northing_col] = pd.to_numeric(out[northing_col], errors="coerce")
    out[easting_col] = pd.to_numeric(out[easting_col], errors="coerce")

    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2177", always_xy=True)
        center_easting, center_northing = transformer.transform(
            CENTRUM_POZNAN[1], CENTRUM_POZNAN[0]
        )
        dx = out[easting_col] - center_easting
        dy = out[northing_col] - center_northing
        out["odl_do_centrum"] = np.sqrt(dx**2 + dy**2) / 1000.0
        return out
    except Exception:
        pass

    lat1 = np.radians(out[northing_col])
    lon1 = np.radians(out[easting_col])
    lat2 = np.radians(CENTRUM_POZNAN[0])
    lon2 = np.radians(CENTRUM_POZNAN[1])
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    earth_radius_km = 6371.0
    out["odl_do_centrum"] = earth_radius_km * c
    return out


def add_room_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"powierzchnia_final", "liczba_izb"}.issubset(out.columns):
        out["metry_na_izbe"] = out["powierzchnia_final"] / out["liczba_izb"]
    return out


def filter_outliers(
    df: pd.DataFrame, column: str, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.DataFrame:
    q_low, q_high = df[column].quantile([lower_quantile, upper_quantile])
    return df[(df[column] >= q_low) & (df[column] <= q_high)].copy()


def apply_original_market_filters(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply domain filtering rules for residential arm's-length market transactions.

    Standard data cleaning step before feature engineering: filters for residential
    premises, standard sale transactions, 100% ownership share, and excludes
    institutional buyers.
    """
    df = df_raw.copy()
    df = df[df['n_rodzaj_nazwa'] == 'lokal'].copy()
    df = df[df['funkcja_lokalu'] == 1.0].copy()
    df = df[df['t_rodzaj_transakcji'] == 'sprzedaz'].copy()
    df = df[df['t_kupujacy'] != 'jednostka_samorzadu'].copy()
    df = df[pd.to_numeric(df['n_udzial_float'], errors='coerce').fillna(1.0) >= 1.0].copy()
    return df


def prepare_market_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build the public market dataset using standard domain preprocessing and feature engineering."""
    df = apply_original_market_filters(df_raw)
    df = add_price_columns(df)
    df = add_time_features(df)
    df = add_inflation_adjustment(df)
    df = add_distance_to_centrum(df)
    df = add_room_features(df)
    df = add_macro_features(df)
    df = add_transaction_features(df)
    df = encode_osiedle(df)
    df = drop_redundant_columns(df, verbose=False)
    return df


def prepare_jst_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build the institutional JST (municipal) transactions subset."""
    df = df_raw.copy()
    df = df[df['n_rodzaj_nazwa'] == 'lokal'].copy()
    df = df[df['funkcja_lokalu'] == 1.0].copy()
    df = df[df['t_rodzaj_transakcji'] == 'sprzedaz'].copy()
    df = df[pd.to_numeric(df['n_udzial_float'], errors='coerce').fillna(1.0) >= 1.0].copy()
    df = df[df['t_sprzedajacy'] == 'jednostka_samorzadu'].copy()
    df = df[df['t_kupujacy'] != 'jednostka_samorzadu'].copy()
    df = add_price_columns(df)
    df = df[(df['calkowita_powierzchnia'] >= 15.0) & (df['calkowita_powierzchnia'] <= 150.0)].copy()
    df['nr_kondygnacji'] = pd.to_numeric(df['nr_kondygnacji'], errors='coerce')
    df = df[df['nr_kondygnacji'] >= 0].copy()
    df = add_time_features(df)
    df = add_inflation_adjustment(df)
    df = add_transaction_features(df)
    df = encode_osiedle(df)
    df = df[df['cena_za_m2'].between(100, 30000)].copy()
    df = drop_redundant_columns(df, verbose=False)
    return df


def select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def process_additional_space(df: pd.DataFrame) -> pd.DataFrame:
    """DEPRECATED: This function is redundant.

    Both fillna(0) and the binary flag are already created by
    ``add_price_columns()`` as ``ma_pomieszczenia_przynalezne``.
    The ``has_additional_space`` column produced here is an exact
    duplicate of ``ma_pomieszczenia_przynalezne`` and is not used
    in MODEL_FEATURES.  Remove calls to this function from the
    notebook pipeline.
    """
    import warnings
    warnings.warn(
        "process_additional_space() is deprecated — "
        "add_price_columns() already creates ma_pomieszczenia_przynalezne. "
        "Remove this call from the pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    df = df.copy()
    if "powierzchnia_pomieszczen_przynaleznych" in df.columns:
        df["powierzchnia_pomieszczen_przynaleznych"] = df[
            "powierzchnia_pomieszczen_przynaleznych"
        ].fillna(0.0)
        df["has_additional_space"] = (
            df["powierzchnia_pomieszczen_przynaleznych"] > 0
        ).astype(int)
    return df


def engineer_building_features(
    df: pd.DataFrame, 
    current_year_col: str = "rok", 
    build_year_col: str = "rok_budowy"
) -> pd.DataFrame:
    """Calculates building age at time of transaction."""
    df = df.copy()
    if current_year_col in df.columns and build_year_col in df.columns:
        df["building_age"] = df[current_year_col] - df[build_year_col]
        df["building_age"] = df["building_age"].apply(lambda x: max(0, x))
    return df


def create_price_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Clusters properties by price to identify geographic/market segments.
    Uses price per sqm clustering.
    """
    df = df.copy()
    if "cena_za_m2" in df.columns:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        price_per_sqm = df["cena_za_m2"].dropna()
        
        scaler = StandardScaler()
        price_scaled = scaler.fit_transform(price_per_sqm.values.reshape(-1, 1))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(price_scaled)
        
        df.loc[price_per_sqm.index, "price_cluster"] = labels
        
        # Reorder clusters: 0 = cheapest, n-1 = most expensive
        cluster_means = df.groupby("price_cluster", observed=True)["cena_za_m2"].mean()
        sorted_clusters = cluster_means.sort_values().index
        mapping = {old: new for new, old in enumerate(sorted_clusters)}
        df["price_cluster"] = df["price_cluster"].map(mapping)
    
    return df


def get_3_case_studies(df: pd.DataFrame, target_col: str = "cena_za_m2") -> pd.DataFrame:
    """Extracts 3 representative cases for report: cheapest, typical, most expensive."""
    df = df.copy()
    
    # Cheapest
    cheapest_idx = df[target_col].idxmin()
    cheapest = df.loc[[cheapest_idx]]
    cheapest.index = ["Cheapest"]
    
    # Most expensive
    expensive_idx = df[target_col].idxmax()
    most_expensive = df.loc[[expensive_idx]]
    most_expensive.index = ["Most Expensive"]
    
    # Typical (closest to median)
    median_val = df[target_col].median()
    typical_idx = (df[target_col] - median_val).abs().idxmin()
    typical = df.loc[[typical_idx]]
    typical.index = ["Typical/Median"]
    
    cases = pd.concat([cheapest, typical, most_expensive])
    return cases

def build_residential_mask(df: pd.DataFrame) -> pd.Series:
    """Combined residential + free-market filter.

    Applies three conceptual layers in a single boolean mask:

    Layer 1 — Price pre-filter (domain-driven, NOT percentile-based):
        cena_za_m2 ∈ [1 500, 30 000] PLN/m².  Removes clear data-entry
        errors, symbolic gifts (1 PLN), and ultra-luxury penthouses that
        lack quality features in the dataset.

    Layer 2 — Physical plausibility (residential profile):
        powierzchnia_final ∈ [15, 150] m²
        metry_na_izbe      ∈ [5, 60]   m²/room
        liczba_izb         ∈ [1, 8]    (NaN allowed — imputed later)
        nr_kondygnacji     ≥ 0         (removes basement entries)

    Layer 3 — Free-market ownership:
        sprzedajacy_jst == 0  (excludes municipal sales, analysed
        separately in the JST section).
    """
    return (
        # Layer 1: price pre-filter (domain-driven scope)
        (df['cena_za_m2'].between(1500, 30000)) &
        # Layer 2: physical plausibility
        (df['powierzchnia_final'].between(15, 150)) &
        (df['metry_na_izbe'].between(5, 60)) &
        (df['liczba_izb'].between(1, 8) | df['liczba_izb'].isna()) &
        (df['nr_kondygnacji'] >= 0) &
        # Layer 3: free-market ownership
        (df['sprzedajacy_jst'] == 0)
    )

def clip_target_outliers(
    df: pd.DataFrame,
    target_col: str = 'cena_za_m2',
    lo_q: float = 0.01,
    hi_q: float = 0.99,
    verbose: bool = True,
) -> pd.DataFrame:
    lo, hi = df[target_col].quantile([lo_q, hi_q])
    mask = df[target_col].between(lo, hi)
    if verbose:
        print(f"clip_target_outliers: removed {(~mask).sum():,} rows "
              f"({(~mask).mean()*100:.1f}%) outside [{lo:.0f}, {hi:.0f}] PLN/m²")
    return df[mask].copy()