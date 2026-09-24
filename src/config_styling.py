from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def set_professional_style() -> None:
    """Central style configuration for the project visual output."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def set_plot_style() -> None:
    """Backward-compatible alias used by the notebook and visualization modules."""
    set_professional_style()


FEATURE_DICT = {
    "powierzchnia_final": "Usable apartment area in square meters.",
    "liczba_izb": "Number of rooms in the apartment.",
    "rok": "Transaction year.",
    "rok_miesiac_float": "Continuous time variable used for trend modeling.",
    "odl_do_centrum": "Distance from the apartment to the Poznań city center (km).",
    "wibor_3m": "Three-month WIBOR rate used as a macro-financial signal.",
    "srednie_wynagrodzenie": "Quarterly average gross salary proxy for local purchasing power.",
    "is_pierwotny": "Indicator for primary-market transactions.",
    "sprzedajacy_firma": "Seller is a business entity.",
    "sprzedajacy_jst": "Seller is a local government unit.",
    "sprzedajacy_osoba": "Seller is a private individual.",
    "ma_pomieszczenia_przynalezne": "Indicator for additional attached spaces.",
    "osiedle_cat": "Encoded neighborhood identifier used in model input.",
    "cena_za_m2": "Target variable: price per square meter (PLN/m²).",
}


def print_data_dictionary(features_in_df):
    """Print a concise feature dictionary for the features currently in use."""
    print("--- Feature Data Dictionary ---")
    for col in features_in_df:
        desc = FEATURE_DICT.get(col, "Derived or engineered feature.")
        print(f"• {col}: {desc}")