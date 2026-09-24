from pathlib import Path

DATA_PATH = Path("rcn_lokale_polaczone.csv")
BASE_YEAR = 2025
RANDOM_STATE = 42

CENTRUM_POZNAN = (52.405672, 16.931276)

CPI_YOY = {
    2011: 1.043,
    2012: 1.037,
    2013: 1.009,
    2014: 1.000,
    2015: 0.991,
    2016: 0.994,
    2017: 1.020,
    2018: 1.016,
    2019: 1.023,
    2020: 1.034,
    2021: 1.051,
    2022: 1.144,
    2023: 1.114,
    2024: 1.036,
    2025: 1.036,
}

PRICE_COLS = [
    "cena_lokalu_brutto",
    "cena_nieruchomosci_brutto",
    "cena_transakcji_brutto",
]

AREA_COL = "powierzchnia_uzytkowa_lokalu"
