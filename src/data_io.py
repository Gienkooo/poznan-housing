from __future__ import annotations

import pandas as pd
from pathlib import Path

COLUMN_MAP = {
    'pow_uzytkowa':          'powierzchnia_uzytkowa_lokalu',
    'pow_przynalezna':       'powierzchnia_pomieszczen_przynaleznych',
    'pow_calkowita':         'calkowita_powierzchnia',
    'pow_final':             'powierzchnia_final',
    'n_cena_brutto':         'cena_nieruchomosci_brutto',
    't_cena_brutto':         'cena_transakcji_brutto',
    't_rynek':               'rodzaj_rynku',
    'liczba_izb':            'liczba_izb',  # same, no change needed
}


def _resolve_data_path(path: str | Path) -> str:
    """Resolve a data file path from either the repo root or the data/ folder."""
    input_path = Path(path)
    candidates = [input_path]
    project_root = Path(__file__).resolve().parents[1]

    if not input_path.is_absolute():
        candidates.extend([
            project_root / input_path,
            project_root / 'data' / input_path,
            Path.cwd() / input_path,
            Path.cwd() / 'data' / input_path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(input_path)


def load_data(path='rcn_lokale_polaczone_extra.csv'):
    resolved_path = _resolve_data_path(path)
    df = pd.read_csv(resolved_path, low_memory=False)
    for col in ['bud_is_residential', 'is_market_transaction']:
        if col in df.columns:
            df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
    return df.rename(columns=COLUMN_MAP)