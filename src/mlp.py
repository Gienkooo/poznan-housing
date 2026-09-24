from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ── Architecture ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        # Zero-init last Linear so block starts as identity
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PriceMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 256,
                 n_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.embed(x)))


# ── Data loaders ─────────────────────────────────────────────────────────────

def _make_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    def _ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
    return (
        DataLoader(_ds(X_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(_ds(X_val, y_val), batch_size=batch_size * 2, shuffle=False),
    )


def _val_loss(model: nn.Module, loader: DataLoader,
              loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(loss_fn(model(xb.to(device)), yb.to(device)).item())
    return float(np.mean(losses))


def _val_mae_pln(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).squeeze(1).cpu().numpy())
            targets.append(yb.squeeze(1).cpu().numpy())
    return float(np.mean(np.abs(
        np.expm1(np.concatenate(targets)) - np.expm1(np.concatenate(preds))
    )))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_features: int,
    hidden: int = 256,
    n_blocks: int = 3,
    dropout: float = 0.2,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 256,
    device: torch.device,
    plot: bool = False,
    fold_label: str = "",
) -> tuple[PriceMLP, dict]:
    model = PriceMLP(n_features, hidden=hidden,
                     n_blocks=n_blocks, dropout=dropout).to(device)
    with torch.no_grad():
        bias_init = float(np.mean(y_tr)) if np.isfinite(np.mean(y_tr)) else 0.0
        model.head[-1].bias.fill_(bias_init)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=8, min_lr=lr * 0.01
    )
    loss_fn = nn.HuberLoss(delta=0.5)

    train_loader, val_loader = _make_loaders(X_tr, y_tr, X_val, y_val, batch_size)

    history: dict = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val = float("inf")
    best_state: dict | None = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        t_loss = float(np.mean(batch_losses))
        v_loss = _val_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        scheduler.step(v_loss)

        if epoch % 10 == 0:
            history["val_mae"].append((epoch, _val_mae_pln(model, val_loader, device)))

        if v_loss < best_val - 1e-5:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if plot and epoch % 20 == 0:
            _live_plot(history, epoch, fold_label)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _live_plot(history: dict, epoch: int, label: str = "") -> None:
    clear_output(wait=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Huber loss (log)")
    axes[0].set_title(f"Loss  {label}  [epoch {epoch}]")
    axes[0].legend()
    if history["val_mae"]:
        ep, mae = zip(*history["val_mae"])
        axes[1].plot(ep, mae, marker="o", color="steelblue")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("MAE (PLN/m²)")
        axes[1].set_title("Validation MAE (PLN)")
    plt.tight_layout()
    plt.show()


# ── Expanding-window CV ───────────────────────────────────────────────────────

def evaluate_mlp_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    *,
    weight_decay : float = 1e-4,
    min_train_year: int = 2013,
    hidden: int = 256,
    n_blocks: int = 3,
    dropout: float = 0.2,
    lr: float = 5e-4,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 256,
    plot_folds: bool = False,
) -> pd.DataFrame:
    """Expanding-window CV keyed on calendar year — mirrors XGBoost evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_years = sorted(yr for yr in years.unique() if yr > min_train_year)
    records = []

    for test_year in test_years:
        tr_mask  = years < test_year
        val_mask = years == test_year
        if tr_mask.sum() < 500 or val_mask.sum() < 50:
            continue

        X_tr_raw = X[tr_mask].copy()
        X_test_raw = X[val_mask].copy()
        y_tr_raw = y[tr_mask].values
        y_test_raw = y[val_mask].values

        valid_mask = np.isfinite(y_tr_raw)
        X_tr_raw = X_tr_raw.iloc[valid_mask].copy()
        y_tr_raw = y_tr_raw[valid_mask]

        if len(X_tr_raw) == 0:
            continue

        train_cut = max(1, int(len(X_tr_raw) * 0.9))
        X_train_raw = X_tr_raw.iloc[:train_cut].copy()
        X_val_sub_raw = X_tr_raw.iloc[train_cut:].copy()
        y_train_raw = y_tr_raw[:train_cut]
        y_val_sub_raw = y_tr_raw[train_cut:]

        lo, hi = np.percentile(y_train_raw, [1, 99])
        clip_mask = (y_train_raw >= lo) & (y_train_raw <= hi)
        X_train_raw = X_train_raw.iloc[clip_mask].copy()
        y_train_raw = y_train_raw[clip_mask]

        if len(X_train_raw) == 0:
            continue

        for frame in (X_train_raw, X_val_sub_raw, X_test_raw):
            numeric_cols = frame.select_dtypes(include=[np.number]).columns
            if len(numeric_cols):
                frame[numeric_cols] = frame[numeric_cols].fillna(frame[numeric_cols].median())

        if 'osiedle_cat' in X.columns:
            encoder = TargetEncoder(cols=['osiedle_cat'], handle_missing="value")
            X_train_raw = encoder.fit_transform(X_train_raw, y_train_raw)
            X_val_sub_raw = encoder.transform(X_val_sub_raw)
            X_test_raw = encoder.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_val_sub_s = scaler.transform(X_val_sub_raw)
        X_test_s = scaler.transform(X_test_raw)

        model, _ = train_mlp(
            X_train_s, y_train_raw, X_val_sub_s, y_val_sub_raw,
            n_features=X_train_s.shape[1],
            hidden=hidden, n_blocks=n_blocks, dropout=dropout,
            weight_decay=weight_decay,
            lr=lr, max_epochs=max_epochs,
            patience=patience, batch_size=batch_size,
            device=device, plot=plot_folds,
            fold_label=f"test={test_year}  train={tr_mask.sum():,}",
        )

        model.eval()
        with torch.no_grad():
            pred_log = (
                model(torch.tensor(X_test_s, dtype=torch.float32).to(device))
                .squeeze(1).cpu().numpy()
            )

        y_real = np.expm1(y_test_raw)
        p_real = np.expm1(pred_log)

        rec = {
            "year":     test_year,
            "n_train":  int(tr_mask.sum()),
            "n_val":    int(val_mask.sum()),
            "r2":       r2_score(y_test_raw, pred_log),
            "mae_pln":  mean_absolute_error(y_real, p_real),
            "rmse_pln": np.sqrt(mean_squared_error(y_real, p_real)),
            "mape":     float(np.mean(np.abs((y_real - p_real) / (y_real + 1e-6))) * 100),
        }
        records.append(rec)
        print(f"  {test_year}: R²={rec['r2']:.3f}  MAE={rec['mae_pln']:,.0f} PLN/m²"
              f"  (train={rec['n_train']:,})")

    results = pd.DataFrame(records)
    print("── MLP expanding-window summary ─────────────────────────")
    print(f"  Mean R²  : {results['r2'].mean():.3f}  ±{results['r2'].std():.3f}")
    print(f"  Mean MAE : {results['mae_pln'].mean():,.0f} PLN/m²")
    return results


# ── Final model ───────────────────────────────────────────────────────────────

def train_final_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.1,
    **kwargs,
) -> tuple[PriceMLP, StandardScaler, Optional[TargetEncoder]]:
    """Train on full dataset; last val_frac for early stopping only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chronological sort to ensure .iloc logic is correct
    if "rok_miesiac_float" in X.columns:
        sort_idx = X["rok_miesiac_float"].argsort().values
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]

    split = int(len(X) * (1 - val_frac))
    X_tr_full_df, X_val_df = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_tr_full_df, y_val_df = y.iloc[:split].copy(), y.iloc[split:].copy()

    # Target clipping inside train split only
    lo, hi = y_tr_full_df.quantile([0.01, 0.99])
    clip_mask = (y_tr_full_df >= lo) & (y_tr_full_df <= hi)
    X_tr_df = X_tr_full_df[clip_mask].copy()
    y_tr_df = y_tr_full_df[clip_mask].copy()

    for frame in (X_tr_df, X_val_df):
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            frame[numeric_cols] = frame[numeric_cols].fillna(frame[numeric_cols].median())

    encoder = None
    if 'osiedle_cat' in X.columns:
        encoder = TargetEncoder(cols=['osiedle_cat'], handle_missing="value")
        X_tr_df = encoder.fit_transform(X_tr_df, y_tr_df)
        X_val_df = encoder.transform(X_val_df)

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr_df.values)
    X_val_s = scaler.transform(X_val_df.values)
    y_tr = y_tr_df.values
    y_val = y_val_df.values

    model, _ = train_mlp(
        X_tr_s, y_tr, X_val_s, y_val,
        n_features=X_tr_s.shape[1],
        device=device, plot=True, fold_label="final",
        **kwargs,
    )
    return model, scaler, encoder


def predict(model: PriceMLP, scaler: StandardScaler,
            X: pd.DataFrame, encoder: TargetEncoder = None) -> np.ndarray:
    """Return predictions in original PLN/m² scale."""
    device = next(model.parameters()).device
    model.eval()
    
    if encoder is not None:
        X = encoder.transform(X)

    with torch.no_grad():
        pred_log = (
            model(torch.tensor(scaler.transform(X.values),
                               dtype=torch.float32).to(device))
            .squeeze(1).cpu().numpy()
        )
    return np.expm1(pred_log)