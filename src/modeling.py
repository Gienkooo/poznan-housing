from __future__ import annotations

import time
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.tree import DecisionTreeRegressor
import torch
import xgboost as xgb



def _imputer_pipeline(X: pd.DataFrame) -> Pipeline:
    """Median imputer only — used for tree models that don't need scaling."""
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def _scaled_pipeline(X: pd.DataFrame) -> Pipeline:
    """Median imputer + StandardScaler — required for Ridge and MLP."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def _make_tscv(n_splits: int = 5, gap: int = 200, test_size: int = 2000) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)


def _real_scale_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Converts log-scale predictions back to PLN before computing errors."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return {
        "R2":      r2_score(y_true_log, y_pred_log),   # R² stays in log space (standard)
        "MAE_PLN": mean_absolute_error(y_true, y_pred),
        "RMSE_PLN": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE":    float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100),
    }


def time_split(
    df: pd.DataFrame,
    date_col: str = "data_dokumentu",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(date_col)
    split_idx = int(len(df_sorted) * (1 - test_size))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


def evaluate_yearly_model_performance(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    *,
    model_params: Optional[dict] = None,
    min_train_year: int = 2013,
    target_col: str = "cena_za_m2",
) -> pd.DataFrame:
    """Expanding-window out-of-time yearly evaluation using tuned XGBoost.

    Trains on all historical data prior to each evaluation year, tests on that year,
    and reports log-space R² as well as real-space MAE and MAPE (PLN/m²) via expm1.
    """
    if model_params is None:
        model_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 5.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    X = X.copy()
    if "osiedle_cat" in X.columns:
        X["osiedle_cat"] = X["osiedle_cat"].astype(str)

    yearly_results = []
    for test_year in sorted(set(int(v) for v in years.unique() if int(v) >= min_train_year)):
        train_mask = years < test_year
        test_mask = years == test_year

        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        X_tr, y_tr = X[train_mask].copy(), y[train_mask]
        X_te, y_te = X[test_mask].copy(), y[test_mask]

        pipe = Pipeline([
            ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(**model_params)),
        ])

        pipe.fit(X_tr, y_tr)
        preds_log = pipe.predict(X_te)

        y_te_pln = np.expm1(y_te)
        preds_pln = np.expm1(preds_log)

        yearly_results.append({
            "year": test_year,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "r2": r2_score(y_te, preds_log),
            "mae_pln": mean_absolute_error(y_te_pln, preds_pln),
            "mape": float(np.mean(np.abs((y_te_pln - preds_pln) / np.clip(y_te_pln, 1, None))) * 100),
            "median_actual_price": float(y_te_pln.median()),
        })

    return pd.DataFrame(yearly_results)


def evaluate_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Baseline model comparison using TimeSeriesSplit.

    Reports R² (log scale, standard convention), MAE, RMSE and MAPE in
    real PLN/m² so results are directly interpretable.

    Args:
        X: Feature matrix (sorted chronologically before calling)
        y: log1p(cena_za_m2) target
        cv_folds: Number of TimeSeriesSplit folds

    Returns:
        DataFrame with one row per model and interpretable metrics
    """

    if 'rok_miesiac_float' in X.columns:
        X = X.sort_values('rok_miesiac_float').copy()
        y = y.loc[X.index].copy()
    elif 'data_dokumentu' in X.columns:
        X = X.sort_values('data_dokumentu').copy()
        y = y.loc[X.index].copy()

    X['osiedle_cat'] = X['osiedle_cat'].astype(str)

    models = {
        "ElasticNet": Pipeline([
            ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=18,
                min_samples_split=2, min_samples_leaf=1,
                max_features="sqrt", n_jobs=-1, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, reg_lambda=5.0,
                tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
            )),
        ]),
    }

    tscv = _make_tscv(n_splits=cv_folds)
    results = []

    for name, pipe in models.items():
        fold_metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_tr_full, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr_full, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            lo, hi = y_tr_full.quantile([0.01, 0.99])
            mask = y_tr_full.between(lo, hi)
            X_tr = X_tr_full[mask]
            y_tr = y_tr_full[mask]

            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            fold_metrics.append(_real_scale_metrics(y_te.values, preds))

        fm = pd.DataFrame(fold_metrics)
        results.append({
            "Model":          name,
            "Mean_CV_R2":     round(fm["R2"].mean(), 4),
            "Std_CV_R2":      round(fm["R2"].std(), 4),
            "Mean_MAE_PLN":   round(fm["MAE_PLN"].mean(), 0),
            "Mean_RMSE_PLN":  round(fm["RMSE_PLN"].mean(), 0),
            "Mean_MAPE":      round(fm["MAPE"].mean(), 2),
            "Fold_R2_Scores": fm["R2"].tolist(),
        })

    return pd.DataFrame(results).sort_values("Mean_CV_R2", ascending=False)


def tune_and_compare(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    n_splits: int = 5,
    test_size: int = 2000,
    gap: int = 200,
) -> tuple[Pipeline, dict, pd.DataFrame]:

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    X = X.copy()
    X['osiedle_cat'] = X['osiedle_cat'].astype(str)

    xgb_pipe = Pipeline([
        ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])

    xgb_param_dist = {
        "model__n_estimators":     [300, 500, 700],
        "model__max_depth":        [4, 6, 8],
        "model__learning_rate":    [0.03, 0.05, 0.1],
        "model__subsample":        [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [3, 5, 10],
        "model__reg_lambda":       [1.0, 5.0, 10.0],
    }

    print(f"Tuning XGBoost (TimeSeriesSplit, n_iter={n_iter})...")
    t0 = time.time()
    xgb_search = RandomizedSearchCV(
        xgb_pipe, param_distributions=xgb_param_dist,
        n_iter=n_iter, cv=tscv, scoring="r2",
        random_state=42, n_jobs=-1, verbose=1, refit=True,
    )
    xgb_search.fit(X, y)
    print(f"  Done in {time.time()-t0:.0f}s — Best R² (log): {xgb_search.best_score_:.4f}")
    print(f"  Best params: {xgb_search.best_params_}")

    mlp_pipe = Pipeline([
        ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", alpha=1e-4, max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=42,
        )),
    ])

    tree_pipe = Pipeline([
        ("target_encoder", TargetEncoder(cols=["osiedle_cat"], handle_missing="value")),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(max_depth=3, random_state=42)),
    ])

    mlp_fold_metrics  = []
    tree_fold_metrics = []
    xgb_fold_metrics  = []

    print("Evaluating MLP + DecisionTree (TimeSeriesSplit)...")
    t0 = time.time()
    best_xgb_pipe = xgb_search.best_estimator_

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        mlp_pipe.fit(X_tr, y_tr)
        mlp_fold_metrics.append(_real_scale_metrics(y_te.values, mlp_pipe.predict(X_te)))

        tree_pipe.fit(X_tr, y_tr)
        tree_fold_metrics.append(_real_scale_metrics(y_te.values, tree_pipe.predict(X_te)))

        best_xgb_pipe.fit(X_tr, y_tr)
        xgb_fold_metrics.append(_real_scale_metrics(y_te.values, best_xgb_pipe.predict(X_te)))

    print(f"  Done in {time.time()-t0:.0f}s")

    def _summarize(fold_metrics: list[dict], model_name: str) -> dict:
        fm = pd.DataFrame(fold_metrics)
        return {
            "Model":          model_name,
            "Mean_CV_R2":     round(fm["R2"].mean(),       4),
            "Std_CV_R2":      round(fm["R2"].std(),        4),
            "Mean_MAE_PLN":   round(fm["MAE_PLN"].mean(),  0),
            "Mean_RMSE_PLN":  round(fm["RMSE_PLN"].mean(), 0),
            "Mean_MAPE":      round(fm["MAPE"].mean(),     2),
            "Fold_R2_Scores": fm["R2"].tolist(),
        }

    comparison_df = pd.DataFrame([
        _summarize(xgb_fold_metrics,  "XGBoost (Tuned)"),
        _summarize(mlp_fold_metrics,  "MLP (scikit-learn)"),
        _summarize(tree_fold_metrics, "DecisionTree (max_depth=3)"),
    ])

    best_xgb_final = xgb_search.best_estimator_
    best_xgb_final.fit(X, y)

    print("\n" + "=" * 60)
    print("Final comparison (interpretable metrics):")
    cols_to_print = [c for c in comparison_df.columns if c != "Fold_R2_Scores"]
    print(comparison_df[cols_to_print].to_string(index=False))
    print("=" * 60)

    return best_xgb_final, xgb_search.best_params_, comparison_df


def generate_shap_explanations(
    model,
    X: pd.DataFrame,
    sample_n: int = 3000,
    save_path: Optional[str] = None
) -> tuple:
    """SHAP TreeExplainer for any fitted tree pipeline or model.

    Handles both raw model and Pipeline input — extracts the XGBoost step
    automatically if a Pipeline is passed.

    Args:
        model: Fitted XGBoost model or Pipeline containing one
        X: Feature DataFrame (pre-imputed, same columns as training)
        sample_n: Number of rows to sample for SHAP (speed vs precision)

    Returns:
        (explainer, shap_values, importance_df)
    """
    try:
        import shap
        import matplotlib.pyplot as plt

        xgb_model = model.named_steps["model"] if hasattr(model, "named_steps") else model

        if hasattr(model, "named_steps"):
            X_transformed = X.copy()
            for step_name in ["target_encoder", "imputer", "scaler"]:
                if step_name in model.named_steps:
                    step = model.named_steps[step_name]
                    X_transformed = pd.DataFrame(
                        step.transform(X_transformed),
                        columns=X_transformed.columns,
                        index=X_transformed.index
                    )
            X_imp = X_transformed
        else:
            X_imp = X.copy()

        X_sample = shap.sample(X_imp, min(sample_n, len(X_imp)), random_state=42)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        importance_df = (
            pd.DataFrame({
                "Feature":   X.columns,
                "Mean_SHAP": np.abs(shap_values).mean(axis=0),
            })
            .sort_values("Mean_SHAP", ascending=False)
            .reset_index(drop=True)
        )

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        return explainer, shap_values, importance_df

    except ImportError:
        print("⚠ SHAP not installed. Run: pip install shap")
        return None, None, None

def generate_mlp_shap(model, scaler, encoder, X_df, sample_n=500, bg_n=100, save_path: Optional[str] = None):
    """
    Liczy wartości SHAP dla modelu PyTorch używając KernelExplainer.
    """
    device = next(model.parameters()).device
    model.eval()

    import shap
    X_enc = encoder.transform(X_df).copy() if encoder else X_df.copy()
    numeric_cols = X_enc.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        X_enc[numeric_cols] = X_enc[numeric_cols].fillna(X_enc[numeric_cols].median())
    X_scaled = scaler.transform(X_enc.values if hasattr(X_enc, "values") else X_enc)

    np.random.seed(42)
    bg_idx = np.random.choice(X_scaled.shape[0], bg_n, replace=False)
    background = X_scaled[bg_idx]

    test_idx = np.random.choice(X_scaled.shape[0], min(sample_n, X_scaled.shape[0]), replace=False)
    X_test_scaled = X_scaled[test_idx]
    
    X_test_original = X_df.iloc[test_idx]

    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            return model(x_tensor).cpu().numpy().flatten()

    print(f"Obliczanie wartości SHAP dla MLP (tło: {bg_n}, próbka: {sample_n})...")
    print("Uwaga: Dla sieci neuronowej może to potrwać od kilkunastu do kilkudziesięciu sekund.")

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test_scaled)

    importance_df = pd.DataFrame({
        "Feature": X_df.columns,
        "Mean_SHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_original, show=False)
    plt.title("SHAP Summary Plot - MLP (PyTorch)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return explainer, shap_values, importance_df