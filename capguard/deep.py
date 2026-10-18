"""Deep learning: autoencoder denso sobre ventanas de 1 hora de métricas normalizadas.

Se entrena a reconstruir el comportamiento habitual (sin etiquetas); lo que reconstruye mal es anómalo.
Implementado con scikit-learn (MLPRegressor, perceptrón multicapa entrenado con Adam) para no depender de
GPU ni de PyTorch; la interfaz (fit/score) permite sustituirlo por una LSTM/TCN sin tocar el resto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .evaluation import alerts_from_score, evaluate, tune_threshold
from .features import BASE_COLS

ZCOLS = [f"z_{c}" for c in BASE_COLS]
WINDOW = 12  # 12 muestras = 1 hora


def make_windows(feats: pd.DataFrame, df: pd.DataFrame, w: int = WINDOW) -> np.ndarray:
    """Matriz (n, w*6) con la última hora de cada servidor; al inicio se repite la primera muestra."""
    out = np.zeros((len(df), w * len(ZCOLS)), dtype=np.float32)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby("server", sort=False):
        z = feats.loc[g.index, ZCOLS].clip(-10, 10).to_numpy(dtype=np.float32)
        padded = np.vstack([np.repeat(z[:1], w - 1, axis=0), z])
        win = np.stack([padded[i : i + len(z)] for i in range(w)], axis=1).reshape(len(z), -1)
        out[[pos[i] for i in g.index]] = win
    return out


class Autoencoder:
    name = "Autoencoder (DL)"

    def __init__(self, seed: int = 0):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", alpha=1e-4,
                                  max_iter=300, early_stopping=True, n_iter_no_change=10,
                                  validation_fraction=0.1, random_state=seed)

    def fit(self, X: np.ndarray, max_rows: int = 40000, seed: int = 0) -> "Autoencoder":
        rng = np.random.default_rng(seed)
        if len(X) > max_rows:
            X = X[rng.choice(len(X), max_rows, replace=False)]
        self.model.fit(X, X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.model.predict(X)) ** 2).mean(axis=1)


def evaluate_autoencoder(art, budget: float = 0.5, k: int = 2) -> dict:
    """Entrena con el periodo de train y evalúa igual que el resto de métodos (mismas tareas y presupuesto)."""
    from .pipeline import TASKS  # evita import circular

    df, feats, sp = art.df, art.feats, art.splits
    X = make_windows(feats, df)
    ae = Autoencoder().fit(X[sp["train"].to_numpy()])
    score = pd.Series(ae.score(X), index=df.index)
    out = {}
    val, test = df[sp["val"]], df[sp["test"]]
    for task, spec in TASKS.items():
        thr = tune_threshold(val, score[sp["val"]], art.windows[(task, "val")], spec["pre"], budget, k)
        alert = alerts_from_score(test, score[sp["test"]], thr, k)
        out[task] = evaluate(test, alert, art.windows[(task, "test")], spec["pre"]) | {
            "method": ae.name, "threshold": thr}
    return {"results": out, "model": ae, "n_iter": int(ae.model.n_iter_),
            "train_loss": float(ae.model.loss_), "scores": score}
