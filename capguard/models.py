"""Cuatro detectores comparables: umbral estático, z-score estacional, Isolation Forest y pronosticador GBM."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .features import ANOMALY_FEATURES, FORECAST_FEATURES


class ThresholdDetector:
    """Lo que hace una monitorización clásica: alarma fija por recurso. Sin aprendizaje."""

    name = "Umbral estático"
    limits = {"cpu": 85.0, "ram": 90.0, "iowait": 25.0}
    fixed_thr = 1.0

    def score(self, df: pd.DataFrame, feats: pd.DataFrame | None = None) -> np.ndarray:
        return np.maximum.reduce([df[c].to_numpy() / lim for c, lim in self.limits.items()])


class SeasonalZ:
    """Desviación máxima (en MAD) respecto a lo normal para esa hora de la semana."""

    name = "Z-score estacional"
    cols = ["z_cpu", "z_ram", "z_iowait", "z_net", "z_load1", "z_req"]

    def score(self, df: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
        return feats[self.cols].abs().max(axis=1).to_numpy()


class IsolationDetector:
    """Detección no supervisada multivariante sobre variables normalizadas por la estacionalidad."""

    name = "Isolation Forest"

    def __init__(self, seed: int = 0):
        self.model = IsolationForest(n_estimators=300, max_samples=4096, random_state=seed, n_jobs=-1)

    def fit(self, feats: pd.DataFrame) -> "IsolationDetector":
        self.model.fit(feats[ANOMALY_FEATURES].to_numpy())
        return self

    def score(self, df: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(feats[ANOMALY_FEATURES].to_numpy())


class SaturationForecaster:
    """Supervisado: probabilidad de que el servidor sature en los próximos 30 min."""

    name = "Pronosticador GBM"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed,
        )

    def fit(self, feats: pd.DataFrame, y: pd.Series, already_saturated: pd.Series) -> "SaturationForecaster":
        m = ~already_saturated.to_numpy()  # solo aprende de momentos aún no saturados
        self.model.fit(feats.loc[m, FORECAST_FEATURES].to_numpy(), y.to_numpy()[m])
        return self

    def score(self, df: pd.DataFrame, feats: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feats[FORECAST_FEATURES].to_numpy())[:, 1]
