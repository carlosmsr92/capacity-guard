"""Validación rigurosa del modelo: walk-forward, calibración e interpretabilidad (machine learning)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from .features import FORECAST_FEATURES
from .pipeline import data_days, run
from .simulator import SimConfig

# (fin train, fin val, fin test) como fracción de la duración de los datos; con 45 días son los días 19/24/31,
# 26/31/38 y 33/38/45
FOLDS = ((19 / 45, 24 / 45, 31 / 45), (26 / 45, 31 / 45, 38 / 45), (33 / 45, 38 / 45, 1.0))


def fold_bounds(df: pd.DataFrame, folds=FOLDS) -> list[tuple[float, float, float]]:
    """Convierte las fracciones de FOLDS en días según la duración real de los datos."""
    days = data_days(df)
    return [tuple(round(f * days, 6) for f in fold) for fold in folds]


def walk_forward(df: pd.DataFrame, cfg: SimConfig, folds=FOLDS, budget: float = 0.5) -> pd.DataFrame:
    """Reentrena en ventana creciente y evalúa en periodos futuros distintos: ¿el resultado es estable?"""
    rows = []
    for i, b in enumerate(fold_bounds(df, folds), 1):
        art = run(cfg, budget=budget, df=df, bounds=b)
        for name, m in art.results["tasks"]["B"]["methods"].items():
            rows.append({"fold": i, "test_days": f"{b[1]:.0f}–{b[2]:.0f}", "method": m["method"], "key": name,
                         "events": m["n_events"], "recall": m["recall"], "early_rate": m["early_rate"],
                         "false_alarms_per_server_day": m["false_alarms_per_server_day"],
                         "auc": art.results["gbm_sample_metrics"]["roc_auc"] if name == "gbm" else np.nan})
    return pd.DataFrame(rows)


def _xy(art, split: str):
    m = art.splits[split] & ~art.df["sat_now"]
    return art.feats.loc[m, FORECAST_FEATURES].to_numpy(), art.df.loc[m, "sat_future"].to_numpy()


def calibration(art, bins=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001)) -> dict:
    """Calibra con isotónica sobre validación y mide en test si 'P=0,3' significa de verdad ~30 %."""
    gbm = art.models["gbm"].model
    Xv, yv = _xy(art, "val")
    Xt, yt = _xy(art, "test")
    iso = CalibratedClassifierCV(FrozenEstimator(gbm), method="isotonic").fit(Xv, yv)
    p_raw, p_cal = gbm.predict_proba(Xt)[:, 1], iso.predict_proba(Xt)[:, 1]

    def table(p):
        idx = np.digitize(p, bins) - 1
        rows = [{"bin": f"{bins[k]:.2f}–{min(bins[k + 1], 1):.2f}", "n": int((idx == k).sum()),
                 "pred_mean": float(p[idx == k].mean()), "observed": float(yt[idx == k].mean())}
                for k in range(len(bins) - 1) if (idx == k).any()]
        n = len(p)
        return rows, float(sum(r["n"] / n * abs(r["pred_mean"] - r["observed"]) for r in rows))

    t_raw, ece_raw = table(p_raw)
    t_cal, ece_cal = table(p_cal)
    return {"brier_raw": float(brier_score_loss(yt, p_raw)), "brier_calibrated": float(brier_score_loss(yt, p_cal)),
            "ece_raw": ece_raw, "ece_calibrated": ece_cal,
            "auc_raw": float(roc_auc_score(yt, p_raw)), "auc_calibrated": float(roc_auc_score(yt, p_cal)),
            "base_rate": float(yt.mean()), "reliability_raw": t_raw, "reliability_calibrated": t_cal,
            "calibrator": iso}


def importance(art, n_repeats: int = 5, top: int = 12) -> list[dict]:
    """Importancia por permutación en test (cuánto empeora la AP al barajar cada variable)."""
    Xt, yt = _xy(art, "test")
    r = permutation_importance(art.models["gbm"].model, Xt, yt, scoring="average_precision",
                               n_repeats=n_repeats, random_state=0)
    order = np.argsort(r.importances_mean)[::-1][:top]
    return [{"feature": FORECAST_FEATURES[i], "importance": float(r.importances_mean[i]),
             "std": float(r.importances_std[i])} for i in order]
