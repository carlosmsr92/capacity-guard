"""Diagnóstico de causa: dado un episodio con alarma, ¿qué tipo de problema es? (ML multiclase)

Distingue los 4 incidentes y los picos legítimos de tráfico (que no hay que 'arreglar'). Importante para no
hacerse trampas: solo se diagnostica DESDE EL MOMENTO EN QUE SALTA LA ALARMA del detector no supervisado
(Isolation Forest, tarea A). Condicionar a "sé que hay un evento" regalaría la clase más difícil por eliminación.
Los eventos que nunca disparan alarma se reportan como cobertura, no se cuentan como aciertos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

from .features import ANOMALY_FEATURES

FEATS = ANOMALY_FEATURES + ["cpu_slope", "ram_eta", "cpu_d3", "io_d3", "req_d3",
                            "raw_cpu", "raw_ram", "raw_iowait", "raw_net", "raw_load1"]
CLASSES = ["noisy_neighbor", "memory_leak", "brute_force", "io_saturation", "legit_surge"]
EVAL_LAGS = {"al saltar la alarma": 0, "30 min después": 6}
TRAIN_LAGS = range(0, 9)


def event_segments(df: pd.DataFrame) -> list[dict]:
    """Segmentos de evento: índices globales (df.index) de cada muestra del evento, en orden."""
    segs = []
    for server, g in df.groupby("server", sort=False):
        ev, ix, i = g["event"].to_numpy(), g.index.to_numpy(), 0
        while i < len(ev):
            if ev[i]:
                j = i
                while j < len(ev) and ev[j] == ev[i]:
                    j += 1
                segs.append({"server": server, "kind": ev[i], "index": ix[i:j]})
                i = j
            else:
                i += 1
    return segs


def anchor_segments(segs, alerts: pd.Series) -> list[dict]:
    """Añade a cada evento la posición de su primera alarma (None si nunca saltó) para anclar el diagnóstico."""
    out = []
    for s in segs:
        hit = np.flatnonzero(alerts.loc[s["index"]].to_numpy())
        out.append(s | {"anchor": int(hit[0]) if hit.size else None})
    return out


def _rows(segs, feats, lags):
    X, y = [], []
    for s in segs:
        if s["anchor"] is None:
            continue
        for lag in lags:
            pos = s["anchor"] + lag
            if pos < len(s["index"]):
                X.append(feats.loc[s["index"][pos], FEATS].to_numpy(dtype=float))
                y.append(s["kind"])
    return np.array(X), np.array(y)


class Diagnoser:
    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                                    min_samples_leaf=15, l2_regularization=1.0, random_state=seed)

    def fit(self, segs, feats) -> "Diagnoser":
        X, y = _rows(segs, feats, TRAIN_LAGS)
        self.model.fit(X, y)
        return self

    def predict_proba(self, feat_row: pd.Series | pd.DataFrame) -> dict[str, float]:
        x = np.asarray(feat_row[FEATS], dtype=float).reshape(1, -1)
        return dict(zip(self.model.classes_, self.model.predict_proba(x)[0].round(4)))


def evaluate_diagnosis(art, shuffle_labels: bool = False, seed: int = 0) -> dict:
    """shuffle_labels=True es una comprobación de cordura: con etiquetas barajadas no debe acertar más que el azar."""
    df, feats, sp = art.df, art.feats, art.splits
    alerts = art.alerts_all_a  # alarmas de Isolation Forest (tarea A) en todo el periodo
    segs = anchor_segments(event_segments(df), alerts)
    onset = {id(s): df.loc[s["index"][0], "timestamp"] for s in segs}
    t_test = df.loc[sp["test"], "timestamp"].min()
    train = [s for s in segs if onset[id(s)] < t_test - pd.Timedelta(hours=14)]  # sin solape con test
    test = [s for s in segs if onset[id(s)] >= t_test]
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        kinds = rng.permutation([s["kind"] for s in train])
        train = [s | {"kind": k} for s, k in zip(train, kinds)]
    dx = Diagnoser().fit(train, feats)
    covered = [s for s in test if s["anchor"] is not None]
    res = {"n_train_events": len(train), "n_test_events": len(test), "coverage": len(covered) / max(len(test), 1),
           "undetected_kinds": pd.Series([s["kind"] for s in test if s["anchor"] is None]).value_counts().to_dict(),
           "by_lag": {}}
    maj = pd.Series([s["kind"] for s in train if s["anchor"] is not None]).value_counts(normalize=True)
    for name, lag in EVAL_LAGS.items():
        X, y = _rows(test, feats, [lag])
        if not len(y):
            continue
        pred = dx.model.predict(X)
        labels = [c for c in CLASSES if c in set(y) | set(pred)]
        cm = confusion_matrix(y, pred, labels=labels)
        res["by_lag"][name] = {
            "n": int(len(y)), "accuracy": float((pred == y).mean()),
            "majority_baseline": float((pd.Series(y) == maj.idxmax()).mean()),
            "per_class_recall": {l: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, l in enumerate(labels)},
            "labels": labels, "confusion": cm.tolist()}
    return {"summary": res, "model": dx}
