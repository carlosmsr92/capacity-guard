"""Evaluación a nivel de EVENTO (lo que importa a operaciones), no solo por muestra.

Un 'evento objetivo' es un incidente real (tarea A) o un episodio de saturación (tarea B).
Para cada uno medimos si saltó alguna alarma y con cuánta antelación; además contamos las alarmas
que no corresponden a nada (falsas alarmas) por servidor y día.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import STEP_MIN


def episodes(mask: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Tramos contiguos de True como (inicio, fin) inclusivos; fusiona huecos <= merge_gap."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    eps, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            eps.append((int(s), int(prev)))
            s = i
        prev = i
    eps.append((int(s), int(prev)))
    return eps


def persistent(mask: np.ndarray, k: int) -> np.ndarray:
    """True solo si las últimas k muestras superan el umbral (evita alarmas por un pico aislado)."""
    if k <= 1:
        return mask.astype(bool)
    return np.convolve(mask.astype(int), np.ones(k, dtype=int), "full")[: len(mask)] >= k


def alerts_from_score(df: pd.DataFrame, score: np.ndarray | pd.Series, thr: float, k: int = 2) -> pd.Series:
    sc = pd.Series(np.asarray(score), index=df.index)
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby("server", sort=False):
        out.loc[g.index] = persistent((sc.loc[g.index] >= thr).to_numpy(), k)
    return out


def incident_windows(df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    """Tarea A: ventanas de incidentes reales (no incluye picos legítimos)."""
    res = {}
    for server, g in df.groupby("server", sort=False):
        ev, wins, i = g["event"].to_numpy(), [], 0
        while i < len(ev):
            if g["is_incident"].to_numpy()[i]:
                j = i
                while j < len(ev) and ev[j] == ev[i]:
                    j += 1
                wins.append((i, j - 1))
                i = j
            else:
                i += 1
        res[server] = wins
    return res


def saturation_windows(df: pd.DataFrame, merge_gap: int = 2) -> dict[str, list[tuple[int, int]]]:
    """Tarea B: episodios en los que algún recurso cruza su límite de saturación."""
    return {s: episodes(g["sat_now"].to_numpy(), merge_gap) for s, g in df.groupby("server", sort=False)}


def evaluate(
    df: pd.DataFrame,
    alert: pd.Series,
    windows: dict[str, list[tuple[int, int]]],
    pre: int,
    post: int = 6,
    early_min: int = 10,
) -> dict:
    """pre/post: muestras de margen antes/después de cada ventana en las que una alarma cuenta como válida."""
    leads, n_win, n_alert_eps, n_false, n_samples = [], 0, 0, 0, 0
    per_event = []  # (servidor, inicio, adelanto en min o None)
    for server, g in df.groupby("server", sort=False):
        a = alert.loc[g.index].to_numpy()
        n = len(a)
        n_samples += n
        wins = windows.get(server, [])
        padded = [(max(0, s - pre), min(n - 1, e + post)) for s, e in wins]
        for (s, _), (lo, hi) in zip(wins, padded):
            n_win += 1
            hit = np.flatnonzero(a[lo : hi + 1])
            lead = (s - (lo + hit[0])) * STEP_MIN if hit.size else None
            per_event.append((server, int(s), lead))
            if lead is not None:
                leads.append(lead)
        for s, e in episodes(a):
            n_alert_eps += 1
            if not any(s <= hi and e >= lo for lo, hi in padded):
                n_false += 1
    leads_arr = np.array(leads, dtype=float)
    days = n_samples * STEP_MIN / 1440.0
    return {
        "n_events": n_win,
        "recall": len(leads) / n_win if n_win else float("nan"),
        "early_rate": float((leads_arr >= early_min).sum() / n_win) if n_win else float("nan"),
        "median_lead_min": float(np.median(leads_arr)) if leads_arr.size else float("nan"),
        "false_alarms_per_server_day": n_false / days if days else float("nan"),
        "alert_episodes": n_alert_eps,
        "false_alarm_episodes": n_false,
        "server_days": days,
        "alert_precision": 1 - n_false / n_alert_eps if n_alert_eps else float("nan"),
        "leads_min": leads_arr.tolist(),
        "per_event": per_event,
    }


def tune_threshold(
    df: pd.DataFrame,
    score: np.ndarray | pd.Series,
    windows: dict,
    pre: int,
    budget: float = 0.5,
    k: int = 2,
    n_grid: int = 60,
) -> float:
    """Umbral más bajo (más sensible) que respeta el presupuesto de falsas alarmas por servidor y día."""
    sc = np.asarray(score)
    grid = np.unique(np.quantile(sc, np.linspace(0.90, 0.9995, n_grid)))
    for thr in grid:
        ev = evaluate(df, alerts_from_score(df, sc, thr, k), windows, pre)
        if ev["false_alarms_per_server_day"] <= budget:
            return float(thr)
    return float(grid[-1])
