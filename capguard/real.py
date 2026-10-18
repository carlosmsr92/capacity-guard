"""Validación sobre datos REALES públicos: métricas de AWS CloudWatch del benchmark NAB (Numenta).

Son series univariantes de CPU/red/peticiones con ventanas de anomalía etiquetadas por humanos.
Aquí solo se puede usar un detector univariante (z-score robusto con ventana móvil de 24 h); sirve
para comprobar que la lógica de evaluación y el enfoque funcionan fuera de la simulación.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import episodes, persistent

Z_THR = 4.0
# commit fijo de NAB (no la rama master): los datos de la validación real no pueden cambiar sin que se note
NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
BASE = f"https://raw.githubusercontent.com/numenta/NAB/{NAB_COMMIT}"
SERIES = [
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
]


def _get(rel: str, cache: Path) -> Path:
    dest = cache / rel.replace("/", "__")
    if not dest.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{rel}", timeout=30) as r:
            dest.write_bytes(r.read())
    return dest


def rolling_robust_z(x: pd.Series, floor: float, window: int = 288, min_periods: int = 48) -> pd.Series:
    """|z| robusto contra la mediana de las últimas 24 h (solo pasado). `floor` evita dividir por una
    dispersión casi nula en series planas, que dispararía falsas alarmas."""
    med = x.rolling(window, min_periods=min_periods).median().shift(1)
    mad = (x - med).abs().rolling(window, min_periods=min_periods).median().shift(1) * 1.4826
    return ((x - med) / np.maximum(mad, floor)).abs()


def evaluate_series(df: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]],
                    z_thr: float = Z_THR, k: int = 2, probation: float = 0.15, floor_mult: float = 0.5) -> dict:
    start = int(len(df) * probation)  # NAB reserva el 15 % inicial como periodo de aprendizaje
    z = rolling_robust_z(df["value"], floor_mult * float(df["value"].iloc[:start].std())).fillna(0.0)
    alert = persistent((z >= z_thr).to_numpy(), k)
    alert[:start] = False
    ts = df["timestamp"]
    hits, delays = 0, []
    for s, e in windows:
        idx = np.flatnonzero(((ts >= s) & (ts <= e)).to_numpy() & alert)
        if idx.size:
            hits += 1
            delays.append((ts.iloc[idx[0]] - s).total_seconds() / 60)
    in_win = np.zeros(len(df), bool)
    for s, e in windows:
        in_win |= ((ts >= s) & (ts <= e + pd.Timedelta(hours=1))).to_numpy()
    eps = episodes(alert)
    false = sum(1 for a, b in eps if not in_win[a : b + 1].any())
    days = (len(df) - start) * 5 / 1440
    return {"windows": len(windows), "detected": hits, "false_alarm_episodes": false,
            "false_alarms_per_day": false / days if days else float("nan"),
            "median_delay_min": float(np.median(delays)) if delays else float("nan")}


def run_real(cache: str | Path = "data/real", z_thr: float = Z_THR) -> pd.DataFrame:
    cache = Path(cache)
    labels = json.loads(_get("labels/combined_windows.json", cache).read_text())
    rows = []
    for rel in SERIES:
        df = pd.read_csv(_get(f"data/{rel}", cache), parse_dates=["timestamp"])
        wins = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in labels[rel]]
        rows.append({"serie": rel.split("/")[1].removesuffix(".csv")} | evaluate_series(df, wins, z_thr))
    return pd.DataFrame(rows)


def summary_html(res: pd.DataFrame, z_thr: float) -> str:
    tot_w, tot_d = int(res.windows.sum()), int(res.detected.sum())
    rows = "".join(
        f"<tr><td>{r.serie}</td><td>{r.windows}</td><td>{r.detected}</td><td>{r.false_alarms_per_day:.2f}</td></tr>"
        for r in res.itertuples())
    return (f"<h2>Validación con datos reales (AWS CloudWatch · NAB)</h2><p>Detector univariante (z-score robusto "
            f"de ventana móvil de 24 h, umbral {z_thr}) sobre {len(res)} series públicas con ventanas de anomalía "
            f"etiquetadas: <b>{tot_d} de {tot_w}</b> ventanas detectadas, con {res.false_alarms_per_day.mean():.2f} "
            f"falsas alarmas por serie y día de media. Solo es una comprobación del enfoque: el benchmark no incluye "
            f"saturaciones de servidores de hosting y los dos parámetros del detector (umbral y suelo de dispersión) "
            f"se eligieron mirando estas mismas series, así que el resultado es optimista.</p><table><tr><th>Serie</th>"
            f"<th>Ventanas</th><th>Detectadas</th><th>Falsas alarmas/día</th></tr>{rows}</table>")
