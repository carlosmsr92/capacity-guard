"""Análisis de negocio: qué pasa, cuándo, dónde y cuánto cuesta (analista de datos / BI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import incident_windows
from .simulator import STEP_MIN

# Supuestos ILUSTRATIVOS (euros). Hay que sustituirlos por datos reales de la empresa (SLA, coste de guardia).
DEFAULT_COSTS = {"c_miss": 200.0, "c_fa": 15.0, "c_mitigated": 20.0}


def incident_table(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por incidente real: tipo, servidor, inicio, duración y si llegó a saturar."""
    rows = []
    for server, g in df.groupby("server", sort=False):
        g = g.reset_index(drop=True)
        for s, e in incident_windows(g)[server]:
            w = g.iloc[s : e + 1]
            rows.append({
                "server": server, "kind": g.loc[s, "event"], "start": g.loc[s, "timestamp"],
                "hour": int(g.loc[s, "timestamp"].hour), "duration_min": (e - s + 1) * STEP_MIN,
                "saturated": bool(w["sat_now"].any()),
                "saturated_min": int(w["sat_now"].sum()) * STEP_MIN,
            })
    return pd.DataFrame(rows)


def eda(df: pd.DataFrame) -> dict:
    inc = incident_table(df)
    days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    by_kind = inc.groupby("kind").agg(
        incidentes=("kind", "size"), duracion_media_min=("duration_min", "mean"),
        pct_que_satura=("saturated", "mean"), min_saturados=("saturated_min", "sum")).round(2)
    by_kind["pct_min_saturados"] = (by_kind["min_saturados"] / by_kind["min_saturados"].sum()).round(3)
    by_server = inc.groupby("server").agg(incidentes=("kind", "size"), min_saturados=("saturated_min", "sum"))
    by_hour = inc.groupby("hour").size().reindex(range(24), fill_value=0)
    return {
        "days": days, "n_incidents": int(len(inc)),
        "incidents_per_server_month": float(len(inc) / df["server"].nunique() / days * 30),
        "by_kind": by_kind.reset_index().to_dict("records"),
        "by_server": by_server.reset_index().to_dict("records"),
        "by_hour": by_hour.tolist(),
        "worst_kind_for_saturation": str(by_kind["min_saturados"].idxmax()),
        "saturated_share_of_time": float(df["sat_now"].mean()),
    }


def cost_table(task_b_methods: dict, costs: dict | None = None, early_min: int = 10) -> pd.DataFrame:
    """Coste de cada método en el periodo de test: avisadas a tiempo + no avisadas + falsas alarmas."""
    c = {**DEFAULT_COSTS, **(costs or {})}
    rows = []
    for name, m in task_b_methods.items():
        n, leads = m["n_events"], np.array(m["leads_min"], dtype=float)
        early = int((leads >= early_min).sum())
        missed, fa = n - early, int(m["false_alarm_episodes"])
        rows.append({
            "method": m["method"], "key": name, "saturaciones": n, "avisadas_a_tiempo": early,
            "no_avisadas": missed, "falsas_alarmas": fa,
            "coste_total": early * c["c_mitigated"] + missed * c["c_miss"] + fa * c["c_fa"],
            "coste_por_servidor_mes": None,
        })
    d = pd.DataFrame(rows)
    sd = next(iter(task_b_methods.values()))["server_days"]
    d["coste_por_servidor_mes"] = (d["coste_total"] / sd * 30).round(2)
    d["ahorro_vs_umbral_pct"] = ((1 - d["coste_total"] / d.loc[d["key"] == "threshold", "coste_total"].iloc[0]) * 100).round(1)
    return d


def sensitivity(task_b_methods: dict, c_miss_grid=(50, 100, 200, 500, 1000), c_fa_grid=(5, 15, 50, 150)) -> pd.DataFrame:
    """Método más barato para cada combinación de coste de no avisar / coste de una falsa alarma."""
    rows = []
    for cm in c_miss_grid:
        for cf in c_fa_grid:
            t = cost_table(task_b_methods, {"c_miss": cm, "c_fa": cf})
            best = t.loc[t["coste_total"].idxmin()]
            rows.append({"c_miss": cm, "c_fa": cf, "mas_barato": best["method"], "key": best["key"]})
    return pd.DataFrame(rows)
