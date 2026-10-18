"""Instantánea pequeña de resultados para desplegar el dashboard sin ejecutar el pipeline (p. ej. Streamlit Cloud).

Copia de `outputs/` solo lo que lee el dashboard, con la misma estructura de carpetas, para que baste con apuntar el
dashboard a `demo/`: resultados en JSON, columnas usadas de métricas y scores (float32) y el uso horario de las
cuentas con oferta. No incluye el modelo (`model.joblib`): el dashboard no lo necesita y así no se publica un pickle.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .store import write_parquet

METRICS_COLUMNS = ["timestamp", "server", "cpu", "ram", "iowait", "event", "sat_now"]
SCORES_COLUMNS = ["timestamp", "server", "risk_gbm", "alert_B_gbm", "split"]
USAGE_COLUMNS = ["timestamp", "account_id", "cpu", "ram", "io", "procesos"]
REQUIRED = ("results.json", "data/metrics.parquet", "data/scores.parquet", "data/alerts_explained.json")


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({c: "float32" for c in df.select_dtypes("float").columns})
    for c in ("server", "event", "split", "account_id"):
        if c in df:
            df[c] = df[c].astype("category")
    return df


def export_demo(src: str | Path = "outputs", dst: str | Path = "demo") -> dict[str, int]:
    """Escribe la instantánea en `dst` a partir de los resultados completos de `src`. Devuelve bytes por fichero."""
    src, dst = Path(src), Path(dst)
    missing = [f for f in REQUIRED if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan resultados en {src}: {missing}. Ejecuta antes `python -m capguard all`.")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)
    shutil.copy2(src / "results.json", dst / "results.json")
    shutil.copy2(src / "data" / "alerts_explained.json", dst / "data" / "alerts_explained.json")
    write_parquet(_compact(pd.read_parquet(src / "data" / "metrics.parquet", columns=METRICS_COLUMNS)),
                  dst / "data" / "metrics.parquet")
    write_parquet(_compact(pd.read_parquet(src / "data" / "scores.parquet", columns=SCORES_COLUMNS)),
                  dst / "data" / "scores.parquet")
    usage = src / "data" / "account_usage_hourly.parquet"
    offers = json.loads((src / "results.json").read_text()).get("customers", {}).get("offers", [])
    if usage.exists() and offers:
        accounts = sorted({o["account_id"] for o in offers})
        u = pd.read_parquet(usage, columns=USAGE_COLUMNS, filters=[("account_id", "in", accounts)])
        write_parquet(_compact(u.assign(account_id=u["account_id"].astype(str))),
                      dst / "data" / "account_usage_hourly.parquet")
    return {str(p.relative_to(dst)): p.stat().st_size for p in sorted(dst.rglob("*")) if p.is_file()}
