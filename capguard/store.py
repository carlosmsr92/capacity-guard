"""Almacenamiento columnar (Parquet + zstd) y consultas analíticas con DuckDB."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return path


def write_usage_parquet(usage: pd.DataFrame, path: str | Path) -> Path:
    """Uso horario por cuenta compacto: float32, sin los límites (se deducen del plan) y textos como categorías."""
    d = usage.drop(columns=[c for c in usage.columns if c.endswith("_limit")])
    d = d.astype({c: "float32" for c in d.select_dtypes("float").columns})
    if "minutos_limitado" in d:
        d["minutos_limitado"] = d["minutos_limitado"].astype("int16")
    for c in ("account_id", "server", "plan", "incidente"):
        if c in d:
            d[c] = d[c].astype("category")
    return write_parquet(d, path)


def connect(parquet: str | Path, view: str = "metrics") -> duckdb.DuckDBPyConnection:
    """Conexión en memoria con una vista sobre el/los Parquet (acepta globs). No carga todo en RAM."""
    con = duckdb.connect()
    path = Path(parquet).as_posix().replace("'", "''")  # literal SQL: las comillas se duplican
    con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
    return con


def read(parquet: str | Path) -> pd.DataFrame:
    return pd.read_parquet(parquet)
