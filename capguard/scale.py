"""Prueba de escala (big data): las mismas variables calculadas con pandas y con DuckDB sobre Parquet.

Se calculan 4 variables de producción (z_cpu, z_ram, z_cpu_m6, ram_slope) y se mide tiempo y memoria pico de
cada motor en un proceso aparte (pico propio de ese proceso, VmHWM), para que la comparación sea honesta. Un test
comprueba que ambos motores devuelven los mismos números.
"""
from __future__ import annotations

import json
import os
import resource  # solo Unix (Linux/macOS/WSL): memoria pico por proceso
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import SimConfig, simulate

FEATURES = ["z_cpu", "z_ram", "z_cpu_m6", "ram_slope"]
FLOORS = {"cpu": 1.0, "ram": 0.7}


def machine() -> str:
    """Descripción de la máquina donde se mide (las cifras de tiempo y memoria dependen de ella)."""
    cpus = os.cpu_count() or 0
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        return f"{cpus} vCPU, {ram:.0f} GB RAM"
    except (ValueError, OSError, AttributeError):
        return f"{cpus} vCPU"


def generate_parquet_dataset(out_dir: str | Path, n_servers: int, days: int, chunk: int = 25, seed: int = 11) -> int:
    """Escribe el dataset en varios ficheros Parquet (por lotes de servidores) sin cargarlo entero en RAM."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = 0
    for i, first in enumerate(range(0, n_servers, chunk)):
        n = min(chunk, n_servers - first)
        df = simulate(SimConfig(days=days, n_servers=n, seed=seed + i, accounts_per_server=8))
        df["server"] = df["server"].str.replace("web-", f"web-{i:03d}-", regex=False)
        df[["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"]].astype(
            {c: "float32" for c in ("cpu", "ram", "iowait", "net", "load1", "req")}
        ).to_parquet(out / f"part-{i:04d}.parquet", compression="zstd", index=False)
        rows += len(df)
    return rows


def pandas_features(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    d = df[["server", "timestamp", "cpu", "ram"]].copy()
    d["how"] = d["timestamp"].dt.dayofweek * 24 + d["timestamp"].dt.hour
    tr = d["timestamp"] < cutoff
    for c in ("cpu", "ram"):
        base = d[tr].groupby(["server", "how"])[c].median().rename("base")
        d = d.join(base, on=["server", "how"])
        res = d[c] - d["base"]
        r_tr = res[tr]
        med = r_tr.groupby(d.loc[tr, "server"]).median().rename("m")
        mad = (r_tr - d.loc[tr, "server"].map(med)).abs().groupby(d.loc[tr, "server"]).median() * 1.4826
        scale = np.maximum(mad, FLOORS[c])
        d[f"z_{c}"] = res / d["server"].map(scale)
        d = d.drop(columns="base")
    d = d.sort_values(["server", "timestamp"])
    g = d.groupby("server", sort=False)
    d["z_cpu_m6"] = g["z_cpu"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    d["ram_slope"] = ((d["ram"] - g["ram"].shift(12)) / 12).fillna(0.0)
    return d.set_index(["server", "timestamp"])[FEATURES]


def duckdb_features_sql(parquet_glob: str, cutoff: str) -> str:
    def block(c: str, floor: float) -> str:
        return f"""
        base_{c} AS (SELECT server, how, median({c}) AS b FROM t WHERE ts < TIMESTAMP '{cutoff}' GROUP BY 1, 2),
        res_{c} AS (SELECT t.server, t.ts, t.{c} - b.b AS r FROM t JOIN base_{c} b USING (server, how)),
        med_{c} AS (SELECT server, median(r) AS m FROM res_{c} WHERE ts < TIMESTAMP '{cutoff}' GROUP BY 1),
        sc_{c} AS (SELECT r.server, greatest(median(abs(r.r - m.m)) * 1.4826, {floor}) AS s
                   FROM res_{c} r JOIN med_{c} m USING (server) WHERE r.ts < TIMESTAMP '{cutoff}' GROUP BY 1)"""
    return f"""
    WITH t AS (SELECT server, timestamp AS ts, cpu, ram,
                      (isodow(timestamp) - 1) * 24 + hour(timestamp) AS how
               FROM read_parquet('{parquet_glob}')),
    {block('cpu', FLOORS['cpu'])}, {block('ram', FLOORS['ram'])},
    z AS (SELECT t.server, t.ts, t.ram,
                 (t.cpu - bc.b) / sc.s AS z_cpu, (t.ram - br.b) / sr.s AS z_ram
          FROM t LEFT JOIN base_cpu bc USING (server, how) LEFT JOIN base_ram br USING (server, how)
                 JOIN sc_cpu sc USING (server) JOIN sc_ram sr USING (server))
    SELECT server, ts AS timestamp, z_cpu, z_ram,
           avg(z_cpu) OVER w6 AS z_cpu_m6,
           coalesce((ram - lag(ram, 12) OVER w) / 12, 0) AS ram_slope
    FROM z
    WINDOW w AS (PARTITION BY server ORDER BY ts),
           w6 AS (PARTITION BY server ORDER BY ts ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)"""


def duckdb_features(parquet_glob: str, cutoff: str, limit_threads: int | None = None):
    import duckdb
    con = duckdb.connect()
    if limit_threads:
        con.execute(f"PRAGMA threads={limit_threads}")
    return con, duckdb_features_sql(parquet_glob, cutoff)


def peak_rss_mb() -> float:
    """Memoria pico de ESTE proceso. En Linux se lee VmHWM: ru_maxrss arrastra el pico del proceso padre a través
    del fork, así que todos los motores marcarían la memoria del padre que los lanzó."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 0)
    except OSError:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)  # Unix sin /proc


def _worker(engine: str, glob: str, cutoff: str) -> None:
    t0 = time.perf_counter()
    if engine == "pandas":
        df = pd.concat([pd.read_parquet(p) for p in sorted(Path(glob).parent.glob(Path(glob).name))])
        feats = pandas_features(df, pd.Timestamp(cutoff))
        checksum = float(np.nansum(feats.to_numpy(dtype="float64")))
        n = len(feats)
    else:
        con, sql = duckdb_features(glob, cutoff)
        n, checksum = con.execute(
            f"SELECT count(*), sum(coalesce(z_cpu, 0) + coalesce(z_ram, 0) + coalesce(z_cpu_m6, 0) + coalesce(ram_slope, 0)) FROM ({sql})").fetchone()
    print(json.dumps({"engine": engine, "rows": int(n), "seconds": round(time.perf_counter() - t0, 2),
                      "peak_rss_mb": peak_rss_mb(),
                      "checksum": round(float(checksum), 3)}))


def benchmark(data_dir: str | Path, cutoff: str, engines=("pandas", "duckdb"), timeout: int = 1800) -> list[dict]:
    glob = str(Path(data_dir) / "part-*.parquet")
    res = []
    for e in engines:
        p = subprocess.run([sys.executable, "-m", "capguard.scale", e, glob, cutoff],
                           capture_output=True, text=True, timeout=timeout)
        res.append(json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0
                   else {"engine": e, "error": p.stderr[-300:]})
    return res


if __name__ == "__main__":
    _worker(*sys.argv[1:4])
