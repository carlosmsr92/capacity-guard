"""Contrato de datos y validación de calidad (ingeniería de datos).

Las métricas de monitorización llegan sucias en producción: agentes que se reinician (huecos), duplicados por
reintentos, sensores atascados, valores imposibles. `validate` mide el problema y `repair` aplica correcciones
conservadoras y trazables (columna `imputed`) en lugar de ocultarlo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .schemas import METRICS as METRICS_SCHEMA
from .simulator import STEP_MIN

METRICS = list(METRICS_SCHEMA.dtypes.keys())[2:]
REQUIRED = list(METRICS_SCHEMA.required)
RANGES = METRICS_SCHEMA.ranges
STUCK_SAMPLES = 24  # 2 h sin variar en absoluto


@dataclass
class Check:
    name: str
    passed: bool
    violations: int
    severity: str  # "error" bloquea el pipeline; "warning" se registra y se sigue
    detail: str = ""


@dataclass
class QualityReport:
    rows: int
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.passed or c.severity == "warning" for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"rows": self.rows, "ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def raise_if_failed(self) -> None:
        bad = [c for c in self.failed() if c.severity == "error"]
        if bad:
            raise ValueError("Calidad de datos insuficiente: " + "; ".join(f"{c.name} ({c.violations})" for c in bad))


def validate(df: pd.DataFrame, step_min: int = STEP_MIN, max_gap_samples: int = 3,
             long_gap_severity: str = "error") -> QualityReport:
    """long_gap_severity="warning" se usa tras `repair`: los huecos largos se conservan (no se inventan datos)."""
    checks: list[Check] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    checks.append(Check("columnas_requeridas", not missing, len(missing), "error", ", ".join(missing)))
    if missing:
        return QualityReport(len(df), checks)

    vals = df[METRICS].to_numpy(dtype=float)
    bad_num = int((~np.isfinite(vals)).sum())
    checks.append(Check("sin_nulos_ni_infinitos", bad_num == 0, bad_num, "error"))

    out_of_range = 0
    for col, (lo, hi) in RANGES.items():
        x = df[col].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            out_of_range += int(((x < lo) | ((x > hi) if hi is not None else False)).sum())
    checks.append(Check("valores_en_rango", out_of_range == 0, out_of_range, "error"))

    dups = int(df.duplicated(["server", "timestamp"]).sum())
    checks.append(Check("sin_duplicados_servidor_timestamp", dups == 0, dups, "error"))

    unsorted = 0
    short_gaps = long_gaps = irregular = stuck = 0
    for _, g in df.groupby("server", sort=False):
        unsorted += int((g["timestamp"].diff().dt.total_seconds().dropna() < 0).sum())
        g = g.sort_values("timestamp")  # huecos y rejilla se miden sobre la serie ordenada
        d = g["timestamp"].diff().dropna().dt.total_seconds().to_numpy() / 60
        d = d[d > 0]
        irregular += int((~np.isclose(d % step_min, 0)).sum())
        gaps = d[d > step_min] / step_min - 1  # muestras que faltan en cada hueco
        short_gaps += int(((gaps > 0) & (gaps <= max_gap_samples)).sum())
        long_gaps += int((gaps > max_gap_samples).sum())
        cpu = g["cpu"].to_numpy(dtype=float)
        if len(cpu) >= STUCK_SAMPLES:
            run = pd.Series(np.r_[False, (np.diff(cpu) == 0) & (cpu[1:] < 100) & (cpu[1:] > 0)]).astype(int)  # CPU pegada a 100 % no es sensor roto
            stuck += int((run.rolling(STUCK_SAMPLES - 1).sum() == STUCK_SAMPLES - 1).sum() > 0)
    checks.append(Check("orden_temporal_por_servidor", unsorted == 0, unsorted, "error"))
    checks.append(Check("rejilla_regular_5min", irregular == 0, irregular, "error"))
    checks.append(Check("huecos_cortos_reparables", short_gaps == 0, short_gaps, "warning",
                        f"huecos de ≤{max_gap_samples} muestras (se interpolan)"))
    checks.append(Check("huecos_largos", long_gaps == 0, long_gaps, long_gap_severity,
                        f"huecos de >{max_gap_samples} muestras: no se inventan datos"))
    checks.append(Check("sensor_atascado_cpu", stuck == 0, stuck, "warning", "servidores con CPU constante ≥2 h"))
    return QualityReport(len(df), checks)


def repair(df: pd.DataFrame, step_min: int = STEP_MIN, max_fill: int = 3) -> tuple[pd.DataFrame, dict]:
    """Reparaciones conservadoras: dedup, orden, fuera-de-rango -> NaN, rejilla completa e interpolación corta."""
    stats = {"duplicates_removed": 0, "out_of_range_nulled": 0, "rows_added_by_grid": 0,
             "imputed_cells_filled": 0, "rows_dropped_unfillable": 0}
    d = df.copy()
    before = len(d)
    d = d.sort_values(["server", "timestamp"]).drop_duplicates(["server", "timestamp"], keep="last")
    stats["duplicates_removed"] = before - len(d)
    for col, (lo, hi) in RANGES.items():
        bad = (d[col] < lo) | (d[col] > hi if hi is not None else False) | ~np.isfinite(d[col].astype(float))
        stats["out_of_range_nulled"] += int((bad & d[col].notna()).sum())
        d.loc[bad, col] = np.nan
    parts = []
    for server, g in d.groupby("server", sort=False):
        grid = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=f"{step_min}min")
        g = g.set_index("timestamp").reindex(grid)
        stats["rows_added_by_grid"] += int(len(grid) - g["server"].notna().sum())
        g["server"] = server
        was_nan = g[METRICS].isna()
        filled = g[METRICS].interpolate(limit_area="inside")
        # solo se rellenan los huecos de ≤ max_fill muestras; en los largos no se inventa ni una muestra
        run = was_nan.ne(was_nan.shift()).cumsum()
        run_len = pd.DataFrame({c: was_nan[c].groupby(run[c]).transform("size") for c in METRICS})
        g[METRICS] = filled.where(~was_nan | (run_len <= max_fill))
        g["imputed"] = was_nan.any(axis=1) & g[METRICS].notna().all(axis=1)
        stats["imputed_cells_filled"] += int((was_nan & g[METRICS].notna()).to_numpy().sum())
        g.index.name = "timestamp"
        parts.append(g.reset_index())
    out = pd.concat(parts, ignore_index=True)
    ok = out[METRICS].notna().all(axis=1)
    stats["rows_dropped_unfillable"] = int((~ok).sum())
    out = out[ok].reset_index(drop=True)
    for c in ("event",):
        if c in out.columns:
            out[c] = out[c].fillna("")
    return out, stats
