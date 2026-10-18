"""Lectura, normalización y capacidades de las fuentes de datos."""
from __future__ import annotations

import csv
import difflib
import io
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from . import schemas
from .quality import repair, validate


class IngestError(ValueError):
    """Error accionable al leer o normalizar una fuente."""


MAX_NON_NUMERIC = 0.05  # más de un 5 % de valores no numéricos en una columna es un problema de formato, no ruido
DATE_HINT = 'Usa --date-format, p. ej. --date-format "%d/%m/%Y %H:%M"'


def _to_number(s: pd.Series) -> pd.Series:
    """Convierte a número aceptando también texto con formato español (1.234,56)."""
    out = pd.to_numeric(s, errors="coerce")
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        txt = s.astype("string").str.strip()
        es = txt.str.fullmatch(r"-?\d{1,3}(\.\d{3})*(,\d+)?|-?\d+,\d+", na=False) & out.isna()
        if es.any():
            out[es] = pd.to_numeric(txt[es].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                    errors="coerce")
    return out


def _to_datetime(s: pd.Series, utc: bool, date_format: str | None) -> tuple[pd.Series, str]:
    """Interpreta fechas; decide día/mes frente a mes/día con los datos y rechaza la ambigüedad en vez de adivinar."""
    if date_format:
        return pd.to_datetime(s, errors="coerce", utc=utc, format=date_format), f"formato {date_format}"
    parts = s.astype("string").str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
    if parts[0].isna().all():  # ISO u otro formato sin ambigüedad día/mes
        return pd.to_datetime(s, errors="coerce", utc=utc), ""
    first, second = parts[0].astype(float), parts[1].astype(float)
    if (first > 12).any() and (second > 12).any():
        raise IngestError("Las fechas mezclan día/mes y mes/día. " + DATE_HINT)
    if not (first > 12).any() and not (second > 12).any():
        raise IngestError("Fechas ambiguas: se pueden leer como día/mes o como mes/día. " + DATE_HINT)
    dayfirst = bool((first > 12).any())
    parsed = pd.to_datetime(s, errors="coerce", utc=utc, dayfirst=dayfirst, format="mixed")
    return parsed, "fechas día/mes" if dayfirst else "fechas mes/día"


@dataclass
class IngestReport:
    rows_read: int
    rows_final: int
    ignored_columns: list[str]
    non_numeric: dict[str, int]
    unit_conversions: list[str]
    frequency: str = ""
    bad_timestamps_dropped: int = 0
    quality_before: dict = None
    quality_after: dict = None
    event_labels: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Dataset:
    metrics: pd.DataFrame
    accounts: pd.DataFrame | None
    usage: pd.DataFrame | None
    plans: pd.DataFrame | None
    report: dict  # IngestReport.to_dict() en CSV; informe de ingesta del feed sucio en simulación
    source: str
    capabilities: dict[str, bool | str]


def _clean(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().strip().lower()
    text = text.replace("%", "percent")
    return re.sub(r"\s+", "_", text)


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Lee CSVs habituales y convierte errores de lectura en mensajes claros."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise IngestError(f"No se puede leer {p}: {exc}") from exc
    text = None
    encoding = ""
    for candidate in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise IngestError(f"No se puede decodificar {p}; prueba UTF-8 o latin-1")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    decimal = "," if sep == ";" and re.search(r"\d,\d", text[:8192]) else "."
    try:  # los números con separador de miles (1.234,5) quedan como texto y los convierte _to_number columna a columna
        return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, encoding=encoding)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise IngestError(f"No se puede interpretar {p}: {exc}") from exc


def _mapping_columns(df: pd.DataFrame, schema: schemas.Schema, mapping: dict | None) -> tuple[pd.DataFrame, list[str]]:
    explicit = mapping or {}
    aliases = {_clean(alias): canonical for canonical, names in schema.aliases.items() for alias in names}
    aliases.update({_clean(c): c for c in schema.required + schema.optional})
    renamed: dict[object, str] = {}
    ignored = []
    for original in df.columns:
        clean = _clean(original)
        target = explicit.get(str(original), explicit.get(clean, aliases.get(clean)))
        if isinstance(target, dict):
            target = target.get("to") or (clean if clean in schema.required else None)
        if target in schema.required + schema.optional:
            renamed[original] = target
        else:
            ignored.append(str(original))
    out = df.rename(columns=renamed)
    if len(set(out.columns)) != len(out.columns):
        raise IngestError("El mapeo produce columnas canónicas duplicadas")
    missing = [c for c in schema.required if c not in out.columns]
    if missing:
        suggestions = []
        available = [_clean(c) for c in df.columns]
        for col in missing:
            close = difflib.get_close_matches(col, available, n=1, cutoff=0.55)
            suggestions.append(f"{col}" + (f" (¿quisiste decir {close[0]}?)" if close else ""))
        raise IngestError("Faltan columnas obligatorias: " + ", ".join(suggestions) + ". Usa --mapping.")
    return out, ignored


def normalize(df: pd.DataFrame, schema: schemas.Schema = schemas.METRICS, mapping: dict | None = None,
              tz: str | None = None, date_format: str | None = None) -> tuple[pd.DataFrame, IngestReport]:
    """Normaliza cabeceras, tipos, unidades y rejilla de una fuente."""
    d, ignored = _mapping_columns(df.copy(), schema, mapping)
    rows_read = len(d)
    d = d[[c for c in d.columns if c in schema.required + schema.optional]]  # fuera columnas no canónicas
    bad_ts = 0
    date_note = ""
    non_numeric: dict[str, int] = {}
    for col, dtype in schema.dtypes.items():
        if col not in d:
            continue
        if dtype == "number":
            old = d[col]
            converted = _to_number(old)
            bad = converted.isna() & old.notna()
            non_numeric[col] = int(bad.sum())
            if col in schema.required and len(d) and bad.mean() > MAX_NON_NUMERIC:
                examples = ", ".join(repr(v) for v in old[bad].astype(str).unique()[:3])
                raise IngestError(f"La columna {col} tiene un {100 * bad.mean():.0f} % de valores no numéricos "
                                  f"(p. ej. {examples}); revisa el separador decimal o de miles del CSV")
            d[col] = converted
        elif dtype == "datetime":
            parsed, how = _to_datetime(d[col], bool(tz), date_format)
            if how and how != "fechas mes/día":
                date_note = how
            if parsed.isna().mean() >= 0.01:
                sample = ", ".join(repr(v) for v in d.loc[parsed.isna(), col].astype(str).unique()[:3])
                raise IngestError(f"El {100 * parsed.isna().mean():.0f} % de los timestamps de {col} no se puede "
                                  f"interpretar (p. ej. {sample}). {DATE_HINT}")
            bad_ts += int(parsed.isna().sum())
            d = d[parsed.notna()]
            parsed = parsed[parsed.notna()]
            if tz:
                d[col] = parsed.dt.tz_convert(tz).dt.tz_localize(None)
            else:
                d[col] = parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) else parsed
    conversions = [date_note] if date_note else []
    event_labels = None
    if "event" in d:
        d["event"] = d["event"].fillna("").astype(str).str.strip()
        d["event"] = d["event"].replace((mapping or {}).get("event_values", {}))
        counts = d.loc[d["event"].ne(""), "event"].value_counts()
        event_labels = {str(k): int(v) for k, v in counts.items()}
    # Fracción (0-1) en lugar de porcentaje: se decide con CPU y RAM juntas y se aplica a las tres columnas.
    # Una sola columna con máximo ≤ 1 no basta: un iowait real ocioso (0,4 %) es perfectamente posible.
    pct = [c for c in ("cpu", "ram", "iowait") if c in d]
    if {"cpu", "ram"} <= set(pct) and d["cpu"].dropna().size and d["cpu"].max() <= 1 and d["ram"].max() <= 1:
        for col in pct:
            d[col] *= 100
        conversions.append(f"{', '.join(pct)}: fracción→porcentaje")
    for source, specification in (mapping or {}).items():
        if not isinstance(specification, dict):
            continue
        target = specification.get("to", source if source in d.columns else None)
        unit = specification.get("from")
        if target not in d.columns or not unit:
            continue
        if unit == "bytes_per_s" and target == "net":
            d[target] *= 8 / 1_000_000
            conversions.append("net: bytes_per_s→Mbps")
        elif unit == "fraction" and target in ("cpu", "ram", "iowait") and not any(target in c for c in conversions):
            d[target] *= 100
            conversions.append(f"{target}: fracción→porcentaje (mapping)")
    quality_before = None
    frequency = ""
    if schema.name == "metrics":
        d = d.sort_values(["server", "timestamp"]).reset_index(drop=True)
        diffs = d.groupby("server")["timestamp"].diff().dropna().dt.total_seconds().div(60)
        if len(diffs):
            median = float(diffs.median())
            frequency = f"{median:g} min"
            if median > 5:
                raise IngestError(f"Resolución detectada de {median:g} min; el modelo necesita resolución de 5 min")
        if len(diffs) and float(diffs.median()) < 5:
            parts = []
            for server, group in d.groupby("server", sort=False):
                group = group.set_index("timestamp").sort_index()
                aggregations = {column: "mean" for column, dtype in schema.dtypes.items()
                                if column in group and dtype == "number"}
                if "load1" in aggregations:
                    aggregations["load1"] = "max"
                for column in ("event", "event_account"):
                    if column in group:
                        aggregations[column] = "last"
                reduced = group.resample("5min").agg(aggregations)
                reduced["server"] = server
                parts.append(reduced.reset_index())
            d = pd.concat(parts, ignore_index=True)
            frequency = f"{frequency}→5 min"
        quality_before = validate(d).to_dict()
        fixed, _ = repair(d)
        after = validate(fixed, long_gap_severity="warning")
        after.raise_if_failed()
        d = fixed.drop(columns=["imputed"], errors="ignore")
        if d.empty:
            raise IngestError("No queda ninguna fila válida tras normalizar y reparar el CSV")
        report = IngestReport(rows_read, len(d), ignored, non_numeric, conversions, frequency, bad_ts,
                              quality_before, after.to_dict(), event_labels)
    else:
        report = IngestReport(rows_read, len(d), ignored, non_numeric, conversions, bad_timestamps_dropped=bad_ts)
    return d, report


MIN_DAYS = 28
MIN_EPISODES = 3


def check_sufficient(metrics: pd.DataFrame) -> None:
    """Sin bastante historia o sin episodios de saturación en validación y test no se pueden calibrar umbrales."""
    from .evaluation import saturation_windows
    from .pipeline import data_days, split_masks
    days = data_days(metrics)
    if days < MIN_DAYS:
        raise IngestError(f"Datos insuficientes para calibrar umbrales: {days:.1f} días (mínimo {MIN_DAYS})")
    masks = split_masks(metrics)
    for part in ("val", "test"):
        n = sum(len(w) for w in saturation_windows(metrics[masks[part]]).values())
        if n < MIN_EPISODES:
            raise IngestError(f"Datos insuficientes para calibrar umbrales: {n} episodios de saturación en {part} "
                              f"(mínimo {MIN_EPISODES})")


def capabilities(dataset: Dataset) -> dict[str, bool | str]:
    """Qué partes del análisis se pueden calcular con estos datos y, si no, por qué."""
    has_event = "event" in dataset.metrics.columns and dataset.metrics["event"].astype(bool).any()
    customers = dataset.accounts is not None and dataset.usage is not None
    archetypes = customers and "archetype" in dataset.accounts.columns
    result: dict[str, bool | str] = {
        "task_a": bool(has_event),
        "diagnosis": bool(has_event),
        "incident_eda": bool(has_event),
        "customers": bool(customers),
        "ground_truth_archetypes": bool(archetypes),
    }
    if not has_event:
        result["task_a_reason"] = "Los datos no contienen la columna event (etiquetas de incidentes)"
        result["diagnosis_reason"] = "Sin etiquetas de incidentes no se puede entrenar ni evaluar el diagnóstico"
        result["incident_eda_reason"] = "Sin etiquetas de incidentes no hay análisis por tipo de incidente"
    if not customers:
        result["customers_reason"] = "Faltan los datos de cuentas (accounts) o de uso por cuenta (usage)"
    if not archetypes:
        result["ground_truth_archetypes_reason"] = "Solo la simulación conoce el arquetipo real de cada cuenta"
    return result


def _labelled(metrics: pd.DataFrame) -> pd.DataFrame:
    from .simulator import add_labels
    d = metrics.copy()
    for col in ("event", "event_account"):
        d[col] = d[col].fillna("").astype(str) if col in d else ""
    return add_labels(d)


def load_source(kind: Literal["sim", "csv"], cfg, paths: dict | None = None,
                mapping: dict | None = None, tz: str | None = None, check: bool = True,
                date_format: str | None = None) -> Dataset:
    """Punto único de entrada de datos: simulación o CSV, ambos validados con los mismos esquemas.

    check=False omite la comprobación de datos suficientes para entrenar (útil para validar un CSV en seco).
    """
    paths = paths or {}
    if kind == "sim":
        from .ingest import ingest
        from .simulator import simulate_world
        world = simulate_world(cfg)
        normalize(world.metrics, schemas.METRICS)  # el simulador también cumple el contrato canónico
        metrics, info = ingest(world.metrics, seed=cfg.seed)  # demostración de feed sucio -> reparado
        dataset = Dataset(metrics, world.accounts, world.usage_hourly, world.plans, info, "sim", {})
    elif kind == "csv":
        if not paths.get("metrics"):
            raise IngestError("La fuente csv requiere --metrics")
        metrics, report = normalize(read_csv_robust(paths["metrics"]), schemas.METRICS, mapping, tz, date_format)
        metrics = _labelled(metrics)
        if check:
            check_sufficient(metrics)
        frames = {}
        for name in ("accounts", "usage", "plans"):
            if paths.get(name):
                schema = getattr(schemas, name.upper())
                frames[name], _ = normalize(read_csv_robust(paths[name]), schema, mapping, tz, date_format)
        dataset = Dataset(metrics, frames.get("accounts"), frames.get("usage"), frames.get("plans"),
                          report.to_dict(), "csv", {})
    else:
        raise IngestError(f"Fuente desconocida: {kind}")
    dataset.capabilities = capabilities(dataset)
    return dataset
