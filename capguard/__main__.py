"""Interfaz de línea de comandos de capguard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import schemas
from .bundle import load_model, save_model
from .features import FORECAST_FEATURES
from .pipeline import benchmark, run
from .report import build_report
from .simulator import SimConfig
from .sources import IngestError, check_sufficient, load_source, normalize, read_csv_robust


def _jsonable(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def cmd_run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(days=a.days, n_servers=a.servers, seed=a.seed)
    art = run(cfg, budget=a.budget)
    art.df.drop(columns=["sat_future"]).to_csv(out / "sim_metrics.csv.gz", index=False)
    lite = json.loads(json.dumps(art.results, default=_jsonable))
    for t in lite["tasks"].values():
        for m in t["methods"].values():
            m.pop("per_event", None)
    (out / "results.json").write_text(json.dumps(lite, indent=2, ensure_ascii=False))
    save_model(out / "model.joblib", feature_builder=art.models["feature_builder"], forecaster=art.models["gbm"],
               threshold=art.thresholds[("B", "gbm")], persistence=2, features=FORECAST_FEATURES)
    bench = benchmark([a.seed + i + 1 for i in range(a.bench_seeds)], budget=a.budget) if a.bench_seeds else None
    note = ""
    if a.real:
        from .real import Z_THR, run_real, summary_html
        try:
            res = run_real(out / "real_cache")
            res.to_csv(out / "real_results.csv", index=False)
            note = summary_html(res, Z_THR)
        except Exception as e:  # sin red: el resto del informe sigue siendo válido
            print(f"[aviso] no se pudo validar con datos reales: {e}")
    (out / "report.html").write_text(build_report(art, bench, note), encoding="utf-8")
    m = lite["tasks"]["B"]["methods"]["gbm"]
    print(f"Listo. Informe: {out/'report.html'}\nAviso temprano GBM: {m['early_rate']:.0%} con "
          f"≥10 min de antelación, {m['false_alarms_per_server_day']:.2f} falsas alarmas/servidor·día")


def cmd_all(a):
    from .orchestrate import build_all
    cfg = SimConfig(days=a.days, n_servers=a.servers, seed=a.seed)
    dataset = None
    if a.source == "csv":
        try:
            dataset = load_source(a.source, cfg, {"metrics": a.metrics, "accounts": a.accounts,
                                                  "usage": a.usage, "plans": a.plans},
                                  json.loads(Path(a.mapping).read_text()) if a.mapping else None, a.tz,
                                  date_format=a.date_format)
        except (IngestError, OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Error de ingesta: {exc}") from exc
    build_all(a.out, cfg, a.budget, a.bench_seeds, a.real, a.scale_servers, a.scale_days, a.llm,
              dataset=dataset)


def cmd_validate_csv(a):
    try:
        mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
        dataset = load_source("csv", SimConfig(), {"metrics": a.metrics, "accounts": a.accounts,
                                                    "usage": a.usage, "plans": a.plans}, mapping, a.tz,
                              check=False, date_format=a.date_format)
    except (IngestError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        check_sufficient(dataset.metrics)
    except IngestError as exc:
        print(f"[aviso] el CSV es válido pero no basta para entrenar con `all --source csv`: {exc}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset.metrics.to_parquet(out / "metrics.parquet", index=False)
    (out / "ingest_report.json").write_text(json.dumps(dataset.report, indent=2, ensure_ascii=False))
    print(json.dumps(dataset.report, indent=2, ensure_ascii=False))
    print(json.dumps({"capabilities": dataset.capabilities}, indent=2, ensure_ascii=False))


def cmd_export_sample(a):
    """CSV de ejemplo (3 servidores × 7 días): sirven para probar validate-csv; para entrenar hacen falta ≥ 28 días."""
    from .simulator import simulate_world
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # empieza el día 8 para que haya días > 12 y la fecha día/mes no sea ambigua
    w = simulate_world(SimConfig(days=7, n_servers=3, seed=7, accounts_per_server=10, start="2026-06-08"))
    cols = ["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req", "event"]
    num = {c: 3 for c in cols[2:-1]}
    w.metrics[cols].round(num).to_csv(out / "metrics.csv", index=False)
    dirty = w.metrics[cols[:-1]].round(num).assign(timestamp=w.metrics["timestamp"].dt.strftime("%d/%m/%Y %H:%M"))
    for c in ("cpu", "ram", "iowait"):
        dirty[c] = (dirty[c] / 100).round(4)  # fracciones en lugar de porcentajes
    dirty = dirty.rename(columns={"timestamp": "Fecha", "server": "Servidor", "cpu": "CPU %", "ram": "Memoria",
                                  "req": "Peticiones"})
    dirty.to_csv(out / "metrics_dirty.csv", sep=";", decimal=",", index=False, encoding="latin-1")
    w.accounts.drop(columns="archetype").to_csv(out / "accounts.csv", index=False)  # el arquetipo es oculto
    w.usage_hourly[["timestamp", "account_id", "cpu", "ram", "io", "procesos", "minutos_limitado"]].round(
        {c: 4 for c in ("cpu", "ram", "io", "procesos")}).to_csv(
        out / "usage.csv", index=False)
    w.plans.to_csv(out / "plans.csv", index=False)
    print(f"Ejemplos escritos en {out}: metrics.csv, metrics_dirty.csv (; , latin-1, fracciones, día/mes), "
          "accounts.csv, usage.csv y plans.csv")


def cmd_scale(a):
    import shutil
    import tempfile

    from . import scale
    d = tempfile.mkdtemp()
    try:
        n = scale.generate_parquet_dataset(d, a.servers, a.days)
        cut = str((pd.Timestamp(SimConfig().start) + pd.Timedelta(days=a.days * 0.6)).date())
        print(n, "filas")
        print(json.dumps(scale.benchmark(d, cut, tuple(a.engines)), indent=1))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cmd_readme(a):
    from .summary import update_readme
    update_readme(a.readme, Path(a.out) / "results.json")
    print("README actualizado con los resultados de", Path(a.out) / "results.json")


def cmd_benchmark(a):
    print(benchmark(list(range(a.seed, a.seed + a.n)), budget=a.budget).round(3).to_string())


def cmd_real(a):
    from .real import run_real
    print(run_real(a.cache).round(2).to_string(index=False))


def cmd_score(a):
    """Puntúa un CSV de métricas reales con el modelo entrenado y marca las alarmas."""
    bundle = load_model(a.model)
    try:
        df, report = normalize(read_csv_robust(a.input), schemas.METRICS, tz=a.tz, date_format=a.date_format)
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    if report.quality_before and not report.quality_before["ok"]:
        print("[aviso] datos de entrada normalizados y reparados")
    df = df.sort_values(["server", "timestamp"]).reset_index(drop=True)
    fb = bundle["feature_builder"]
    unknown = fb.unknown_servers(df)
    if unknown:
        print(f"[aviso] servidores no vistos al entrenar (se usa el perfil medio de la flota): {unknown}")
    feats = fb.transform(df)
    df["risk_30min"] = bundle["forecaster"].model.predict_proba(feats[FORECAST_FEATURES].to_numpy())[:, 1]
    from .evaluation import alerts_from_score
    df["alert"] = alerts_from_score(df, df["risk_30min"], bundle["threshold"], bundle["persistence"])
    df[["timestamp", "server", "risk_30min", "alert"]].to_csv(a.out, index=False)
    print(f"{int(df.alert.sum())} muestras en alarma de {len(df)} -> {a.out}")


def main():
    p = argparse.ArgumentParser(prog="capguard")
    sub = p.add_subparsers(required=True)
    r = sub.add_parser("run", help="simula, entrena, evalúa y genera el informe")
    r.add_argument("--days", type=int, default=45); r.add_argument("--servers", type=int, default=8)
    r.add_argument("--seed", type=int, default=7); r.add_argument("--budget", type=float, default=0.5)
    r.add_argument("--bench-seeds", type=int, default=0, help="repeticiones con otras semillas")
    r.add_argument("--real", action="store_true", help="añade validación con datos reales NAB (requiere red)")
    r.add_argument("--out", default="outputs"); r.set_defaults(f=cmd_run)
    al = sub.add_parser("all", help="el proyecto completo: ingesta, calidad, modelos, análisis, escala, informe, dashboard")
    al.add_argument("--days", type=int, default=45); al.add_argument("--servers", type=int, default=8)
    al.add_argument("--seed", type=int, default=7); al.add_argument("--budget", type=float, default=0.5)
    al.add_argument("--bench-seeds", type=int, default=3); al.add_argument("--real", action="store_true")
    al.add_argument("--scale-servers", type=int, default=100); al.add_argument("--scale-days", type=int, default=30)
    al.add_argument("--llm", action="store_true", help="usar un LLM (requiere ANTHROPIC_API_KEY)")
    al.add_argument("--source", choices=("sim", "csv"), default="sim")
    al.add_argument("--metrics"); al.add_argument("--accounts"); al.add_argument("--usage"); al.add_argument("--plans")
    al.add_argument("--mapping", help="JSON con el mapeo de columnas")
    al.add_argument("--tz", help="zona horaria a la que convertir (las horas sin zona se interpretan como UTC)")
    al.add_argument("--date-format", help='formato de fecha si es ambiguo, p. ej. "%%d/%%m/%%Y %%H:%%M"')
    al.add_argument("--out", default="outputs"); al.set_defaults(f=cmd_all)
    vc = sub.add_parser("validate-csv", help="valida y normaliza un CSV de métricas")
    vc.add_argument("--metrics", required=True); vc.add_argument("--accounts"); vc.add_argument("--usage")
    vc.add_argument("--plans"); vc.add_argument("--mapping"); vc.add_argument("--tz", help="zona horaria a la que convertir (horas sin zona = UTC)"); vc.add_argument("--out", default="outputs/data")
    vc.add_argument("--date-format", help="formato de fecha si es ambiguo")
    vc.set_defaults(f=cmd_validate_csv)
    es = sub.add_parser("export-sample", help="escribe CSV pequeños de ejemplo")
    es.add_argument("--out", default="examples"); es.set_defaults(f=cmd_export_sample)
    sc = sub.add_parser("scale", help="prueba de escala pandas vs DuckDB")
    sc.add_argument("--servers", type=int, default=300); sc.add_argument("--days", type=int, default=45)
    sc.add_argument("--engines", nargs="+", default=["pandas", "duckdb"]); sc.set_defaults(f=cmd_scale)
    rd = sub.add_parser("readme", help="vuelca los resultados en el README entre marcadores")
    rd.add_argument("--readme", default="README.md"); rd.add_argument("--out", default="outputs"); rd.set_defaults(f=cmd_readme)
    b = sub.add_parser("benchmark"); b.add_argument("--seed", type=int, default=100)
    b.add_argument("--n", type=int, default=5); b.add_argument("--budget", type=float, default=0.5)
    b.set_defaults(f=cmd_benchmark)
    rl = sub.add_parser("real"); rl.add_argument("--cache", default="data/real"); rl.set_defaults(f=cmd_real)
    s = sub.add_parser("score", help="puntúa un CSV de métricas con model.joblib")
    s.add_argument("--model", default="outputs/model.joblib",
                   help="bundle generado por tu propio pipeline (es un pickle: no cargues modelos de terceros)"); s.add_argument("--input", required=True); s.add_argument("--tz"); s.add_argument("--date-format")
    s.add_argument("--out", default="alerts.csv"); s.set_defaults(f=cmd_score)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
