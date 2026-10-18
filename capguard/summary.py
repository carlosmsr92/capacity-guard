"""Resumen en Markdown generado a partir de results.json (así el README nunca contradice a los resultados)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .analysis import DEFAULT_COSTS

START, END = "<!--RESULTS-->", "<!--/RESULTS-->"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _bench_md(x: dict) -> list[str]:
    """Media ± desviación en otros mundos simulados: la cifra robusta (una sola semilla puede engañar)."""
    b = x.get("benchmark")
    if not b:
        return []
    rows = [{"Método": r["method"],
             "Avisadas ≥10 min antes (media ± desv.)": f"{100 * r['early_rate_mean']:.0f} % ± {100 * r['early_rate_std']:.0f}",
             "Falsas alarmas / servidor·día": f"{r['false_alarms_per_server_day_mean']:.2f} ± "
                                              f"{r['false_alarms_per_server_day_std']:.2f}"}
            for r in b["rows"] if r["task"] == "B"]
    return [f"**Robustez: el mismo experimento en {len(b['seeds'])} mundos simulados distintos "
            f"(semillas {', '.join(map(str, b['seeds']))}):**", "", _md(pd.DataFrame(rows)), ""]


def _quality_line(q: dict, csv: bool) -> str:
    fails = sum(1 for ch in q["quality_before"]["checks"] if not ch["passed"])
    if csv:
        return (f"- **Calidad de datos:** el CSV falla {fails} comprobaciones antes de normalizar y reparar; "
                f"{q['rows_read']:,} filas leídas → {q['rows_final']:,} finales.")
    return (f"- **Calidad de datos:** el feed sucio falla {fails} comprobaciones y tras reparar pasa el contrato; "
            f"{q['rows_truth'] - q['rows_clean']} de {q['rows_truth']:,} filas irrecuperables; error de imputación ≤ "
            f"{100 * max(q['imputation_mae_vs_metric_std'].values()):.0f} % de la desviación típica.")


def _diagnosis_line(x: dict, k0: str) -> str:
    if not x.get("diagnosis"):
        reason = x.get("capabilities", {}).get("diagnosis_reason", "no disponible con estos datos")
        return f"- **Diagnóstico de causa:** se omite ({reason})."
    d, sh = x["diagnosis"]["by_lag"], x["diagnosis_shuffled"]["by_lag"][k0]["accuracy"]
    return (f"- **Diagnóstico de causa:** {100 * d[k0]['accuracy']:.0f} % de acierto al saltar la alarma "
            f"(predecir siempre la clase mayoritaria: {100 * d[k0]['majority_baseline']:.0f} %; con etiquetas barajadas "
            f"{100 * sh:.0f} %). Las firmas simuladas son limpias: con datos reales hay que esperar menos.")


def _customers_line(c: dict) -> str:
    offers = pd.DataFrame(c.get("offers", []))
    kinds = offers["oferta"].value_counts().to_dict() if not offers.empty else {}
    commercial = kinds.get("upgrade", 0) + kinds.get("temporal", 0)
    seg, fv, gt = c.get("segmentation", {}), c.get("forward_validation", {}), c.get("ground_truth")
    extra = []
    if seg.get("available"):
        extra.append(f"{seg['k']} segmentos ({', '.join(sorted(set(map(str, seg['cluster_names'].values()))))})")
    if gt:
        extra.append(f"ARI frente a los arquetipos simulados {gt['ari']:.2f}")
    if fv.get("available"):
        extra.append(f"precisión hacia delante de la regla {100 * fv['regla']['precision']:.0f} % "
                     f"(ingenua {100 * fv['ingenua']['precision']:.0f} %)")
    return (f"- **Clientes cerca de su límite:** {c['n_flagged']} de {c['n_accounts']} cuentas marcadas; "
            f"{commercial} borradores comerciales, todos en `pendiente_revision` (nada se envía solo); "
            f"{len(c.get('technical_accounts', []))} cuentas con causa técnica excluidas de cualquier venta"
            + (f"; {'; '.join(extra)}" if extra else "") + ". La aceptación de las ofertas es un supuesto ilustrativo.")


def results_markdown(x: dict) -> str:
    cfg = x["main"]["config"]
    B = x["main"]["tasks"]["B"]["methods"]
    rows = [{"Método": m["method"], "Saturaciones avisadas ≥10 min antes": f"{100 * m['early_rate']:.0f} %",
             "Falsas alarmas / servidor·día": f"{m['false_alarms_per_server_day']:.2f}"} for m in B.values()]
    ae = x["deep"]["B"]
    rows.append({"Método": "Autoencoder (deep learning)", "Saturaciones avisadas ≥10 min antes": f"{100 * ae['early_rate']:.0f} %",
                 "Falsas alarmas / servidor·día": f"{ae['false_alarms_per_server_day']:.2f}"})
    wf = pd.DataFrame(x["walk_forward"])
    w = wf.groupby("method", sort=False).agg(e=("early_rate", "mean"), es=("early_rate", "std"),
                                              f=("false_alarms_per_server_day", "mean")).reset_index()
    wrows = [{"Método": r.method, "Avisadas ≥10 min antes (media ± desv.)": f"{100 * r.e:.0f} % ± {100 * r.es:.0f}",
              "Falsas alarmas / servidor·día": f"{r.f:.2f}"} for r in w.itertuples()]
    c = pd.DataFrame(x["costs"])
    crow = [{"Método": r.method, "Coste en test (€)": f"{r.coste_total:,.0f}", "Ahorro vs umbral": f"{r.ahorro_vs_umbral_pct:.0f} %"}
            for r in c.itertuples()]
    q, k0 = x["ingest"], "al saltar la alarma"
    csv = x.get("source", "sim") == "csv"
    cal = x["calibration"]
    runs = [(k, v, r) for k, v in x["scale"].items() for r in v["results"] if "error" not in r]
    # las escalas grandes no se repiten en cada ejecución: vienen de una medición guardada en docs/
    srows = [{"Escala": f"{v['rows']:,} filas", "Motor": r["engine"], "Tiempo (s)": r["seconds"],
              "Memoria pico (MB)": int(r["peak_rss_mb"]), "Máquina": v.get("machine", "–"),
              "Origen": "medición previa (docs/scale_benchmark.json)" if k.startswith("full_") else "esta ejecución"}
             for k, v, r in runs]
    ds = x["main"]["dataset"]
    origin = (f"datos CSV · {ds['servers']} servidores × {ds['days']:.0f} días" if csv else
              f"semilla {cfg['seed']} · {cfg['n_servers']} servidores × {cfg['days']} días · datos simulados")
    out = [f"*Generado por `python -m capguard all` · {origin}.*", "", f"**Aviso temprano de saturación (30 min), periodo de test, presupuesto ≤ {cfg['budget_fa_per_server_day']} "
           f"falsas alarmas por servidor y día en validación:**", "", _md(pd.DataFrame(rows)), "",
           *_bench_md(x),
           "**Validación walk-forward (3 periodos futuros):**", "", _md(pd.DataFrame(wrows)), "",
           f"**Coste con supuestos ilustrativos (no avisar {DEFAULT_COSTS['c_miss']:.0f} €, falsa alarma "
           f"{DEFAULT_COSTS['c_fa']:.0f} €, avisar a tiempo {DEFAULT_COSTS['c_mitigated']:.0f} €):**", "",
           _md(pd.DataFrame(crow)), "",
           _quality_line(q, csv), _diagnosis_line(x, k0),
           f"- **Calibración isotónica:** Brier {cal['brier_raw']:.4f} → {cal['brier_calibrated']:.4f} "
           f"({'no mejora, se descarta' if cal['brier_calibrated'] >= cal['brier_raw'] else 'mejora'}).",
           "", "**Prueba de escala (4 variables de producción; pandas vs DuckDB sobre Parquet):**", "", _md(pd.DataFrame(srows))]
    if x.get("customers", {}).get("available"):
        out.extend(["", _customers_line(x["customers"])])
    return "\n".join(out)


def update_readme(readme: str | Path, results_json: str | Path) -> None:
    x = json.loads(Path(results_json).read_text())
    text = Path(readme).read_text()
    block = f"{START}\n{results_markdown(x)}\n{END}"
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    if new == text and START not in text:
        raise SystemExit("README sin marcadores <!--RESULTS-->")
    Path(readme).write_text(new)
