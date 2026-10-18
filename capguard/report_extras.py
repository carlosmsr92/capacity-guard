"""Secciones adicionales del informe: calidad de datos, escala, negocio, validación, DL, diagnóstico e IA."""
from __future__ import annotations

import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import DEFAULT_COSTS
from .report import _b64

FEATURE_ES = {"headroom_min": "margen hasta el límite más cercano", "io_d3": "tendencia del iowait",
              "ram_slope": "pendiente de la RAM", "cpu_unexplained": "CPU no explicada por el tráfico"}


def _feature_es(name: str) -> str:
    if name in FEATURE_ES:
        return FEATURE_ES[name]
    if name.startswith("z_"):
        return f"{name[2:]} fuera de lo normal para esa hora"
    return name


def _pct(v) -> str:
    return "–" if v is None or v != v else f"{100 * v:.0f} %"


def _tbl(df: pd.DataFrame, cls: str = "") -> str:
    return df.to_html(index=False, border=0, classes=cls, escape=True, na_rep="–")


def _fig_hours(eda: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9, 2.9))
    ax[0].bar(range(24), eda["by_hour"], color="#0072b2")
    ax[0].set_title("Inicio de incidentes por hora del día"); ax[0].set_xlabel("hora")
    k = pd.DataFrame(eda["by_kind"]).sort_values("min_saturados")
    ax[1].barh(k["kind"], k["min_saturados"] / 60, color="#c0392b")
    ax[1].set_title("Horas de saturación por tipo de incidente")
    fig.tight_layout()
    return _b64(fig)


def _fig_importance(imp: list[dict]) -> str:
    d = pd.DataFrame(imp).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color="#0072b2")
    ax.set_xlabel("Caída de precisión media (AP) al barajar la variable")
    return _b64(fig)


def _fig_reliability(cal: dict) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    r = pd.DataFrame(cal["reliability_raw"])
    ax.plot(r["pred_mean"], r["observed"], "o-", color="#0072b2")
    ax.set_xlabel("probabilidad predicha"); ax.set_ylabel("frecuencia observada")
    ax.set_title("Fiabilidad de las probabilidades (test)", fontsize=9)
    return _b64(fig)


def _scale_text(sc: dict) -> str:
    runs = [(d, r) for d in sc.values() for r in d.get("results", []) if "error" not in r]
    common = max({d["rows"] for d, r in runs if r["engine"] == "pandas"} & {d["rows"] for d, r in runs if r["engine"] == "duckdb"})
    p = next(r for d, r in runs if d["rows"] == common and r["engine"] == "pandas")
    k = next(r for d, r in runs if d["rows"] == common and r["engine"] == "duckdb")
    biggest = max(runs, key=lambda t: t[0]["rows"])
    return (f"<p>Con {common:,} filas (el mayor tamaño comparado): pandas {p['seconds']} s y {int(p['peak_rss_mb'])} MB; "
            f"DuckDB {k['seconds']} s y {int(k['peak_rss_mb'])} MB. El mayor tamaño probado fue {biggest[0]['rows']:,} filas "
            f"({biggest[1]['engine']}, {biggest[1]['seconds']} s). No se ejecutó pandas con ese tamaño. Más allá de una "
            f"máquina, el mismo SQL se portaría a Spark o a un almacén en la nube; eso no se ha probado aquí.</p>")


def _sens_text(sens: pd.DataFrame) -> str:
    wins = pd.Series(sens.to_numpy().ravel()).value_counts()
    parts = ", ".join(f"{m} en {n} de {wins.sum()}" for m, n in wins.items())
    tail = (" Cuando las falsas alarmas son caras frente a no avisar, gana el umbral estático."
            if "Umbral estático" in wins.index else "")
    return f"Método más barato en la cuadrícula de costes: {parts} combinaciones.{tail}"


def _walk_text(wf: pd.DataFrame, budget: float) -> str:
    best = wf.groupby("fold").apply(lambda g: g.loc[g["early_rate"].idxmax(), "key"], include_groups=False)
    g = wf[wf["key"] == "gbm"]
    over = int((g["false_alarms_per_server_day"] > budget).sum())
    a = (f"El pronosticador GBM es el mejor en aviso temprano en {int((best == 'gbm').sum())} de {len(best)} periodos. ")
    b = (f"Sus falsas alarmas superan el presupuesto de {budget} (con el que se calibró en validación) en {over} de {len(g)} "
         f"periodos: el presupuesto no se transfiere exactamente fuera de muestra.")
    return f"<p>{a}{b}</p>"


def _cal_text(cal: dict) -> str:
    r = pd.DataFrame(cal["reliability_raw"])
    r = r[r["n"] >= 30]
    worst = r.loc[(r["observed"] - r["pred_mean"]).abs().idxmax()]
    verdict = "no mejora" if cal["brier_calibrated"] >= cal["brier_raw"] else "mejora"
    kept = "se mantiene el modelo sin calibrar" if verdict == "no mejora" else "conviene usar el modelo calibrado"
    return (f"<p>Calibrar con regresión isotónica <b>{verdict}</b> (Brier {cal['brier_raw']:.4f} → {cal['brier_calibrated']:.4f}; "
            f"AUC {cal['auc_raw']:.3f} → {cal['auc_calibrated']:.3f}), así que {kept}. Mayor desviación de las probabilidades "
            f"en el tramo {worst['bin']}: predice ~{100 * worst['pred_mean']:.0f} % y se observa ~{100 * worst['observed']:.0f} %.</p>")


def _deep_text(d: dict, gbm: dict, z: dict) -> str:
    ae = d["B"]["early_rate"]
    rel = "no supera" if ae <= max(gbm["early_rate"], z["early_rate"]) else "supera"
    return (f"<p>Autoencoder: {100 * ae:.0f} % de saturaciones avisadas ≥10 min antes, frente a {100 * gbm['early_rate']:.0f} % del GBM y "
            f"{100 * z['early_rate']:.0f} % del z-score: <b>{rel}</b> a los otros métodos en aviso temprano. Se incluye como comparación "
            f"honesta, no como mejora. Implementado con scikit-learn (MLP), no con PyTorch.</p>")


def extra_sections(x: dict) -> str:
    img = lambda b: f'<img src="data:image/png;base64,{b}" style="max-width:100%">'
    out = []

    out.append(_origin_section(x))
    q = x["ingest"]
    qb = {c["name"]: c["violations"] for c in q["quality_before"]["checks"] if not c["passed"]}
    qa = {c["name"]: c["violations"] for c in q["quality_after"]["checks"] if not c["passed"]}
    if x.get("source", "sim") == "csv":
        out.append(_csv_ingest_section(q, qb, qa))
    else:
        out.append(_sim_ingest_section(q, qb, qa))

    sc = x.get("scale")
    if sc:
        rows = []
        for d in sc.values():
            for r in d.get("results", []):
                if "error" not in r:
                    rows.append({"escala": f"{d['servers']} servidores × {d['days']} días", "filas": f"{r['rows']:,}",
                                 "motor": r["engine"], "segundos": r["seconds"], "memoria pico (MB)": int(r["peak_rss_mb"]),
                                 "máquina": d.get("machine", "–")})
        out.append(f"""<h2>Big data · prueba de escala</h2>
<p>Las mismas 4 variables (baseline estacional por mediana/MAD, ventanas móviles y pendientes) con pandas y con DuckDB sobre
Parquet. Cada motor corre en un proceso aparte para medir la memoria pico. Un test comprueba que ambos devuelven
exactamente los mismos números.</p>{_tbl(pd.DataFrame(rows))}
{_scale_text(sc)}""")

    e, c = x["eda"], pd.DataFrame(x["costs"])
    incidents_html = _incident_eda(e) if "by_kind" in e else _skipped("incident_eda", x)
    sens = pd.DataFrame(x["sensitivity"]).pivot(index="c_miss", columns="c_fa", values="mas_barato")
    sens.index = [f"no avisar = {i} €" for i in sens.index]
    sens.columns = [f"falsa alarma = {i} €" for i in sens.columns]
    best = c.loc[c["coste_total"].idxmin()]
    out.append(f"""<h2>Analista de datos y BI · de las métricas a decisiones</h2>{incidents_html}
<h3>Modelo de coste (supuestos ilustrativos)</h3>
<p>Supuestos: avisar a tiempo cuesta {DEFAULT_COSTS['c_mitigated']:.0f} €, no avisar {DEFAULT_COSTS['c_miss']:.0f} €, una
falsa alarma {DEFAULT_COSTS['c_fa']:.0f} € (tiempo de guardia). Son
<b>valores de ejemplo</b>: hay que sustituirlos por los reales de la empresa (créditos de SLA, coste de guardia).</p>
{_tbl(c[['method', 'saturaciones', 'avisadas_a_tiempo', 'no_avisadas', 'falsas_alarmas', 'coste_total',
         'coste_por_servidor_mes', 'ahorro_vs_umbral_pct']].rename(columns={'method': 'método'}))}
<p>Con estos supuestos el más barato es <b>{best['method']}</b>. Sensibilidad (qué método sale más barato según el coste
relativo): {_tbl(sens.reset_index().rename(columns={'index': ''}))}
{_sens_text(sens)}</p>""")

    wf = pd.DataFrame(x["walk_forward"])
    agg = wf.groupby("method", sort=False).agg(
        early=("early_rate", "mean"), early_sd=("early_rate", "std"),
        fa=("false_alarms_per_server_day", "mean"), fa_sd=("false_alarms_per_server_day", "std")).reset_index()
    agg["avisadas ≥10 min antes"] = (100 * agg["early"]).round(0).astype(int).astype(str) + " % ± " + (100 * agg["early_sd"]).round(0).astype(int).astype(str)
    agg["falsas alarmas / servidor·día"] = agg["fa"].round(2).astype(str) + " ± " + agg["fa_sd"].round(2).astype(str)
    cal = x["calibration"]
    out.append(f"""<h2>Machine learning · validación rigurosa</h2>
<h3>Walk-forward (3 periodos futuros distintos, reentrenando con ventana creciente)</h3>
{_tbl(agg[['method', 'avisadas ≥10 min antes', 'falsas alarmas / servidor·día']].rename(columns={'method': 'método'}))}
{_walk_text(wf, x['main']['config']['budget_fa_per_server_day'])}
<h3>Calibración e interpretabilidad</h3>{_cal_text(cal)}
{img(_fig_reliability(cal))} {img(_fig_importance(x['importance']))}
<p>Las variables más influyentes (por permutación, en test): {', '.join(f"{i['feature']} ({_feature_es(i['feature'])})"
                                                                     for i in x['importance'][:3])}.
Conviene comprobar con el equipo que tengan sentido físico para el problema antes de confiar en el modelo.</p>""")

    d = x["deep"]
    rows = []
    for t in ("A", "B"):
        m = d[t]
        if not m["n_events"]:
            continue  # sin eventos etiquetados la tarea no se puede evaluar
        rows.append({"tarea": t, "detectadas": f"{100 * m['recall']:.0f} %", "≥10 min antes": f"{100 * m['early_rate']:.0f} %",
                     "falsas alarmas / servidor·día": f"{m['false_alarms_per_server_day']:.2f}"})
    out.append(f"""<h2>Deep learning · autoencoder</h2>
<p>Red neuronal densa (64-16-64) que reconstruye ventanas de 1 hora de métricas normalizadas; lo que reconstruye mal es
anómalo. Entrenada sin etiquetas, calibrada igual que el resto de métodos.</p>{_tbl(pd.DataFrame(rows))}
{_deep_text(d, x['main']['tasks']['B']['methods']['gbm'], x['main']['tasks']['B']['methods']['zscore'])}""")

    dg = x["diagnosis"]
    if dg is None:
        out.append(f"<h2>Diagnóstico de causa (ML multiclase)</h2>{_skipped('diagnosis', x)}")
    else:
        out.append(_diagnosis_section(dg, x))
    out.append(_explanations_section(x["explanations"][:3]))
    out.append(_customers_section(x.get("customers", {})))
    return "".join(out)


TECHNICAL_SOURCE = {"incidentes_etiquetados": "incidentes etiquetados y atribuidos a la cuenta",
                    "uso_por_encima_del_limite": "uso por encima del límite del plan (sin etiquetas de incidentes)",
                    "sin_datos": "no hay datos para detectarlas"}


def _customers_section(customers: dict) -> str:
    title = "<h2>Clientes cerca de su límite y ofertas</h2>"
    if not customers.get("available"):
        reason = customers.get("reason", "no disponible")
        return f'{title}<div class="note"><b>Se omite:</b> {html.escape(reason)}.</div>'
    offers = pd.DataFrame(customers.get("offers", []))
    seg, fv, gt = customers.get("segmentation", {}), customers.get("forward_validation", {}), customers.get("ground_truth")
    by_kind = offers["oferta"].value_counts().rename_axis("oferta").reset_index(name="borradores") if not offers.empty \
        else pd.DataFrame()
    parts = [f"""{title}
<p>Todo se mide en <b>utilización</b> (uso / límite del plan). Regla: una cuenta está «cerca de su límite» si en los
últimos 14 días pasó ≥ 5 días por encima del 85 % de algún límite o ≥ 3 días limitada. Marca
<b>{customers['n_flagged']}</b> de {customers['n_accounts']} cuentas. Sensibilidad de la regla:</p>
{_tbl(pd.DataFrame(customers.get('rule_sensitivity', [])))}"""]
    if fv.get("available"):
        rows = [{"regla": name, "marcadas": v["marcadas"], "precisión": _pct(v["precision"]), "cobertura": _pct(v["cobertura"])}
                for name, v in (("regla propuesta", fv["regla"]), ("ingenua (≥1 día limitada)", fv["ingenua"]))]
        parts.append(f"""<p><b>Validación hacia delante</b> (marcar con datos hasta el {fv['corte']} y comprobar quién sigue
cerca del límite en las 2 semanas siguientes; {fv['positivos_futuros']} cuentas lo están):</p>{_tbl(pd.DataFrame(rows))}""")
    if seg.get("available"):
        cent = pd.DataFrame(seg["centroids"])
        cent.insert(0, "nombre", cent["cluster_id"].map(lambda i: seg["cluster_names"].get(i, seg["cluster_names"].get(str(i)))))
        parts.append(f"""<p><b>Segmentación</b> (KMeans, k = {seg['k']} elegido por silueta entre los k estables; estabilidad
entre semillas {seg['stability']:.2f}; fidelidad del árbol sustituto {seg['tree_fidelity']:.2f}). Los nombres salen de
reglas fijas sobre cada centroide, no de un modelo:</p>{_tbl(cent.round(2))}
<pre style="font-size:12px">{html.escape(seg['tree_rules'])}</pre>""")
    else:
        parts.append(f'<div class="note"><b>Sin segmentación:</b> {html.escape(str(seg.get("reason", "")))}.</div>')
    if gt:
        parts.append(f"<p>Con datos simulados se conoce el arquetipo real: ARI clusters vs. arquetipos = <b>{gt['ari']:.2f}</b>.</p>")
    value = customers.get("offer_value", {})
    sens = pd.DataFrame(value.get("sensitivity", [])).rename(columns={"acceptance": "aceptación supuesta",
                                                                      "expected_monthly_value": "€/mes adicionales"})
    parts.append(f"""<p>Cuentas con causa técnica ({len(customers.get('technical_accounts', []))}, detectadas por
{TECHNICAL_SOURCE.get(customers.get('technical_source'), customers.get('technical_source'))}): nunca reciben una
venta, solo un aviso a la guardia. Todas las ofertas son <b>borradores en estado pendiente_revision</b>: nada se envía
sin que una persona lo apruebe.</p>{_tbl(by_kind)}
<p><b>Valor (supuesto ilustrativo):</b> {html.escape(value.get('note', ''))}</p>{_tbl(sens)}""")
    if not offers.empty:
        parts.append(_tbl(offers[["account_id", "cluster", "oferta", "plan_actual", "plan_propuesto", "precio_actual",
                                  "precio_propuesto", "estado"]].head(15)))
    return "".join(parts)


def _diagnosis_section(dg: dict, x: dict) -> str:
    lag_rows = [{"momento": k, "eventos": v["n"], "acierto": f"{100 * v['accuracy']:.0f} %",
                 "azar (clase mayoritaria)": f"{100 * v['majority_baseline']:.0f} %"} for k, v in dg["by_lag"].items()]
    return f"""<h2>Diagnóstico de causa (ML multiclase)</h2>
<p>Desde el momento en que salta la alarma del detector no supervisado se clasifica la causa: vecino ruidoso, fuga de
memoria, fuerza bruta, saturación de disco o pico legítimo. Cobertura: {100 * dg['coverage']:.0f} % de los eventos de test
dispararon alarma.</p>{_tbl(pd.DataFrame(lag_rows))}
<p><b>Comprobación de cordura:</b> con las etiquetas de entrenamiento barajadas el acierto cae a
{100 * x['diagnosis_shuffled']['by_lag']['al saltar la alarma']['accuracy']:.0f} % (azar), por lo que no hay fuga de
información. Aun así, las firmas simuladas son muy limpias: con datos reales es de esperar un acierto bastante menor.</p>"""


def _explanations_section(ex: list) -> str:
    llm = any(e.get("source") == "llm" for e in ex)
    mode = "redactados por el LLM y validados por el guardarraíl" if llm else "modo plantilla"
    note = ("" if llm else "<p>En esta ejecución no se usó el LLM (sin clave de API o sin <code>--llm</code>); su ruta "
            "está verificada con un cliente simulado, incluidos casos de cifra inventada y de caída del servicio.</p>")
    return f"""<h2>IA · de la alarma a una explicación para la guardia</h2>
<p>Cada alarma se acompaña de un paquete de hechos medidos y de una explicación en español. Con una clave de API puede
redactarla un LLM, pero solo a partir de esos hechos: un guardarraíl rechaza cualquier cifra que no esté en la
evidencia y, si falla, se usa la plantilla determinista. Ejemplos ({mode}):</p>
{''.join(f'<div class="note" style="background:#f5f7fa;border-color:#0072b2">{html.escape(e["text"])}</div>' for e in ex)}
{note}"""


CAPABILITY_NAMES = {"task_a": "Tarea A (detección de incidentes)", "diagnosis": "Diagnóstico de causa",
                    "incident_eda": "Análisis por tipo de incidente", "customers": "Clientes cerca de su límite",
                    "ground_truth_archetypes": "Comparación de segmentos con la verdad"}


def _skipped(name: str, x: dict) -> str:
    reason = x.get("capabilities", {}).get(f"{name}_reason", "no disponible con estos datos")
    return f'<div class="note"><b>Se omite:</b> {html.escape(reason)}.</div>'


def _origin_section(x: dict) -> str:
    cap = x.get("capabilities", {})
    source = {"sim": "simulación (verdad-terreno conocida)", "csv": "CSV aportado por el usuario"}.get(
        x.get("source", "sim"), x.get("source", "sim"))
    rows = [{"capacidad": label, "disponible": "sí" if cap.get(k) else "no", "motivo": cap.get(f"{k}_reason", "")}
            for k, label in CAPABILITY_NAMES.items() if k in cap]
    return f"""<h2>Ingesta y origen de los datos</h2>
<p>Origen: <b>{source}</b>. Lo que no se puede calcular con estos datos se omite con un aviso; no se inventan
etiquetas.</p>{_tbl(pd.DataFrame(rows)) if rows else ''}"""


def _quality_table(qb: dict, qa: dict) -> str:
    return _tbl(pd.DataFrame({'comprobación': list(qb), 'fallos antes': list(qb.values()),
                              'fallos después': [qa.get(k, 0) for k in qb]}))


def _sim_ingest_section(q: dict, qb: dict, qa: dict) -> str:
    mae = pd.Series(q["imputation_mae_vs_metric_std"]).mul(100).round(1)
    return f"""<h2>Ingeniería de datos · calidad y reparación</h2>
<p>Se simula un feed sucio (duplicados, nulos, valores imposibles, huecos cortos, orden alterado), se valida contra un
contrato de datos, se repara de forma trazable y se mide el error de la reparación contra la verdad que se ocultó.</p>
{_quality_table(qb, qa)}
<p>Tras reparar quedan <b>{q['rows_truth'] - q['rows_clean']}</b> filas irrecuperables de {q['rows_truth']:,} (se dejan
como hueco, no se inventan). Error de las {q['imputed_rows']} filas imputadas, en % de la desviación típica de cada
métrica: {', '.join(f'{k} {v}%' for k, v in mae.items())}. Almacenamiento: Parquet + zstd, consultado con DuckDB.</p>"""


def _csv_ingest_section(q: dict, qb: dict, qa: dict) -> str:
    conv = ", ".join(q["unit_conversions"]) or "ninguna"
    ignored = ", ".join(q["ignored_columns"]) or "ninguna"
    table = _quality_table(qb, qa) if qb else "<p>El CSV cumple el contrato de datos sin necesidad de reparar.</p>"
    return f"""<h2>Ingeniería de datos · calidad y reparación</h2>
<p>El CSV se normaliza al formato canónico (cabeceras, alias, tipos, unidades y rejilla de 5 min) y se valida contra el
mismo contrato de datos que la simulación.</p>{table}
<p>Filas leídas: {q['rows_read']:,} → finales: {q['rows_final']:,}. Resolución detectada: {q['frequency'] or '–'}.
Conversiones de unidades: {html.escape(conv)}. Columnas ignoradas: {html.escape(ignored)}. Valores no numéricos
convertidos en hueco: {sum(q['non_numeric'].values())}. Timestamps ilegibles descartados:
{q.get('bad_timestamps_dropped', 0)}. Almacenamiento: Parquet + zstd, consultado con DuckDB.</p>"""


def _incident_eda(e: dict) -> str:
    kinds = pd.DataFrame(e["by_kind"]).rename(columns={
        "kind": "tipo", "incidentes": "incidentes", "duracion_media_min": "duración media (min)",
        "pct_que_satura": "% que satura", "min_saturados": "min saturados", "pct_min_saturados": "% del tiempo saturado"})
    kinds["% que satura"] = (kinds["% que satura"] * 100).round(0)
    kinds["% del tiempo saturado"] = (kinds["% del tiempo saturado"] * 100).round(0)
    return f"""<img src="data:image/png;base64,{_fig_hours(e)}" style="max-width:100%">
{_tbl(kinds)}<p>Hay unos {e['incidents_per_server_month']:.0f} incidentes por servidor y mes (simulado). El tipo que más
tiempo de saturación causa es <b>{html.escape(str(e['worst_kind_for_saturation']))}</b>.</p>"""
