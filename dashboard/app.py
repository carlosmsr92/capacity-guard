"""Dashboard de BI de capacity-guard (Streamlit). Ejecutar: streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from capguard import analysis  # noqa: E402
from capguard.simulator import SAT_LIMITS  # noqa: E402

OUT = Path(os.environ.get("CAPGUARD_OUT", ROOT / "outputs"))
DATA = OUT / "data"
METHODS = {"threshold": "Umbral estático", "zscore": "Z-score estacional", "iforest": "Isolation Forest", "gbm": "Pronosticador GBM"}

st.set_page_config(page_title="capacity-guard", page_icon="🛡️", layout="wide")


def _stamp() -> int:
    f = OUT / "results.json"
    return f.stat().st_mtime_ns if f.exists() else 0


@st.cache_data(show_spinner=False)
def load_results(stamp: int) -> dict:
    return json.loads((OUT / "results.json").read_text())


def _md(text) -> str:
    """Escapa markdown en textos con datos del CSV (nombres de servidor, cuentas, etiquetas): solo se muestran."""
    return re.sub(r"([\\`*_{}\[\]()<>#+!|~])", r"\\\1", str(text))


def _sql_path(path: Path) -> str:
    """Ruta como literal SQL (las comillas se duplican) para read_parquet."""
    return path.as_posix().replace("'", "''")


def _db() -> duckdb.DuckDBPyConnection:
    """Conexión corta y con pocos hilos: Streamlit ejecuta el script en su propio hilo y no conviene dejar
    conexiones abiertas (con un hilo por core cada una) a merced del recolector de basura."""
    return duckdb.connect(config={"threads": 4})


@st.cache_data(show_spinner=False)
def servers(stamp: int) -> list[str]:
    with _db() as con:
        return [r[0] for r in con.execute(f"SELECT DISTINCT server FROM read_parquet('{_sql_path(DATA / 'metrics.parquet')}') ORDER BY 1").fetchall()]


@st.cache_data(show_spinner=False)
def server_frame(stamp: int, server: str, start: str, end: str) -> pd.DataFrame:
    m, s = _sql_path(DATA / "metrics.parquet"), _sql_path(DATA / "scores.parquet")
    with _db() as con:
        return con.execute(f"""SELECT m.timestamp, m.cpu, m.ram, m.iowait, m.event, m.sat_now, s.risk_gbm, s.alert_B_gbm, s.split
        FROM read_parquet('{m}') m JOIN read_parquet('{s}') s USING (server, timestamp)
        WHERE server = ? AND m.timestamp BETWEEN ? AND ? ORDER BY m.timestamp""", [server, start, end]).df()


@st.cache_data(show_spinner=False)
def bounds(stamp: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    with _db() as con:
        a, b = con.execute(f"SELECT min(timestamp), max(timestamp) FROM read_parquet('{_sql_path(DATA / 'metrics.parquet')}')").fetchone()
    return pd.Timestamp(a), pd.Timestamp(b)


def segments(df: pd.DataFrame, col: str):
    on = (df[col] != "") if df[col].dtype == object or str(df[col].dtype).startswith("str") else df[col].astype(bool)
    grp = (on != on.shift()).cumsum()
    for _, g in df[on].groupby(grp[on]):
        yield g["timestamp"].iloc[0], g["timestamp"].iloc[-1], (g[col].iloc[0] if col == "event" else "")


if not all((OUT / f).exists() for f in ("results.json", "data/metrics.parquet", "data/scores.parquet")):
    st.error("Faltan los resultados o sus datos (no se versionan en git). Ejecuta primero: `python -m capguard all`")
    st.stop()

R = load_results(_stamp())
main_b = R["main"]["tasks"]["B"]["methods"]
st.title("🛡️ capacity-guard")
CSV = R.get("source", "sim") == "csv"
CAP = R.get("capabilities", {})
st.caption("Detección de anomalías y aviso temprano de saturación en servidores de hosting · "
           + ("datos REALES (CSV)" if CSV else "datos SIMULADOS"))
skipped = [CAP[f"{k}_reason"] for k, v in CAP.items() if v is False and f"{k}_reason" in CAP]
if skipped:
    st.warning("Partes omitidas con estos datos: " + " · ".join(_md(s) for s in skipped))
tabs = st.tabs(["Resumen", "Servidores", "Negocio", "Calidad de datos", "Modelo", "Clientes", "Alertas explicadas"])

with tabs[0]:
    g = main_b["gbm"]
    c = st.columns(5)
    c[0].metric("Muestras", f"{R['ingest']['rows_final' if CSV else 'rows_clean']:,}")
    c[1].metric("Saturaciones en test", g["n_events"])
    c[2].metric("Avisadas ≥10 min antes (GBM)", f"{100 * g['early_rate']:.0f} %")
    c[3].metric("Falsas alarmas/servidor·día", f"{g['false_alarms_per_server_day']:.2f}")
    c[4].metric("AUC del pronosticador", f"{R['main']['gbm_sample_metrics']['roc_auc']:.2f}")
    rows = [{"Método": m["method"], "Detectadas": f"{100 * m['recall']:.0f} %", "Con ≥10 min de antelación": f"{100 * m['early_rate']:.0f} %",
             "Antelación mediana (min)": m["median_lead_min"], "Falsas alarmas / servidor·día": round(m["false_alarms_per_server_day"], 2)}
            for m in main_b.values()]
    st.subheader("Aviso temprano de saturación (periodo de test)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info("Cada método se calibra en validación para no superar un presupuesto de falsas alarmas; el umbral estático se deja "
            "como lo configuraría una monitorización clásica.")

with tabs[1]:
    t0, t1 = bounds(_stamp())
    c1, c2 = st.columns([1, 3])
    server = c1.selectbox("Servidor", servers(_stamp()), key="server")
    only_test = c1.checkbox("Solo periodo de test", value=True, key="only_test")
    default_start = max(t0, t1 - pd.Timedelta(days=11)) if only_test else t0
    rng = c2.slider("Rango de fechas", t0.to_pydatetime(), t1.to_pydatetime(),
                    (default_start.to_pydatetime(), t1.to_pydatetime()), key="range")
    d = server_frame(_stamp(), server, str(rng[0]), str(rng[1]))
    if d.empty:
        st.warning("Sin datos en ese rango.")
    else:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=("CPU %", "RAM %", "iowait %", "Riesgo de saturar en 30 min"))
        for i, (col, lim) in enumerate((("cpu", SAT_LIMITS["cpu"]), ("ram", SAT_LIMITS["ram"]), ("iowait", SAT_LIMITS["iowait"])), 1):
            fig.add_trace(go.Scatter(x=d["timestamp"], y=d[col], mode="lines", line=dict(width=1.2, color="#222"), showlegend=False), row=i, col=1)
            fig.add_hline(y=lim, line=dict(color="#c0392b", dash="dash", width=1), row=i, col=1)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["risk_gbm"], mode="lines", line=dict(color="#0072b2", width=1.2), showlegend=False), row=4, col=1)
        al = d[d["alert_B_gbm"]]
        fig.add_trace(go.Scatter(x=al["timestamp"], y=al["risk_gbm"], mode="markers", marker=dict(color="#e69f00", size=5),
                                 name="alarma", showlegend=True), row=4, col=1)
        for a, b, kind in segments(d, "event"):
            fig.add_vrect(x0=a, x1=b, fillcolor="#999", opacity=0.12, line_width=0, annotation_text=kind, annotation_font_size=9)
        for a, b, _ in segments(d, "sat_now"):
            fig.add_vrect(x0=a, x1=b, fillcolor="#c0392b", opacity=0.20, line_width=0)
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Gris = evento en curso (con su causa) · rojo = saturación · naranja = alarma del pronosticador. "
                   "Fuera del periodo de test el modelo ha visto esos datos al entrenar.")

with tabs[2]:
    e = R["eda"]
    if "by_kind" in e:
        c = st.columns(3)
        c[0].metric("Incidentes (simulados)", e["n_incidents"])
        c[1].metric("Por servidor y mes", f"{e['incidents_per_server_month']:.0f}")
        c[2].metric("Tipo con más tiempo saturado", e["worst_kind_for_saturation"])
        kinds = pd.DataFrame(e["by_kind"])
        col1, col2 = st.columns(2)
        col1.bar_chart(pd.Series(e["by_hour"], name="incidentes"), height=260)
        col1.caption("Inicio de incidentes por hora del día")
        col2.bar_chart(kinds.set_index("kind")["min_saturados"] / 60, height=260)
        col2.caption("Horas de saturación por tipo de incidente")
    else:
        st.info(CAP.get("incident_eda_reason", "Sin etiquetas de incidentes."))
    st.subheader("Modelo de coste (editable)")
    st.caption("Valores de ejemplo: sustitúyelos por los de la empresa (créditos de SLA, coste de guardia).")
    s1, s2, s3 = st.columns(3)
    c_miss = s1.slider("Coste de NO avisar a tiempo (€)", 10, 2000, 200, 10, key="c_miss")
    c_fa = s2.slider("Coste de una falsa alarma (€)", 1, 300, 15, 1, key="c_fa")
    c_mit = s3.slider("Coste de avisar a tiempo (€)", 0, 200, 20, 5, key="c_mit")
    ct = analysis.cost_table(main_b, {"c_miss": c_miss, "c_fa": c_fa, "c_mitigated": c_mit})
    best = ct.loc[ct["coste_total"].idxmin(), "method"]
    st.dataframe(ct[["method", "avisadas_a_tiempo", "no_avisadas", "falsas_alarmas", "coste_total", "coste_por_servidor_mes", "ahorro_vs_umbral_pct"]]
                 .rename(columns={"method": "Método"}), hide_index=True, width="stretch")
    st.success(f"Con estos costes, el método más barato es: **{best}**")

with tabs[3]:
    q = R["ingest"]
    if CSV:
        st.write(f"CSV: {q['rows_read']:,} filas leídas → {q['rows_final']:,} finales · resolución {_md(q['frequency'])} · "
                 f"conversiones: {_md(', '.join(q['unit_conversions']) or 'ninguna')} · "
                 f"columnas ignoradas: {_md(', '.join(q['ignored_columns']) or 'ninguna')}.")
    else:
        st.write(f"Feed sucio: {q['rows_dirty']:,} filas → reparado: {q['rows_clean']:,} (verdad: {q['rows_truth']:,}). "
                 f"Filas imputadas: {q['imputed_rows']}.")
    names = [c["name"] for c in q["quality_before"]["checks"]]
    before = {c["name"]: c["violations"] for c in q["quality_before"]["checks"]}
    after = {c["name"]: c["violations"] for c in q["quality_after"]["checks"]}
    st.dataframe(pd.DataFrame({"comprobación": names, "antes": [before[n] for n in names], "después": [after.get(n, 0) for n in names]}),
                 hide_index=True, width="stretch")
    if not CSV:
        st.write("Error de la imputación (% de la desviación típica de cada métrica):")
        st.dataframe(pd.Series(q["imputation_mae_vs_metric_std"]).mul(100).round(1).rename("%").to_frame().T,
                     width="stretch")

with tabs[4]:
    st.subheader("Walk-forward (3 periodos futuros)")
    wf = pd.DataFrame(R["walk_forward"])
    st.dataframe(wf.pivot_table(index="method", columns="test_days", values="early_rate").round(2), width="stretch")
    st.caption("Fracción de saturaciones avisadas ≥10 min antes, por periodo de test.")
    c1, c2 = st.columns(2)
    imp = pd.DataFrame(R["importance"]).set_index("feature")["importance"]
    c1.write("Importancia de variables (permutación)")
    c1.bar_chart(imp, height=300)
    ae = R["deep"]["B"]
    c2.write("Autoencoder (deep learning), tarea B")
    c2.json({"detectadas": round(ae["recall"], 2), "avisadas_>=10min": round(ae["early_rate"], 2),
             "falsas_alarmas_servidor_dia": round(ae["false_alarms_per_server_day"], 2)})
    st.subheader("Diagnóstico de causa")
    dg = R["diagnosis"] or {"by_lag": {}}
    if not R["diagnosis"]:
        st.info(CAP.get("diagnosis_reason", "Diagnóstico no disponible."))
    for lag, v in dg["by_lag"].items():
        st.write(f"**{lag}** · {v['n']} eventos · acierto {100 * v['accuracy']:.0f} % (clase mayoritaria {100 * v['majority_baseline']:.0f} %)")
        st.dataframe(pd.DataFrame(v["confusion"], index=[f"real: {l}" for l in v["labels"]], columns=v["labels"]), width="stretch")

with tabs[5]:
    customers = R.get("customers", {})
    if not customers.get("available"):
        st.info(_md(customers.get("reason", CAP.get("customers_reason", "Clientes no disponibles con esta fuente."))))
    else:
        feats = pd.DataFrame(customers.get("features", []))
        offers = pd.DataFrame(customers.get("offers", []))
        m = st.columns(4)
        m[0].metric("Cuentas", customers["n_accounts"])
        m[1].metric("Cerca de su límite", customers["n_flagged"])
        m[2].metric("Con causa técnica (sin venta)", len(customers.get("technical_accounts", [])))
        m[3].metric("Borradores comerciales", int(offers["oferta"].isin(["upgrade", "temporal"]).sum()) if not offers.empty else 0)
        st.warning("Todas las ofertas son borradores en estado `pendiente_revision`: nada se envía sin aprobación humana.")
        c1, c2 = st.columns(2)
        cluster_values = ["Todos"] + sorted(offers["cluster"].dropna().unique().tolist()) if not offers.empty else ["Todos"]
        offer_values = ["Todos"] + sorted(offers["oferta"].dropna().unique().tolist()) if not offers.empty else ["Todos"]
        cluster = c1.selectbox("Cluster", cluster_values, key="customer_cluster")
        offer = c2.selectbox("Oferta", offer_values, key="customer_offer")
        if cluster != "Todos":
            offers = offers[offers["cluster"] == cluster]
        if offer != "Todos":
            offers = offers[offers["oferta"] == offer]
        cols = ["account_id", "cluster", "oferta", "plan_actual", "plan_propuesto", "precio_actual", "precio_propuesto",
                "estado", "motivo"]
        st.dataframe(offers[[c for c in cols if c in offers]], hide_index=True, width="stretch")
        value = customers.get("offer_value", {})
        if value.get("sensitivity"):
            st.caption("Valor mensual adicional con distintas tasas de aceptación (SUPUESTO ilustrativo): " + " · ".join(
                f"{100 * s['acceptance']:.0f} % → {s['expected_monthly_value']:,.0f} €" for s in value["sensitivity"]))
        if not offers.empty:
            pick = st.selectbox("Cuenta", offers["account_id"].tolist(), key="customer_account")
            usage_path = DATA / "account_usage_hourly.parquet"
            if usage_path.exists():
                from capguard.accounts import LIMIT_FIELD, PLANS, RESOURCES
                u = pd.read_parquet(usage_path, filters=[("account_id", "==", pick)])
                plan = offers.set_index("account_id").loc[pick, "plan_actual"]
                if plan in PLANS and not u.empty:
                    pct = pd.DataFrame({r: 100 * u[r].astype(float) / getattr(PLANS[plan], LIMIT_FIELD[r])
                                        for r in RESOURCES}).set_index(u["timestamp"])
                    st.line_chart(pct, height=240)
                    st.caption("% del límite del plan por hora y recurso (100 % = limitado).")
            st.write(_md(offers.set_index("account_id").loc[pick, "texto"]))
            with st.expander("Evidencia usada para redactar la oferta"):
                st.json(offers.set_index("account_id").loc[pick, "evidencia"])

with tabs[6]:
    exp = json.loads((DATA / "alerts_explained.json").read_text())
    if not exp:
        st.warning("No hay alarmas explicadas.")
    else:
        labels = [f"{e['server']} · {e['timestamp']}" for e in exp]
        pick = st.selectbox("Alarma", range(len(exp)), format_func=lambda i: labels[i], key="alert_pick")
        st.write(_md(exp[pick]["text"]))
        st.caption(f"Procedencia: {exp[pick]['source']}" + (f" · guardarraíl: {exp[pick]['guardrail']}" if exp[pick]["guardrail"] else ""))
        with st.expander("Evidencia usada (lo único que puede ver el redactor)"):
            st.json(exp[pick]["evidence"])
