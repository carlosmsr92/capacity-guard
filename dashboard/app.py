"""Dashboard de BI de capacity-guard (Streamlit). Ejecutar: streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from capguard import analysis  # noqa: E402
from capguard.simulator import SAT_LIMITS  # noqa: E402
from dashboard.formato import fn0, fn2, fni, grafico, tabla, texto_es  # noqa: E402
from dashboard.labels import KPIS, feature, label, plans_in_text  # noqa: E402
from dashboard.marca import ICONO, cabecera, pie  # noqa: E402


def _has_results(d: Path) -> bool:
    return all((d / f).exists() for f in ("results.json", "data/metrics.parquet", "data/scores.parquet"))


# CAPGUARD_OUT manda; si no, outputs/ (resultados propios) y, si no existe, la instantánea versionada demo/
# (p. ej. en Streamlit Cloud, que clona el repo sin outputs/data y no ejecuta el pipeline)
if os.environ.get("CAPGUARD_OUT"):
    OUT = Path(os.environ["CAPGUARD_OUT"])
else:
    OUT = ROOT / "outputs" if _has_results(ROOT / "outputs") else ROOT / "demo"
DEMO = OUT.resolve() == (ROOT / "demo").resolve()
DATA = OUT / "data"
# Paleta con significado fijo (contraste ≥3:1 sobre el fondo claro #ffffff y el oscuro #0e141c del tema):
# gris = métrica, rojo = límite/saturación, azul = riesgo/modelo, ámbar = alarma
C_METRIC, C_LIMIT, C_RISK, C_ALARM = "#808891", "#d64545", "#2f80c8", "#d97706"
METHODS = {"threshold": "Umbral estático", "zscore": "Z-score estacional", "iforest": "Isolation Forest", "gbm": "Pronosticador GBM"}

st.set_page_config(page_title="capacity-guard · CMSR92", page_icon=ICONO, layout="wide")


def _stamp() -> int:
    f = OUT / "results.json"
    return f.stat().st_mtime_ns if f.exists() else 0


@st.cache_data(show_spinner=False)
def load_results(stamp: int) -> dict:
    return json.loads((OUT / "results.json").read_text())


def json_legible(data) -> str:
    """JSON sangrado y sin resaltado de sintaxis: los colores del visor de st.json no llegan a 4,5:1 en el tema claro."""
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


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


# Formato español en los ejes y textos emergentes de Altair/Vega-Lite (miles con punto, decimales con coma)
ALT_ES = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["", "\u00a0€"]}}


def hbar(df: pd.DataFrame, cat: str, val: str, color: str, height: int) -> alt.Chart:
    """Barras horizontales ordenadas de mayor a menor, con espacio para etiquetas largas en español."""
    return alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f"{val}:Q", title=None),
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=",.2f")],
    ).properties(height=height).configure(locale=ALT_ES)


def segments(df: pd.DataFrame, col: str):
    on = (df[col] != "") if df[col].dtype == object or str(df[col].dtype).startswith("str") else df[col].astype(bool)
    grp = (on != on.shift()).cumsum()
    for _, g in df[on].groupby(grp[on]):
        yield g["timestamp"].iloc[0], g["timestamp"].iloc[-1], (g[col].iloc[0] if col == "event" else "")


if not _has_results(OUT):
    st.error("Faltan los resultados o sus datos (no se versionan en git). Ejecuta primero: `python -m capguard all`")
    st.stop()

R = load_results(_stamp())
main_b = R["main"]["tasks"]["B"]["methods"]
CSV = R.get("source", "sim") == "csv"
CAP = R.get("capabilities", {})
# En móvil, los valores de texto largos de las tarjetas pasan de línea en vez de cortarse con «…»
st.markdown("""<style>
[data-testid="stMetricValue"] div, [data-testid="stMetricValue"] p {white-space: normal; text-overflow: clip; line-height: 1.2}
</style>""", unsafe_allow_html=True)
cabecera("capacity-guard", "Avisa antes de que un servidor de hosting se sature y detecta clientes cerca del límite de su plan · "
         + ("datos reales (CSV)" if CSV else "datos simulados"))
if DEMO:
    cfg = R["main"]["config"]
    st.info(f"Estás viendo datos de demostración: una copia de los resultados de `python -m capguard all` "
            f"(semilla {cfg['seed']}, {cfg['n_servers']} servidores durante {cfg['days']} días). Si ejecutas ese "
            "comando en local, el dashboard mostrará tus propios resultados.")
skipped = [CAP[f"{k}_reason"] for k, v in CAP.items() if v is False and f"{k}_reason" in CAP]
if skipped:
    st.warning("Partes omitidas con estos datos: " + " · ".join(_md(s) for s in skipped))
tabs = st.tabs(["Resumen", "Servidores", "Negocio", "Calidad de datos", "Modelo", "Clientes", "Alertas explicadas"])

with tabs[0]:
    g = main_b["gbm"]
    c = st.columns(5)
    c[0].metric("Muestras", f"{fni(R['ingest']['rows_final' if CSV else 'rows_clean'])}", border=True)
    c[1].metric("Saturaciones en test", g["n_events"], border=True)
    c[2].metric("Avisadas ≥10 min antes (GBM)", f"{fn0(100 * g['early_rate'])} %", border=True)
    c[3].metric("Falsas alarmas/servidor·día", f"{fn2(g['false_alarms_per_server_day'])}", border=True)
    c[4].metric("AUC del pronosticador", f"{fn2(R['main']['gbm_sample_metrics']['roc_auc'])}", border=True)
    rows = [{"Método": m["method"], "Detectadas": f"{fn0(100 * m['recall'])} %", "Con ≥10 min de antelación": f"{fn0(100 * m['early_rate'])} %",
             "Antelación mediana (min)": m["median_lead_min"], "Falsas alarmas / servidor·día": round(m["false_alarms_per_server_day"], 2)}
            for m in main_b.values()]
    st.subheader("Aviso temprano de saturación (periodo de test)")
    st.dataframe(tabla(pd.DataFrame(rows), {"Antelación mediana (min)": 0}), hide_index=True, width="stretch")
    st.info("Cada método se ajusta con datos de validación para no pasar de un número máximo de falsas alarmas. "
            "El umbral fijo se deja como lo pondría una monitorización clásica, para tener con qué comparar.")

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
            fig.add_trace(go.Scatter(x=d["timestamp"], y=d[col], mode="lines", line=dict(width=1.2, color=C_METRIC), showlegend=False), row=i, col=1)
            fig.add_hline(y=lim, line=dict(color=C_LIMIT, dash="dash", width=1), row=i, col=1)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["risk_gbm"], mode="lines", line=dict(color=C_RISK, width=1.2), showlegend=False), row=4, col=1)
        al = d[d["alert_B_gbm"]]
        fig.add_trace(go.Scatter(x=al["timestamp"], y=al["risk_gbm"], mode="markers", marker=dict(color=C_ALARM, size=5),
                                 name="alarma", showlegend=True), row=4, col=1)
        for a, b, kind in segments(d, "event"):
            fig.add_vrect(x0=a, x1=b, fillcolor=C_METRIC, opacity=0.18, line_width=0, annotation_text=label(kind), annotation_font_size=9)
        for a, b, _ in segments(d, "sat_now"):
            fig.add_vrect(x0=a, x1=b, fillcolor=C_LIMIT, opacity=0.20, line_width=0)
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m %H:%M")  # fechas sin nombres de mes en inglés
        grafico(fig, width="stretch")
        st.caption("Gris: incidente en curso (con su causa) · rojo: saturación · naranja: alarma del modelo. "
                   "Fuera del periodo de prueba el modelo ya vio esos datos al entrenar, así que ahí no cuenta como evaluación.")

with tabs[2]:
    e = R["eda"]
    if "by_kind" in e:
        c = st.columns([1, 1, 2])  # el tipo de incidente es un texto largo
        c[0].metric("Incidentes (simulados)", e["n_incidents"], border=True)
        c[1].metric("Por servidor y mes", f"{fn0(e['incidents_per_server_month'])}", border=True)
        c[2].metric("Tipo con más tiempo saturado", label(e["worst_kind_for_saturation"]), border=True)
        kinds = pd.DataFrame(e["by_kind"]).assign(kind=lambda k: k["kind"].map(label))
        col1, col2 = st.columns(2)
        col1.bar_chart(pd.Series(e["by_hour"], name="incidentes"), height=260)
        col1.caption("Inicio de incidentes por hora del día")
        hours = pd.DataFrame({"Tipo": kinds["kind"], "Horas": kinds["min_saturados"] / 60})
        col2.altair_chart(hbar(hours, "Tipo", "Horas", C_LIMIT, 260), width="stretch")
        col2.caption("Horas de saturación por tipo de incidente")
    else:
        st.info(CAP.get("incident_eda_reason", "Sin etiquetas de incidentes."))
    st.subheader("Modelo de coste (editable)")
    st.caption("Son valores de ejemplo: cámbialos por los de tu empresa (compensaciones al cliente por caídas, coste de la guardia).")
    s1, s2, s3 = st.columns(3)
    c_miss = s1.slider("Coste de NO avisar a tiempo (€)", 10, 2000, 200, 10, key="c_miss")
    c_fa = s2.slider("Coste de una falsa alarma (€)", 1, 300, 15, 1, key="c_fa")
    c_mit = s3.slider("Coste de avisar a tiempo (€)", 0, 200, 20, 5, key="c_mit")
    ct = analysis.cost_table(main_b, {"c_miss": c_miss, "c_fa": c_fa, "c_mitigated": c_mit})
    best = ct.loc[ct["coste_total"].idxmin(), "method"]
    st.dataframe(tabla(ct[["method", "avisadas_a_tiempo", "no_avisadas", "falsas_alarmas", "coste_total", "coste_por_servidor_mes",
                           "ahorro_vs_umbral_pct"]].rename(columns=label), {label("ahorro_vs_umbral_pct"): 1}),
                 hide_index=True, width="stretch")
    st.success(f"Con estos costes, el método más barato es: **{best}**")

with tabs[3]:
    q = R["ingest"]
    if CSV:
        st.write(f"CSV: {fni(q['rows_read'])} filas leídas → {fni(q['rows_final'])} finales · resolución {_md(q['frequency'])} · "
                 f"conversiones: {_md(', '.join(q['unit_conversions']) or 'ninguna')} · "
                 f"columnas ignoradas: {_md(', '.join(q['ignored_columns']) or 'ninguna')}.")
    else:
        st.write(f"Datos con errores: {fni(q['rows_dirty'])} filas → tras repararlos: {fni(q['rows_clean'])} "
                 f"(las correctas eran {fni(q['rows_truth'])}). Filas rellenadas: {q['imputed_rows']}.")
    names = [c["name"] for c in q["quality_before"]["checks"]]
    before = {c["name"]: c["violations"] for c in q["quality_before"]["checks"]}
    after = {c["name"]: c["violations"] for c in q["quality_after"]["checks"]}
    st.dataframe(tabla(pd.DataFrame({"Comprobación": [label(n) for n in names], "Antes": [before[n] for n in names],
                                     "Después": [after.get(n, 0) for n in names]})),
                 hide_index=True, width="stretch")
    if not CSV:
        st.write("Error de los valores rellenados, en % de la variación típica de cada métrica:")
        st.dataframe(tabla(pd.Series(q["imputation_mae_vs_metric_std"]).rename(label).mul(100).rename("%").to_frame().T, 1),
                     width="stretch")

with tabs[4]:
    st.subheader("Walk-forward (3 periodos futuros)")
    wf = pd.DataFrame(R["walk_forward"])
    st.dataframe(tabla(wf.pivot_table(index="method", columns="test_days", values="early_rate")
                       .rename_axis(index=label("method"), columns=label("test_days"))), width="stretch")
    st.caption("Porcentaje de saturaciones avisadas con al menos 10 minutos de margen, en cada periodo de prueba.")
    c1, c2 = st.columns(2)
    imp = pd.DataFrame(R["importance"]).assign(feature=lambda f: f["feature"].map(feature))
    c1.write("Importancia de variables (permutación)")
    c1.altair_chart(hbar(imp, "feature", "importance", C_RISK, 360), width="stretch")
    ae = R["deep"]["B"]
    c2.write("Autoencoder (deep learning), tarea B")
    c2.metric(KPIS["detectadas"], f"{fn0(100 * ae['recall'])} %", border=True)
    c2.metric(KPIS["avisadas_>=10min"], f"{fn0(100 * ae['early_rate'])} %", border=True)
    c2.metric(KPIS["falsas_alarmas_servidor_dia"], f"{fn2(ae['false_alarms_per_server_day'])}", border=True)
    st.subheader("Diagnóstico de causa")
    dg = R["diagnosis"] or {"by_lag": {}}
    if not R["diagnosis"]:
        st.info(CAP.get("diagnosis_reason", "Diagnóstico no disponible."))
    for lag, v in dg["by_lag"].items():
        st.write(f"**{lag}** · {v['n']} eventos · acierto {fn0(100 * v['accuracy'])} % (clase mayoritaria {fn0(100 * v['majority_baseline'])} %)")
        names = [label(k) for k in v["labels"]]
        st.dataframe(tabla(pd.DataFrame(v["confusion"], index=[f"real: {k}" for k in names], columns=names)), width="stretch")

with tabs[5]:
    customers = R.get("customers", {})
    if not customers.get("available"):
        st.info(_md(customers.get("reason", CAP.get("customers_reason", "Clientes no disponibles con esta fuente."))))
    else:
        feats = pd.DataFrame(customers.get("features", []))
        offers = pd.DataFrame(customers.get("offers", []))
        m = st.columns(4)
        m[0].metric("Cuentas", customers["n_accounts"], border=True)
        m[1].metric("Cerca de su límite", customers["n_flagged"], border=True)
        m[2].metric("Con causa técnica (sin venta)", len(customers.get("technical_accounts", [])), border=True)
        m[3].metric("Borradores comerciales", int(offers["oferta"].isin(["upgrade", "temporal"]).sum()) if not offers.empty else 0,
                    border=True)
        st.warning(f"Todas las ofertas son borradores («{label('pendiente_revision')}»): ninguna se envía sin que una persona la apruebe.")
        c1, c2 = st.columns(2)
        cluster_values = ["Todos"] + sorted(offers["cluster"].dropna().unique().tolist()) if not offers.empty else ["Todos"]
        offer_values = ["Todos"] + sorted(offers["oferta"].dropna().unique().tolist()) if not offers.empty else ["Todos"]
        cluster = c1.selectbox("Cluster", cluster_values, key="customer_cluster", format_func=label)
        offer = c2.selectbox("Oferta", offer_values, key="customer_offer", format_func=label)
        if cluster != "Todos":
            offers = offers[offers["cluster"] == cluster]
        if offer != "Todos":
            offers = offers[offers["oferta"] == offer]
        cols = ["account_id", "cluster", "oferta", "plan_actual", "plan_propuesto", "precio_actual", "precio_propuesto",
                "estado", "motivo"]
        shown = offers[[c for c in cols if c in offers]]
        shown = shown.assign(**{c: shown[c].map(label, na_action="ignore") for c in ("cluster", "oferta", "plan_actual", "plan_propuesto", "estado")
                                if c in shown})
        if "motivo" in shown:
            shown = shown.assign(motivo=shown["motivo"].map(texto_es, na_action="ignore"))
        st.dataframe(tabla(shown.rename(columns=label)), hide_index=True, width="stretch")
        value = customers.get("offer_value", {})
        if value.get("sensitivity"):
            st.caption("Ingresos extra al mes según cuántas ofertas se acepten (es una suposición para ilustrar, no un dato): " + " · ".join(
                f"{fn0(100 * s['acceptance'])} % → {fn0(s['expected_monthly_value'])} €" for s in value["sensitivity"]))
        if not offers.empty:
            pick = st.selectbox("Cuenta", offers["account_id"].tolist(), key="customer_account")
            usage_path = DATA / "account_usage_hourly.parquet"
            if usage_path.exists():
                from capguard.accounts import LIMIT_FIELD, PLANS, RESOURCES
                u = pd.read_parquet(usage_path, filters=[("account_id", "==", pick)])
                plan = offers.set_index("account_id").loc[pick, "plan_actual"]
                if plan in PLANS and not u.empty:
                    pct = pd.DataFrame({r: 100 * u[r].astype(float) / getattr(PLANS[plan], LIMIT_FIELD[r])
                                        for r in RESOURCES}).set_index(u["timestamp"]).rename(columns=label)
                    long = pct.rename_axis("Fecha").reset_index().melt("Fecha", var_name="Recurso", value_name="% del límite")
                    st.altair_chart(alt.Chart(long).mark_line(strokeWidth=1.2).encode(
                        x=alt.X("Fecha:T", axis=alt.Axis(format="%d/%m", title=None)),
                        y=alt.Y("% del límite:Q", title=None), color=alt.Color("Recurso:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=[alt.Tooltip("Fecha:T", format="%d/%m %H:%M"), "Recurso:N", alt.Tooltip("% del límite:Q", format=".0f")],
                    ).properties(height=240).configure(locale=ALT_ES), width="stretch")
                    st.caption("% del límite del plan por hora y recurso (100 % = limitado).")
            st.write(_md(texto_es(plans_in_text(offers.set_index("account_id").loc[pick, "texto"]))))
            with st.expander("Datos con los que se redactó la oferta"):
                st.code(json_legible(offers.set_index("account_id").loc[pick, "evidencia"]), language=None)

with tabs[6]:
    exp = json.loads((DATA / "alerts_explained.json").read_text())
    if not exp:
        st.warning("No hay alarmas explicadas.")
    else:
        labels = [f"{e['server']} · {e['timestamp']}" for e in exp]
        pick = st.selectbox("Alarma", range(len(exp)), format_func=lambda i: labels[i], key="alert_pick")
        st.write(_md(texto_es(exp[pick]["text"])))
        st.caption(f"Redactado con: {exp[pick]['source']}" + (f" · control de cifras: {exp[pick]['guardrail']}" if exp[pick]["guardrail"] else ""))
        with st.expander("Datos con los que se redactó (el texto no puede usar nada más)"):
            st.code(json_legible(exp[pick]["evidence"]), language=None)

pie("capacity-guard")
