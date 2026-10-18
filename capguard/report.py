"""Informe HTML autocontenido (figuras incrustadas) con resultados, límites y metodología."""
from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import Artifacts, by_kind
from .simulator import SAT_LIMITS

COLORS = {"threshold": "#8c8c8c", "zscore": "#e69f00", "iforest": "#009e73", "gbm": "#0072b2"}
LABELS = {"threshold": "Umbral estático", "zscore": "Z-score estacional", "iforest": "Isolation Forest",
          "gbm": "Pronosticador GBM"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_timeline(art: Artifacts) -> str | None:
    te = art.df[art.splits["test"]]
    best = None
    for server, s, lead in art.results["tasks"]["B"]["methods"]["gbm"]["per_event"]:
        if lead is not None and (best is None or lead > best[2]):
            best = (server, s, lead)
    if best is None:
        return None
    server, s, lead = best
    g = te[te["server"] == server].reset_index()  # 'index' = índice original
    lo, hi = max(0, s - 60), min(len(g), s + 24)
    w = g.iloc[lo:hi]
    t = w["timestamp"]
    fig, ax = plt.subplots(4, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1]})
    for a, col, lim, lab in ((ax[0], "cpu", SAT_LIMITS["cpu"], "CPU %"), (ax[1], "ram", SAT_LIMITS["ram"], "RAM %"),
                             (ax[2], "iowait", SAT_LIMITS["iowait"], "iowait %")):
        a.plot(t, w[col], color="#222", lw=1.3)
        a.axhline(lim, color="#c0392b", ls="--", lw=1)
        a.set_ylabel(lab)
    prob = art.scores["gbm"].loc[w["index"]].to_numpy()
    ax[3].plot(t, prob, color=COLORS["gbm"], lw=1.3)
    ax[3].axhline(art.thresholds[("B", "gbm")], color=COLORS["gbm"], ls=":", lw=1)
    ax[3].set_ylabel("P(saturar\nen 30 min)")
    for a in ax:
        a.fill_between(t, 0, 1, where=w["sat_now"].to_numpy(), transform=a.get_xaxis_transform(),
                       color="#c0392b", alpha=0.12, lw=0)
        a.fill_between(t, 0, 1, where=(w["event"] != "").to_numpy(), transform=a.get_xaxis_transform(),
                       color="#999", alpha=0.10, lw=0)
    for name in ("threshold", "zscore", "iforest", "gbm"):
        al = art.alerts[("B", name)].loc[w["index"]].to_numpy()
        if al.any():
            ax[0].axvline(t.iloc[int(np.argmax(al))], color=COLORS[name], lw=1.6, label=LABELS[name])
    ax[0].legend(loc="upper left", fontsize=8, ncol=2, frameon=False, title="Primera alarma", title_fontsize=8)
    ev = w["event"].replace("", np.nan).dropna()
    ax[0].set_title(f"{server} · causa: {ev.iloc[0] if len(ev) else '?'} · zona roja = saturación · "
                    f"gris = evento en curso", fontsize=9)
    fig.autofmt_xdate()
    return _b64(fig)


def fig_summary(art: Artifacts) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    mB = art.results["tasks"]["B"]["methods"]
    mA = art.results["tasks"]["A"]["methods"]
    names = list(LABELS)
    x = np.arange(len(names))
    axes[0].bar(x, [mB[n]["early_rate"] * 100 for n in names], color=[COLORS[n] for n in names])
    axes[0].set_title("B · Saturaciones avisadas ≥10 min antes (%)")
    axes[1].bar(x, [mB[n]["false_alarms_per_server_day"] for n in names], color=[COLORS[n] for n in names])
    axes[1].set_title("B · Falsas alarmas / servidor·día")
    axes[2].bar(x, [mA[n]["recall"] * 100 for n in names], color=[COLORS[n] for n in names])
    axes[2].set_title("A · Incidentes detectados (%)")
    for a in axes:
        a.set_xticks(x, ["Umbral", "Z-score", "IForest", "GBM"])
    fig.tight_layout()
    return _b64(fig)


def fig_leads(art: Artifacts) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    rng = np.random.default_rng(0)
    for i, n in enumerate(LABELS):
        leads = art.results["tasks"]["B"]["methods"][n]["leads_min"]
        ax.scatter(np.array(leads) + rng.normal(0, 0.8, len(leads)), np.full(len(leads), i) + rng.normal(0, 0.07, len(leads)),
                   s=14, alpha=0.6, color=COLORS[n])
        ax.plot([np.median(leads)] * 2, [i - 0.3, i + 0.3], color="#222", lw=2)
    ax.axvline(0, color="#c0392b", lw=1)
    ax.set_yticks(range(4), list(LABELS.values()))
    ax.set_xlabel("Minutos de antelación respecto al inicio de la saturación\n(negativo = tarde; se mira hasta 60 min atrás) · raya negra = mediana")
    return _b64(ax.figure)


def _pct(x):  # noqa: D401
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def _table(task: dict, key_cols: list[tuple[str, str, callable]]) -> str:
    head = "".join(f"<th>{h}</th>" for _, h, _ in [("m", "Método", None)] + key_cols)
    rows = ""
    for n, m in task["methods"].items():
        cells = "".join(f"<td>{f(m[k])}</td>" for k, _, f in key_cols)
        cls = ' class="best"' if n == "gbm" else ""
        rows += f"<tr{cls}><td>{html.escape(m['method'])}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{rows}</table>"


def build_report(art: Artifacts, bench: pd.DataFrame | None = None, real_note: str = "", extra_html: str = "",
                 source: str = "sim") -> str:
    r = art.results
    labelled = r["tasks"]["A"]["methods"]["gbm"]["n_events"] > 0 or art.df["event"].astype(bool).any()
    cols = [("n_events", "Eventos", lambda v: f"{v}"), ("recall", "Detectados", _pct),
            ("early_rate", "Con ≥10 min de antelación", _pct),
            ("median_lead_min", "Antelación mediana (min)", lambda v: "–" if np.isnan(v) else f"{v:+.0f}"),
            ("false_alarms_per_server_day", "Falsas alarmas / servidor·día", lambda v: f"{v:.2f}"),
            ("alert_precision", "Precisión de alarmas", _pct)]
    task_a = (f"<p>Se acierta si salta alguna alarma durante el incidente (los picos legítimos de tráfico cuentan como "
              f"falsa alarma).</p>{_table(r['tasks']['A'], cols)}" if labelled else
              '<div class="note"><b>Se omite:</b> los datos no traen etiquetas de incidentes (columna event), así que '
              'la detección de incidentes no se puede evaluar.</div>')
    kind_tbl = ""
    for n in ("threshold", "zscore", "gbm"):
        d = by_kind(art, "B", n)
        kind_tbl += f"<h4>{LABELS[n]}</h4>" + d.assign(
            recall=d.recall.map(_pct), early_rate=d.early_rate.map(_pct),
            median_lead_min=d.median_lead_min.map(lambda v: f"{v:+.0f}")).rename(
            columns={"n": "episodios", "recall": "avisados", "early_rate": "≥10 min antes",
                     "median_lead_min": "antelación mediana"}).to_html(border=0, classes="small")
    gk = by_kind(art, "B", "gbm")
    gk = gk[(gk.index != "sin_causa_etiquetada") & (gk["n"] >= 3)]
    # las etiquetas de evento pueden venir de un CSV: siempre escapadas antes de entrar en el HTML
    kind_note = (f"Con el GBM, el tipo con más aviso temprano (≥10 min) es "
                 f"<b>{html.escape(str(gk['early_rate'].idxmax()))}</b> ({_pct(gk['early_rate'].max())}) y el que "
                 f"menos, <b>{html.escape(str(gk['early_rate'].idxmin()))}</b> "
                 f"({_pct(gk['early_rate'].min())})." if len(gk) >= 2 else "")
    bench_html = ""
    if bench is not None:
        b = bench.copy()
        b.columns = [f"{a}_{s}" for a, s in b.columns]
        b = b.reset_index()
        rows = "".join(
            f"<tr><td>{x.task}</td><td>{x.method}</td><td>{100*x.recall_mean:.0f}% ± {100*x.recall_std:.0f}</td>"
            f"<td>{100*x.early_rate_mean:.0f}% ± {100*x.early_rate_std:.0f}</td>"
            f"<td>{x.false_alarms_per_server_day_mean:.2f} ± {x.false_alarms_per_server_day_std:.2f}</td></tr>"
            for x in b.itertuples())
        bench_html = ("<h2>Robustez: varios mundos simulados</h2><p>Mismo experimento repetido con distintas "
                      "semillas (media ± desviación). Evita sacar conclusiones de una única simulación.</p>"
                      "<table><tr><th>Tarea</th><th>Método</th><th>Detectados</th><th>≥10 min antes</th>"
                      f"<th>Falsas alarmas/día</th></tr>{rows}</table>")
    tl, sm, ld = fig_timeline(art), fig_summary(art), fig_leads(art)
    csv = source == "csv"
    origin = ("datos <b>reales</b> aportados en CSV" if csv else f"datos <b>simulados</b> (semilla {art.cfg.seed})")
    data_method = ("CSV del usuario normalizado al formato canónico (cabeceras, tipos, unidades, rejilla de 5 min) y "
                   "validado con el mismo contrato de datos que la simulación; la saturación se deriva de las métricas."
                   if csv else "simulador con ciclos diarios/semanales, backup nocturno (iowait alto pero normal), "
                   "4 tipos de incidente (vecino ruidoso, fuga de memoria, fuerza bruta, saturación de disco) y picos "
                   "legítimos de tráfico; la carga de cada servidor sale de la suma del uso de sus cuentas.")
    limits = ("Las cifras dependen de cuántos episodios de saturación haya en el periodo de test: con pocos episodios "
              "son ruidosas. Antes de usarlo en guardia hay que recalibrar el presupuesto de alarmas con el equipo y "
              "validarlo con incidentes históricos." if csv else
              "Los datos son sintéticos: las cifras muestran que el método funciona sobre el comportamiento que se "
              "simuló, no son una promesa sobre producción. Con métricas reales habría que reentrenar, recalibrar el "
              "presupuesto de alarmas con el equipo de guardia y validar con incidentes históricos.")
    ds, g = r["dataset"], r["gbm_sample_metrics"]
    img = lambda b: f'<img src="data:image/png;base64,{b}" style="max-width:100%">' if b else ""
    return f"""<!doctype html><html lang="es"><meta charset="utf-8"><title>capacity-guard · informe</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}}
h1{{margin-bottom:.2rem}}h2{{margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
table{{border-collapse:collapse;margin:.6rem 0;font-size:13.5px}}td,th{{border:1px solid #ddd;padding:5px 9px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}tr.best{{background:#eef6fc}}table.small{{font-size:12.5px}}
.note{{background:#fff8e6;border-left:4px solid #e0a800;padding:.6rem .9rem;margin:1rem 0}}
.kpi{{display:flex;gap:1rem;flex-wrap:wrap}}.kpi div{{background:#f5f7fa;padding:.6rem 1rem;border-radius:8px}}
.kpi b{{display:block;font-size:20px}}</style>
<h1>capacity-guard</h1><p>Detección de anomalías y aviso temprano de saturación en servidores de hosting.
Informe generado sobre {origin}.</p>
<div class="kpi"><div><b>{ds['rows']:,}</b>muestras (5 min)</div><div><b>{ds['servers']}</b>servidores</div>
<div><b>{ds['incidents_test']}</b>incidentes en test</div><div><b>{ds['saturation_episodes_test']}</b>saturaciones en test</div>
<div><b>{g['roc_auc']:.2f}</b>AUC del pronosticador</div></div>
<h2>Resumen</h2>{img(sm)}
<p>Presupuesto de alarmas: cada detector sensible se calibra en validación para no superar
{r['config']['budget_fa_per_server_day']} falsas alarmas por servidor y día; el umbral estático se deja como lo
configuraría una monitorización clásica (CPU 85 %, RAM 90 %, iowait 25 %).</p>
<h2>Tarea B · Aviso temprano de saturación</h2>
<p>Una saturación es CPU ≥ 90 %, RAM ≥ 95 % o iowait ≥ 30 %. Se premia avisar con antelación suficiente para actuar.</p>
{_table(r['tasks']['B'], cols)}{img(ld)}
<h2>Ejemplo real del conjunto de test</h2>{img(tl)}
<h2>¿En qué tipo de incidente funciona cada método?</h2>
<p>El adelanto depende de cómo se forme el problema: lo que crece despacio se puede anticipar; lo que llega de golpe
solo se puede detectar rápido. {kind_note}</p>
{kind_tbl}
<h2>Tarea A · Detección de incidentes</h2>
{task_a}
{bench_html}{real_note}{extra_html}
<h2>Metodología</h2>
<ul><li><b>Datos:</b> {data_method}</li>
<li><b>Partición temporal</b> 60 % entrenamiento · 15 % validación · 25 % test, sin mezclar el futuro con el pasado.</li>
<li><b>Variables causales:</b> desviación respecto a lo normal para esa hora de la semana (mediana/MAD por servidor),
CPU no explicada por el tráfico, pendientes y tiempo estimado hasta llenar la RAM.</li>
<li><b>Alarma:</b> el score debe superar el umbral en 2 muestras seguidas (10 min).</li></ul>
<div class="note"><b>Límites.</b> {limits}</div></html>"""
