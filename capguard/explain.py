"""IA aplicada: de una alarma a una explicación útil para el técnico de guardia.

Diseño en capas, pensado para que el LLM no pueda inventar nada:
  1. `build_evidence`: paquete de hechos medidos (métricas, desviaciones, riesgo, diagnóstico probable).
  2. `render_template`: explicación determinista en español (funciona sin red ni claves).
  3. `llm_explain` (opcional): un LLM redacta SOLO a partir de esos hechos; `check_numbers` rechaza cualquier
     cifra que no esté en la evidencia y, si falla, se usa la plantilla.
El sistema avisa y sugiere; la decisión y la atención al cliente siguen siendo humanas.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

KIND_ES = {"noisy_neighbor": "vecino ruidoso (una cuenta consume recursos sin más tráfico)",
           "memory_leak": "fuga de memoria (la RAM sube de forma sostenida)",
           "brute_force": "ataque de fuerza bruta / bots (pico de peticiones)",
           "io_saturation": "saturación de disco (iowait alto)",
           "legit_surge": "pico legítimo de tráfico (campaña o visita masiva)"}
RUNBOOK = {  # acciones de ejemplo; el equipo real las sustituye por su propio runbook
    "noisy_neighbor": ["Identificar la cuenta con más CPU/IO en el servidor", "Aplicar o revisar límites por cuenta (LVE/cgroups)"],
    "memory_leak": ["Localizar el proceso cuyo RSS crece", "Programar un reinicio controlado antes de llegar al límite"],
    "brute_force": ["Limitar peticiones a wp-login/xmlrpc en el WAF", "Bloquear IPs/rangos que concentran el tráfico"],
    "io_saturation": ["Comprobar backups o tareas pesadas en curso", "Limitar el I/O de la tarea o moverla a otra ventana"],
    "legit_surge": ["No hay nada que arreglar: vigilar la capacidad", "Valorar mover cuentas si se acerca al límite"],
}
DEFAULT_ACTIONS = ["Revisar el servidor con el runbook general de incidentes", "Identificar la cuenta o el proceso que más consume"]


def kind_es(kind: str) -> str:
    """Descripción en español de una causa; las etiquetas propias de un CSV se muestran tal cual."""
    return KIND_ES.get(kind, f"{kind} (etiqueta de los datos)")
SYSTEM_PROMPT = (
    "Eres un asistente para técnicos de guardia de un hosting. Redacta un aviso breve (máx. 90 palabras) en español "
    "usando EXCLUSIVAMENTE los datos del JSON. No inventes cifras, causas ni acciones que no estén en el JSON. "
    "Si la probabilidad del diagnóstico es baja, dilo. No prometas resultados.")
_METRIC_NAMES = {"cpu": "CPU %", "ram": "RAM %", "iowait": "iowait %", "net": "red Mbps", "load1": "load", "req": "peticiones/min"}


def build_evidence(df: pd.DataFrame, feats: pd.DataFrame, idx, risk: float, diagnosis: dict | None = None) -> dict:
    row = df.loc[idx]
    g = df[df["server"] == row["server"]]
    pos = g.index.get_loc(idx)
    past = g.iloc[max(0, pos - 12)]
    cols = list(_METRIC_NAMES)
    z = feats.loc[idx, [f"z_{c}" for c in cols] + ["cpu_unexplained"]].astype(float)
    top = z.abs().sort_values(ascending=False).head(3).index
    ev = {
        "server": row["server"], "timestamp": str(row["timestamp"])[:16],
        "risk_30min_pct": round(100 * float(risk), 1),
        "now": {c: round(float(row[c]), 1) for c in cols},
        "one_hour_ago": {c: round(float(past[c]), 1) for c in cols},
        "largest_deviations": [{"signal": ("CPU no explicada por el tráfico" if k == "cpu_unexplained" else _METRIC_NAMES[k[2:]]),
                                "mad_units": round(float(z[k]), 1)} for k in top],
    }
    slope = float(feats.loc[idx, "ram_slope"])
    if slope > 0.01 and row["ram"] < 95 and (95 - row["ram"]) / slope * 5 <= 720:  # solo si el límite está a <12 h
        ev["ram_minutes_to_limit"] = int((95 - row["ram"]) / slope * 5)
    if diagnosis:
        ranked = sorted(diagnosis.items(), key=lambda kv: -kv[1])
        ev["diagnosis"] = {"most_likely": ranked[0][0], "probability_pct": round(100 * float(ranked[0][1]), 0),
                           "alternative": ranked[1][0], "alternative_pct": round(100 * float(ranked[1][1]), 0)}
        ev["suggested_actions"] = RUNBOOK.get(ranked[0][0], DEFAULT_ACTIONS)
    return ev


def render_template(ev: dict) -> str:
    d = ev.get("diagnosis")
    parts = [f"[{ev['server']} · {ev['timestamp']}] Riesgo de saturación en 30 min: {ev['risk_30min_pct']:.0f} %."]
    if d:
        parts.append(f"Causa más probable: {kind_es(d['most_likely'])} ({d['probability_pct']:.0f} %); "
                     f"segunda opción: {kind_es(d['alternative']).split(' (')[0]} ({d['alternative_pct']:.0f} %).")
    n, p = ev["now"], ev["one_hour_ago"]
    parts.append(f"Ahora: CPU {n['cpu']} % (hace 1 h {p['cpu']} %), RAM {n['ram']} % (antes {p['ram']} %), "
                 f"iowait {n['iowait']} % (antes {p['iowait']} %), {n['req']:.0f} peticiones/min.")
    parts.append("Señales más fuera de lo normal: " + ", ".join(
        f"{x['signal']} ({x['mad_units']:+.1f} desviaciones)" for x in ev["largest_deviations"]) + ".")
    if "ram_minutes_to_limit" in ev:
        parts.append(f"A este ritmo la RAM llega al límite en ~{ev['ram_minutes_to_limit']} min.")
    if "suggested_actions" in ev:
        parts.append("Acciones sugeridas: " + "; ".join(ev["suggested_actions"]) + ".")
    return " ".join(parts)


def _allowed_numbers(ev) -> list[float]:
    out: list[float] = []

    def walk(x):
        if isinstance(x, dict):
            [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)):
            [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            out.extend([float(x), abs(float(x))])
            if abs(x) <= 1:
                out.append(float(x) * 100)
    walk(ev)
    return out + [30.0, 1.0, 60.0]  # horizonte de 30 min y "1 hora" forman parte del propio diseño


def check_numbers(text: str, ev: dict, strip: list[str] | None = None) -> tuple[bool, list[str]]:
    """Guardarraíl anti-alucinación: toda cifra del texto debe existir en la evidencia (con tolerancia de redondeo)."""
    strip = strip if strip is not None else [ev["server"], ev["timestamp"]]
    clean = text
    for value in strip:
        clean = clean.replace(str(value), " ")
    allowed = _allowed_numbers(ev)
    bad = []
    for tok in re.findall(r"\d+(?:[.,]\d+)?", clean):
        v = float(tok.replace(",", "."))
        d = len(tok.replace(",", ".").split(".")[1]) if re.search(r"[.,]", tok) else 0
        # la cifra debe ser un dato de la evidencia redondeado a los decimales con que el texto la escribe
        if not any(abs(round(a, d) - v) < 1e-6 for a in allowed):
            bad.append(tok)
    return (not bad), bad


def llm_explain(ev: dict, client, model: str | None = None, system: str = SYSTEM_PROMPT) -> str:
    model = model or os.environ.get("CAPGUARD_LLM_MODEL", "claude-haiku-5-5")
    # max_tokens deja margen al razonamiento interno del modelo; la longitud del texto la acota el prompt (90 palabras)
    msg = client.messages.create(model=model, max_tokens=2048, system=system,
                                 output_config={"effort": "low"},
                                 messages=[{"role": "user", "content": json.dumps(ev, ensure_ascii=False)}])
    stop = getattr(msg, "stop_reason", "end_turn")
    if stop != "end_turn":  # p. ej. "max_tokens" (texto cortado) o "refusal": mejor la plantilla que un aviso a medias
        raise RuntimeError(f"respuesta incompleta del LLM (stop_reason={stop})")
    return "".join(b.text for b in msg.content if getattr(b, "type", "text") == "text").strip()


def explain(ev: dict, client=None, model: str | None = None) -> dict:
    """Devuelve el texto final y su procedencia. Cualquier fallo del LLM degrada a la plantilla, nunca a un error."""
    base = render_template(ev)
    if client is None:
        return {"text": base, "source": "plantilla", "guardrail": None}
    try:
        text = llm_explain(ev, client, model)
        ok, bad = check_numbers(text, ev)
        if ok and text:
            return {"text": text, "source": "llm", "guardrail": "ok"}
        return {"text": base, "source": "plantilla", "guardrail": f"rechazado: cifras no respaldadas {bad}"}
    except Exception as e:
        return {"text": base, "source": "plantilla", "guardrail": f"LLM no disponible ({type(e).__name__})"}


def default_client():
    """Cliente de Anthropic si hay clave en el entorno; si no, None (se usa la plantilla)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()
