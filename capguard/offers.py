"""Ofertas comerciales en borrador: política determinista, guardas y revisión humana obligatoria.

Nada se envía: cada oferta sale en estado `pendiente_revision` para que una persona la apruebe o la descarte. El
texto lo redacta una plantilla (o un LLM opcional) y pasa el mismo guardarraíl de cifras que las alarmas.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from .accounts import PLANS
from .explain import check_numbers, llm_explain

LADDER = ("basico", "plus", "plus_intermedio", "pro", "vps")  # de menor a mayor; vps es el plan máximo
# perfil -> (tipo de oferta, descuento); el plan de destino lo decide _target_plan
OFFER_POLICY = {
    "crecimiento": ("upgrade", 0.15),
    "campanas": ("temporal", 0.0),
    "intensivo": ("upgrade", 0.05),
    "esporadico": ("consejo", 0.0),
    "moderado": ("sin_oferta", 0.0),
    "causa_tecnica": ("guardia_tecnica", 0.0),
}
DEFAULT_OFFER_PARAMS = {"months": 3, "temporary_days": 30, "temporary_share": 0.5, "cooldown_days": 90,
                        "max_discount": 0.20, "min_tenure_days": 30}
OFFER_SYSTEM_PROMPT = (
    "Redacta en español una propuesta comercial breve (máx. 60 palabras) para un cliente de hosting usando "
    "EXCLUSIVAMENTE los datos del JSON. No inventes cifras ni plazos, no uses urgencia artificial ni presión.")
ILLUSTRATIVE_ACCEPTANCE = (0.10, 0.25, 0.40)


@dataclass
class Offer:
    account_id: str
    cluster: str
    oferta: str
    plan_actual: str
    plan_propuesto: str
    precio_actual: float
    precio_propuesto: float
    texto: str
    fuente: str = "plantilla"
    estado: str = "pendiente_revision"  # nunca se envía sin revisión humana
    motivo: str = ""


def profile(cluster: str) -> str:
    """Perfil base de un cluster numerado («campanas_2» -> «campanas»)."""
    return re.sub(r"_\d+$", "", str(cluster))


def _target_plan(kind: str, cluster: str, plan: str) -> str | None:
    """Plan propuesto: el intermedio para intensivos de planes pequeños; si no, el siguiente de la escalera."""
    if kind not in ("upgrade", "temporal"):
        return None
    if profile(cluster) == "intensivo" and plan in ("basico", "plus"):
        return "plus_intermedio"
    i = LADDER.index(plan) if plan in LADDER else 0
    return LADDER[i + 1] if i + 1 < len(LADDER) else None


def build_offer_evidence(row: pd.Series, cluster: str, kind: str, target: str | None, price_now: float,
                         price_new: float, discount: float, params: dict) -> dict:
    """Los únicos hechos que puede usar el redactor (plantilla o LLM)."""
    return {"account_id": row["account_id"], "perfil": profile(cluster), "oferta": kind,
            "plan_actual": row["plan"], "plan_propuesto": target or row["plan"],
            "precio_actual": round(price_now, 2), "precio_propuesto": round(price_new, 2),
            "descuento_pct": round(100 * discount), "meses_descuento": params["months"],
            "dias_ampliacion": params["temporary_days"],
            "p95_utilizacion_pct": round(100 * float(row.get("p95_max", 0.0))),
            "dias_cerca_limite": int(row.get("dias_cerca_limite", 0)), "dias_limitado": int(row.get("dias_limitado", 0))}


def render_offer(ev: dict) -> str:
    kind = ev["oferta"]
    if kind == "temporal":
        return (f"Su plan {ev['plan_actual']} ha estado cerca del límite {ev['dias_cerca_limite']} días. Proponemos "
                f"ampliar temporalmente sus recursos a los del plan {ev['plan_propuesto']} durante "
                f"{ev['dias_ampliacion']} días por {ev['precio_propuesto']:.2f} € al mes (ahora {ev['precio_actual']:.2f} €).")
    if kind == "upgrade":
        discount = (f" con un descuento del {ev['descuento_pct']} % los primeros {ev['meses_descuento']} meses"
                    if ev["descuento_pct"] else "")
        return (f"Su plan {ev['plan_actual']} ha estado cerca del límite {ev['dias_cerca_limite']} días. Proponemos "
                f"pasar al plan {ev['plan_propuesto']}{discount}: {ev['precio_propuesto']:.2f} € al mes "
                f"(ahora {ev['precio_actual']:.2f} €).")
    if kind == "consejo":
        return ("Los picos de uso son breves y esporádicos: recomendamos revisar la caché y las tareas programadas "
                "antes de plantear un cambio de plan.")
    if kind == "plan_maximo":
        return "La cuenta ya está en el plan máximo: conviene revisar con el cliente una solución dedicada."
    if kind == "guardia_tecnica":
        return "El uso alto tiene una causa técnica: no se propone ninguna venta. Debe revisarlo la guardia."
    return "No se propone ninguna oferta comercial para este perfil."


def offer_text(ev: dict, client=None, model: str | None = None) -> dict:
    """Texto de la oferta; el LLM es opcional y cualquier cifra no respaldada devuelve la plantilla."""
    base = render_offer(ev)
    strip = [ev["account_id"]]
    if client is None:
        ok, bad = check_numbers(base, ev, strip)
        return {"text": base, "source": "plantilla", "guardrail": "ok" if ok else f"plantilla con cifras {bad}"}
    try:
        text = llm_explain(ev, client, model, system=OFFER_SYSTEM_PROMPT)
        ok, bad = check_numbers(text, ev, strip)
        if ok and text:
            return {"text": text, "source": "llm", "guardrail": "ok"}
        return {"text": base, "source": "plantilla", "guardrail": f"rechazado: cifras no respaldadas {bad}"}
    except Exception as e:
        return {"text": base, "source": "plantilla", "guardrail": f"LLM no disponible ({type(e).__name__})"}


def generate_offers(feats: pd.DataFrame, technical: set[str] | None = None, params: dict | None = None,
                    history: dict[str, pd.Timestamp] | None = None, reference_date=None, client=None) -> list[dict]:
    """Un borrador por cuenta marcada. `history` = fecha de la última oferta de cada cuenta (enfriamiento)."""
    params = {**DEFAULT_OFFER_PARAMS, **(params or {})}
    technical, history = technical or set(), history or {}
    ref = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp.now().normalize()
    rows = []
    for _, row in feats[feats["near_limit"]].iterrows():
        account = row["account_id"]
        cluster = "causa_tecnica" if account in technical else str(row.get("cluster") or "moderado")
        kind, discount = OFFER_POLICY.get(profile(cluster), OFFER_POLICY["moderado"])
        motivo = "cuenta marcada por proximidad repetida al límite"
        alta = pd.to_datetime(row.get("alta"), errors="coerce")
        last = history.get(account)
        if kind in ("upgrade", "temporal") and pd.notna(alta) and (ref - alta).days < params["min_tenure_days"]:
            kind, motivo = "sin_oferta", f"alta hace menos de {params['min_tenure_days']} días"
        elif kind in ("upgrade", "temporal") and last is not None and (ref - pd.Timestamp(last)).days < params["cooldown_days"]:
            kind, motivo = "sin_oferta", f"ya recibió una oferta hace menos de {params['cooldown_days']} días"
        target = _target_plan(kind, cluster, row["plan"])
        if kind in ("upgrade", "temporal") and target is None:
            kind = "plan_maximo"
        discount = min(discount, params["max_discount"]) if kind == "upgrade" else 0.0
        now = float(PLANS[row["plan"]].precio_mes) if row["plan"] in PLANS else float("nan")
        if kind == "upgrade":
            new = PLANS[target].precio_mes * (1 - discount)
        elif kind == "temporal":  # suplemento: una parte de la diferencia con el plan superior, solo mientras dura
            new = now + params["temporary_share"] * (PLANS[target].precio_mes - now)
        else:
            new = now
        ev = build_offer_evidence(row, cluster, kind, target, now, new, discount, params)
        txt = offer_text(ev, client)
        rows.append({**asdict(Offer(account, cluster, kind, row["plan"], target or row["plan"], round(now, 2),
                                    round(new, 2), txt["text"], txt["source"], motivo=motivo)),
                     "guardarrail": txt["guardrail"], "evidencia": ev})
    return rows


def offer_value(offers: list[dict], acceptance=ILLUSTRATIVE_ACCEPTANCE) -> dict:
    """Ingreso mensual adicional si se acepta una fracción de las ofertas. La aceptación es un SUPUESTO ilustrativo."""
    commercial = [o for o in offers if o["oferta"] in ("upgrade", "temporal")]
    uplift = sum(o["precio_propuesto"] - o["precio_actual"] for o in commercial)
    return {"n_offers": len(offers), "commercial": len(commercial), "illustrative": True,
            "note": "La tasa de aceptación es un supuesto de ejemplo; sustitúyela por la histórica de la empresa.",
            "sensitivity": [{"acceptance": a, "expected_monthly_value": round(uplift * a, 2)} for a in acceptance]}
