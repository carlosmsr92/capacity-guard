"""Capa central de etiquetas del dashboard: traduce nombres internos (variables, estados, causas) a texto en español.

Solo afecta a lo que se muestra: los datos, filtros y claves siguen usando los nombres internos.
"""
from __future__ import annotations

import re

METRICS = {"cpu": "CPU", "ram": "RAM", "iowait": "Iowait", "io": "Iowait", "net": "Red", "load1": "Carga del sistema",
           "req": "Peticiones", "procesos": "Procesos"}
KINDS = {"noisy_neighbor": "Vecino ruidoso", "memory_leak": "Fuga de memoria", "brute_force": "Fuerza bruta",
         "io_saturation": "Saturación de entrada/salida", "legit_surge": "Pico legítimo"}
STATES = {"pendiente_revision": "Pendiente de revisión"}
OFFERS = {"upgrade": "Mejora de plan", "temporal": "Ampliación temporal", "consejo": "Consejo técnico",
          "basico": "Básico", "plus": "Plus", "pro": "Pro", "plus_intermedio": "Plus intermedio", "vps": "VPS",
          "campanas": "Campañas", "esporadico": "Esporádico", "crecimiento": "Crecimiento", "intensivo": "Intensivo"}
COLUMNS = {"method": "Método", "test_days": "Días de test", "saturaciones": "Saturaciones",
           "avisadas_a_tiempo": "Avisadas a tiempo", "no_avisadas": "No avisadas", "falsas_alarmas": "Falsas alarmas",
           "coste_total": "Coste total (€)", "coste_por_servidor_mes": "Coste por servidor y mes (€)",
           "ahorro_vs_umbral_pct": "Ahorro frente al umbral (%)", "account_id": "Cuenta", "cluster": "Cluster",
           "oferta": "Oferta", "plan_actual": "Plan actual", "plan_propuesto": "Plan propuesto",
           "precio_actual": "Precio actual (€)", "precio_propuesto": "Precio propuesto (€)", "estado": "Estado",
           "motivo": "Motivo"}
CHECKS = {"sin_duplicados_servidor_timestamp": "Sin duplicados (servidor, instante)",
          "rejilla_regular_5min": "Rejilla regular de 5 min", "sensor_atascado_cpu": "Sensor de CPU atascado"}
KPIS = {"detectadas": "Tasa de detección", "avisadas_>=10min": "Avisadas con 10 min de antelación",
        "falsas_alarmas_servidor_dia": "Falsas alarmas por servidor y día"}
FEATURES = {"headroom_min": "Margen mínimo hasta saturar", "cpu_unexplained": "CPU no explicada por el tráfico",
            "hour_sin": "Hora del día (seno)", "hour_cos": "Hora del día (coseno)", "weekend": "Fin de semana"}
STEP_MIN = 5  # minutos por muestra: las ventanas de las variables se expresan en muestras

_GENERIC = {**METRICS, **KINDS, **STATES, **OFFERS, **COLUMNS, **CHECKS, **KPIS}


def _metric(name: str) -> str:
    return METRICS.get(name, name.replace("_", " "))


def feature(name: str) -> str:
    """Nombre legible de una variable del modelo (p. ej. cpu_p95 → «CPU (percentil 95)»); si no encaja, se limpia."""
    if name in FEATURES:
        return FEATURES[name]
    if name in METRICS:
        return METRICS[name]
    patterns = (
        (r"z_(\w+?)_m(\d+)", lambda m: f"{_metric(m[1])}: desviación media ({int(m[2]) * STEP_MIN} min)"),
        (r"z_(\w+)", lambda m: f"{_metric(m[1])}: desviación de lo habitual"),
        (r"raw_(\w+)", lambda m: f"{_metric(m[1])} (valor actual)"),
        (r"(\w+?)_p(\d+)", lambda m: f"{_metric(m[1])} (percentil {m[2]})"),
        (r"(\w+?)_max(\d+)", lambda m: f"{_metric(m[1])}: máximo en {int(m[2]) * STEP_MIN} min"),
        (r"(\w+?)_min(\d+)", lambda m: f"{_metric(m[1])}: mínimo en {int(m[2]) * STEP_MIN} min"),
        (r"(\w+?)_d(\d+)", lambda m: f"{_metric(m[1])}: cambio en {int(m[2]) * STEP_MIN} min"),
        (r"(\w+?)_slope", lambda m: f"{_metric(m[1])}: tendencia"),
        (r"(\w+?)_eta", lambda m: f"{_metric(m[1])}: tiempo hasta saturar"),
    )
    for pattern, fmt in patterns:
        if m := re.fullmatch(pattern, name):
            return fmt(m)
    return label(name)


def label(value) -> str:
    """Etiqueta legible de un nombre interno (estado, causa, métrica, indicador). Vacío se queda vacío."""
    text = str(value)
    if text in _GENERIC:
        return _GENERIC[text]
    if (m := re.fullmatch(r"(\w+?)_(\d+)", text)) and m[1] in _GENERIC:  # p. ej. campanas_2 → «Campañas 2»
        return f"{_GENERIC[m[1]]} {m[2]}"
    return text.replace("_", " ").capitalize() if "_" in text else text
