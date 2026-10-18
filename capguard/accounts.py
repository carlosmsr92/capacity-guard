"""Catálogo de planes y generación reproducible de cuentas y de su demanda (simulador acoplado).

Cada cuenta tiene un plan con límites estilo LVE (cores, MB de RAM, MB/s de disco, procesos) y un arquetipo latente
que fija cómo evoluciona su demanda. El uso es `min(demanda, límite)`: lo que supera el límite se estrangula y se
cuenta como minutos limitados. La carga del servidor se calcula después a partir de la suma de estos usos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

RESOURCES = ("cpu", "ram", "io", "procesos")
LIMIT_FIELD = {"cpu": "cpu_cores", "ram": "ram_mb", "io": "io_mbps", "procesos": "procesos"}


@dataclass(frozen=True)
class Plan:
    precio_mes: float
    cpu_cores: float
    ram_mb: float
    io_mbps: float
    procesos: float


PLANS = {
    "basico": Plan(9.0, 1.0, 1024.0, 10.0, 50.0),
    "plus": Plan(18.0, 2.0, 2048.0, 20.0, 100.0),
    "plus_intermedio": Plan(27.0, 3.0, 3072.0, 30.0, 150.0),
    "pro": Plan(42.0, 4.0, 4096.0, 50.0, 250.0),
    "vps": Plan(85.0, 8.0, 8192.0, 100.0, 500.0),
}

ARCHETYPES = {
    "holgado": 0.65,
    "crecimiento_sostenido": 0.09,
    "estacional_campanas": 0.08,
    "intensivo_constante": 0.07,
    "picos_esporadicos": 0.08,
    "causa_tecnica": 0.03,
}
# Utilización típica en hora punta (fracción del límite) de cada arquetipo
LEVEL = {
    "holgado": (0.10, 0.40),
    "crecimiento_sostenido": (0.45, 0.60),
    "estacional_campanas": (0.40, 0.55),
    "intensivo_constante": (0.85, 1.00),
    "picos_esporadicos": (0.30, 0.45),
    "causa_tecnica": (0.20, 0.40),
}


def plans_frame() -> pd.DataFrame:
    return pd.DataFrame([{"plan": name, **spec.__dict__} for name, spec in PLANS.items()])


def make_accounts(server: str, rng: np.random.Generator, n: int = 40) -> pd.DataFrame:
    """Crea cuentas seudónimas y sus arquetipos latentes para un servidor.

    Cada servidor tiene al menos una cuenta `causa_tecnica` y una `estacional_campanas`, a las que se atribuyen
    los incidentes y los picos legítimos de tráfico.
    """
    archetypes = rng.choice(list(ARCHETYPES), size=n, p=list(ARCHETYPES.values())).astype(object)
    for needed in ("causa_tecnica", "estacional_campanas"):
        if n >= 2 and needed not in archetypes:
            free = np.flatnonzero(archetypes == "holgado")
            archetypes[int(rng.choice(free if len(free) else np.arange(n)))] = needed
    plans = rng.choice(["basico", "plus", "pro"], size=n, p=[0.55, 0.35, 0.10])
    return pd.DataFrame({
        "account_id": [f"{server}-acct-{i + 1:03d}" for i in range(n)],
        "server": server,
        "plan": plans.astype(str),
        "alta": "2026-01-01",
        "sector": np.where(archetypes == "estacional_campanas", "ecommerce", "servicios"),
        "archetype": archetypes.astype(str),
    })


def plan_limits(accounts: pd.DataFrame) -> dict[str, np.ndarray]:
    """Límites del plan de cada cuenta, con forma (cuentas, 1) para operar contra matrices cuenta × tiempo."""
    return {r: np.array([getattr(PLANS[p], LIMIT_FIELD[r]) for p in accounts["plan"]], dtype=float)[:, None]
            for r in RESOURCES}


def account_demand(accounts: pd.DataFrame, shared: np.ndarray, rng: np.random.Generator,
                   step_min: int = 5) -> dict[str, np.ndarray]:
    """Demanda absoluta (unidades del plan) por recurso, con forma cuenta × tiempo.

    `shared` es el patrón común del servidor (perfil diario y semanal); cada cuenta lo escala con su arquetipo,
    su mezcla de recursos y su propio ruido AR(1).
    """
    arch = accounts["archetype"].to_numpy()
    n, t = len(arch), len(shared)
    s = shared / np.quantile(shared, 0.95)  # ~1 en hora punta
    time = np.arange(t) / max(1, t - 1)
    steps_day = 24 * 60 // step_min

    level = np.array([rng.uniform(*LEVEL[a]) for a in arch])[:, None]
    util = level * s[None, :]
    flat = arch == "intensivo_constante"
    util[flat] = level[flat] * (0.85 + 0.15 * s[None, :])
    grow = arch == "crecimiento_sostenido"
    util[grow] *= 0.6 + 0.9 * time[None, :]
    for i in np.flatnonzero(arch == "estacional_campanas"):
        for _ in range(int(rng.integers(2, 4))):  # 2-3 campañas de 3-6 días
            start, dur = int(rng.integers(0, max(1, t - steps_day))), int(rng.uniform(3, 6) * steps_day)
            util[i, start:start + dur] *= 2.0
    for i in np.flatnonzero(arch == "picos_esporadicos"):
        starts = np.flatnonzero(rng.random(t) < 0.004)
        burst = np.zeros(t)
        for st in starts:
            burst[st:st + int(rng.integers(6, 24))] = 1.0
        util[i] *= 1.0 + 1.6 * burst
    util *= np.exp(lfilter([1.0], [1.0, -0.9], rng.normal(0.0, 0.04, (n, t)), axis=1))

    # Mezcla de recursos: cada cuenta no holgada tiene un recurso dominante
    mix = rng.uniform(0.35, 0.85, (n, len(RESOURCES)))
    dominant = rng.integers(0, len(RESOURCES), n)
    mix[np.arange(n), dominant] = 1.0
    mix[arch == "holgado", 0] = 1.0
    limits = plan_limits(accounts)
    out = {}
    for j, r in enumerate(RESOURCES):
        u = util * mix[:, [j]]
        if r == "ram":  # la memoria sigue menos al tráfico que la CPU
            u = 0.6 * u + 0.4 * level * mix[:, [j]]
        out[r] = np.clip(u, 0, None) * limits[r]
    return out
