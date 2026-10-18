"""Simulador de métricas de servidores de hosting compartido.

Genera series de CPU, RAM, iowait, red, load y peticiones/min con:
  * estacionalidad diaria y semanal (más tráfico en horas de oficina, menos en fin de semana),
  * una ventana de backup nocturno (iowait alto pero *normal*),
  * incidentes reales etiquetados (vecino ruidoso, fuga de memoria, fuerza bruta, saturación de disco),
  * picos legítimos de tráfico (campañas) que NO son incidentes pero pueden saturar un servidor.

Acoplamiento con las cuentas de clientes: cada servidor aloja `accounts_per_server` cuentas con un plan (límites
estilo LVE) y un arquetipo latente. La carga base del servidor es la suma del uso de CPU de sus cuentas dividida
por la capacidad esperada de un servidor típico (escala fija para toda la flota), y cada evento se atribuye a una cuenta (`event_account`), cuyo uso
registra la huella del evento por encima de su límite.

Todo es sintético y reproducible por semilla. Las etiquetas de verdad-terreno permiten medir
de forma honesta cuántos incidentes se detectan y con cuánta antelación.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .accounts import RESOURCES, account_demand, make_accounts, plan_limits, plans_frame

STEP_MIN = 5
RAM_MB_PER_CORE = 4096.0  # RAM física del servidor por core
IO_MBPS_PER_POINT = 2.0  # MB/s de disco que equivalen a 1 punto de iowait
SAT_LIMITS = {"cpu": 90.0, "ram": 95.0, "iowait": 30.0}
INCIDENT_TYPES = ("noisy_neighbor", "memory_leak", "brute_force", "io_saturation")
BENIGN_TYPES = ("legit_surge",)
KINDS = INCIDENT_TYPES + BENIGN_TYPES
KIND_PROBS = (0.22, 0.18, 0.20, 0.20, 0.20)
DURATION_H = {
    "noisy_neighbor": (1.0, 3.0),
    "memory_leak": (6.0, 12.0),
    "brute_force": (0.5, 1.5),
    "io_saturation": (1.0, 2.0),
    "legit_surge": (1.0, 3.0),
}


@dataclass
class SimConfig:
    days: int = 45
    n_servers: int = 8
    seed: int = 7
    start: str = "2026-06-01"  # lunes
    gap_hours: tuple = (10.0, 48.0)  # tiempo entre eventos de un mismo servidor
    accounts_per_server: int = 40


@dataclass
class World:
    metrics: pd.DataFrame
    accounts: pd.DataFrame
    usage_hourly: pd.DataFrame
    plans: pd.DataFrame


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    return lfilter([1.0], [1.0, -phi], rng.normal(0.0, sigma, n))


def _envelope(n: int, start: int, dur: int, up: float = 0.25, down: float = 0.15) -> np.ndarray:
    env = np.zeros(n)
    end = min(n, start + dur)
    d = end - start
    if d <= 0:
        return env
    e = np.ones(d)
    u, w = max(1, int(d * up)), max(1, int(d * down))
    e[:u] = np.linspace(0.05, 1.0, u)
    e[-w:] = np.minimum(e[-w:], np.linspace(1.0, 0.05, w))
    env[start:end] = e
    return env


def _plan_events(rng: np.random.Generator, n: int, cfg: SimConfig) -> list[tuple[str, int, int]]:
    events: list[tuple[str, int, int]] = []
    t = int(rng.integers(24, 200))
    while True:
        kind = str(rng.choice(KINDS, p=KIND_PROBS))
        dur = int(rng.uniform(*DURATION_H[kind]) * 60 / STEP_MIN)
        if t + dur >= n:
            break
        events.append((kind, t, dur))
        t += dur + int(rng.uniform(*cfg.gap_hours) * 60 / STEP_MIN)
    return events


@dataclass
class _ServerSim:
    metrics: pd.DataFrame
    accounts: pd.DataFrame
    usage: dict[str, np.ndarray]  # uso absoluto cuenta × tiempo, con la huella de los eventos atribuidos
    throttled: np.ndarray  # cuenta × tiempo: la demanda superó algún límite del plan
    culprit: np.ndarray  # cuenta × tiempo: 0 o 1 + índice en KINDS del evento atribuido a la cuenta


@lru_cache(maxsize=1)
def cpu_per_account(n: int = 4000, days: int = 14, seed: int = 0) -> float:
    """Cores medios que usa una cuenta típica por unidad de carga del perfil del servidor (Monte Carlo fijo).

    Es la constante que convierte «suma de cores de las cuentas» en la carga del servidor: con la mezcla habitual de
    arquetipos y planes, un servidor medio queda con la misma carga media que el perfil diario/semanal de referencia.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01", periods=days * 24 * 60 // STEP_MIN, freq=f"{STEP_MIN}min")
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    shared = (0.52 + 0.33 * np.sin(2 * np.pi * (hours - 9) / 24)) * np.where(idx.dayofweek.to_numpy() >= 5, 0.78, 1.0)
    accounts = make_accounts("ref", rng, n)
    usage = np.minimum(account_demand(accounts, shared, rng, STEP_MIN)["cpu"], plan_limits(accounts)["cpu"])
    return float(usage.mean(axis=0).mean() / shared.mean())


def _simulate_server(name: str, idx: pd.DatetimeIndex, cfg: SimConfig, rng: np.random.Generator,
                     accounts: pd.DataFrame | None = None) -> _ServerSim:
    n = len(idx)
    hours = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    weekend = idx.dayofweek.to_numpy() >= 5

    # --- parámetros propios del servidor
    cores = int(rng.choice([16, 24, 32]))
    cpu_gain = rng.uniform(48, 62)
    ram_base = rng.uniform(38, 50)
    ram_gain = rng.uniform(8, 14)
    io_base = rng.uniform(1.0, 2.5)
    net_gain = rng.uniform(400, 900)  # Mbps con carga 1.0
    req_gain = rng.uniform(2500, 6000)  # peticiones/min con carga 1.0
    backup_hour = int(rng.integers(2, 5))
    ram_mb = cores * RAM_MB_PER_CORE

    profile = 0.52 + 0.33 * np.sin(2 * np.pi * (hours - 9) / 24)
    shared = profile * np.where(weekend, 0.78, 1.0) * np.exp(_ar1(rng, n, 0.9, 0.035))

    # --- cuentas: demanda -> uso limitado por el plan -> carga base del servidor = suma de usos
    if accounts is None:
        accounts = make_accounts(name, rng, cfg.accounts_per_server)
    demand = account_demand(accounts, shared, rng, STEP_MIN)
    limits = plan_limits(accounts)
    usage = {r: np.minimum(demand[r], limits[r]) for r in RESOURCES}
    throttled = np.any([demand[r] > limits[r] * (1 + 1e-9) for r in RESOURCES], axis=0)
    # carga = suma del uso de CPU de las cuentas / capacidad esperada de un servidor típico (escala fija de flota):
    # un servidor con clientes más intensivos está de verdad más cargado, no solo tiene otra forma de curva
    load = np.clip(usage["cpu"].sum(axis=0) / (len(accounts) * cpu_per_account()), 0.05, None)

    # --- eventos, cada uno atribuido a una cuenta concreta
    arch = accounts["archetype"].to_numpy()
    pools = {True: np.flatnonzero(arch == "estacional_campanas"), False: np.flatnonzero(arch == "causa_tecnica")}
    ids = accounts["account_id"].to_numpy()
    req_mult = np.ones(n)
    waste = np.ones(n)
    cpu_extra = np.zeros(n)
    ram_extra = np.zeros(n)
    io_extra = np.zeros(n)
    label = np.array([""] * n, dtype=object)
    owner = np.array([""] * n, dtype=object)
    culprit = np.zeros((len(arch), n), dtype=np.int8)
    for kind, s, d in _plan_events(rng, n, cfg):
        pool = pools[kind in BENIGN_TYPES]
        a = int(rng.choice(pool if len(pool) else np.arange(len(arch))))
        env = _envelope(n, s, d)
        e = min(n, s + d)
        dc, dr, di, dq, dw = (np.zeros(n) for _ in range(5))
        if kind == "noisy_neighbor":
            dc = rng.uniform(32, 52) * env
            di = rng.uniform(6, 14) * env
        elif kind == "memory_leak":
            ramp = np.zeros(n)
            ramp[s:e] = np.linspace(0.0, 1.0, e - s)
            dr = rng.uniform(35, 55) * ramp
        elif kind == "brute_force":
            dq = rng.uniform(3, 7) * env
            dw[s:e] = 0.25
        elif kind == "io_saturation":
            di = rng.uniform(30, 55) * env
            dc = rng.uniform(3, 8) * env
        elif kind == "legit_surge":
            dq = rng.uniform(0.8, 1.8) * env
        cpu_extra += dc
        ram_extra += dr
        io_extra += di
        req_mult += dq
        waste += dw
        label[s:e] = kind
        owner[s:e] = ids[a]
        culprit[a, s:e] = KINDS.index(kind) + 1
        # huella en la cuenta culpable: lo que el evento añade al servidor, en unidades del plan
        extra = {"cpu": (dc + cpu_gain * load * (dq + dw)) / 100 * cores, "ram": (dr + ram_gain * load * dq) / 100 * ram_mb,
                 "io": di * IO_MBPS_PER_POINT, "procesos": usage["procesos"][a] * (dq + dw)}
        for r in RESOURCES:
            if kind in BENIGN_TYPES:  # tráfico legítimo: el límite del plan lo estrangula (señal comercial)
                throttled[a] |= usage[r][a] + extra[r] > limits[r][a, 0]
                usage[r][a] = np.minimum(usage[r][a] + extra[r], limits[r][a, 0])
            else:  # incidente: escapa al aislamiento por cuenta y supera el límite (firma técnica)
                usage[r][a] += extra[r]

    eff = load * req_mult
    req = req_gain * eff * np.exp(rng.normal(0, 0.05, n))
    cpu = 6 + cpu_gain * eff * waste + cpu_extra + rng.normal(0, 2.0, n)
    drift = np.clip(_ar1(rng, n, 0.995, 0.15), -4, 4)
    ram = ram_base + ram_gain * eff + drift + ram_extra + rng.normal(0, 0.4, n)
    backup = ((hours >= backup_hour) & (hours < backup_hour + 1)).astype(float)
    iowait = io_base + 2.5 * eff + 7.0 * backup + io_extra + np.abs(rng.normal(0, 0.6, n))
    net = net_gain * eff + rng.normal(0, 15, n)
    cpu = np.clip(cpu, 0, 100)
    load1 = cpu / 100 * cores * 0.9 + iowait / 10 + np.abs(rng.normal(0, 0.3, n))

    metrics = pd.DataFrame(
        {
            "timestamp": idx,
            "server": name,
            "cpu": cpu,
            "ram": np.clip(ram, 0, 100),
            "iowait": np.clip(iowait, 0, 100),
            "net": np.clip(net, 0, None),
            "load1": load1,
            "req": req,
            "event": label,
            "event_account": owner,
        }
    )
    return _ServerSim(metrics, accounts, usage, throttled, culprit)


def _hourly_usage(sim: _ServerSim, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Agrega el uso de cada cuenta a horas operando sobre las matrices (sin pasar por formato largo)."""
    acc = sim.accounts
    n_acc, per_h = len(acc), 60 // STEP_MIN
    hours = len(idx) // per_h

    def by_hour(x: np.ndarray) -> np.ndarray:
        return x[:, : hours * per_h].reshape(n_acc, hours, per_h)

    limits = plan_limits(acc)
    out = {
        "timestamp": np.tile(idx[::per_h][:hours].to_numpy(), n_acc),
        "account_id": np.repeat(acc["account_id"].to_numpy(), hours),
        "server": np.repeat(acc["server"].to_numpy(), hours),
        "plan": np.repeat(acc["plan"].to_numpy(), hours),
    }
    for r in RESOURCES:
        out[r] = by_hour(sim.usage[r]).mean(axis=2).ravel()
        out[f"{r}_limit"] = np.repeat(limits[r][:, 0], hours)
    out["minutos_limitado"] = (by_hour(sim.throttled).sum(axis=2) * STEP_MIN).ravel()
    names = np.array(("",) + KINDS, dtype=object)
    out["incidente"] = names[by_hour(sim.culprit).max(axis=2).ravel()]
    return pd.DataFrame(out)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # cualquier etiqueta no benigna cuenta como incidente: un CSV real puede traer sus propias causas (p. ej. "ddos")
    df["is_incident"] = df["event"].ne("") & ~df["event"].isin(BENIGN_TYPES)
    df["is_benign_event"] = df["event"].isin(BENIGN_TYPES)
    df["sat_now"] = (
        (df["cpu"] >= SAT_LIMITS["cpu"]) | (df["ram"] >= SAT_LIMITS["ram"]) | (df["iowait"] >= SAT_LIMITS["iowait"])
    )
    return df


def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    """Solo las métricas de servidor (derivadas de las cuentas, sin construir las tablas de uso)."""
    return simulate_world(cfg, with_usage=False).metrics


def simulate_world(cfg: SimConfig | None = None, with_usage: bool = True) -> World:
    """Simula servidores, cuentas, uso horario por cuenta y catálogo de planes."""
    cfg = cfg or SimConfig()
    n = cfg.days * 24 * 60 // STEP_MIN
    idx = pd.date_range(cfg.start, periods=n, freq=f"{STEP_MIN}min")
    root = np.random.SeedSequence(cfg.seed)
    metrics, accounts, usage = [], [], []
    for i, child in enumerate(root.spawn(cfg.n_servers)):
        sim = _simulate_server(f"web-{i + 1:02d}", idx, cfg, np.random.default_rng(child))
        metrics.append(sim.metrics)
        accounts.append(sim.accounts)
        if with_usage:
            usage.append(_hourly_usage(sim, idx))
    usage_hourly = pd.concat(usage, ignore_index=True) if usage else pd.DataFrame()
    return World(add_labels(pd.concat(metrics, ignore_index=True)), pd.concat(accounts, ignore_index=True),
                 usage_hourly, plans_frame())
