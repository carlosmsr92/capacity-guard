"""Esquemas canónicos de los datos que consume el sistema."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Schema:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    dtypes: dict[str, str]
    ranges: dict[str, tuple[float | None, float | None]]
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)


METRICS = Schema(
    "metrics",
    ("timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"),
    ("event", "event_account"),
    {"timestamp": "datetime", "server": "string", "cpu": "number", "ram": "number",
     "iowait": "number", "net": "number", "load1": "number", "req": "number"},
    {"cpu": (0, 100), "ram": (0, 100), "iowait": (0, 100), "net": (0, None),
     "load1": (0, None), "req": (0, None)},
    {
        "timestamp": ("time", "datetime", "fecha", "ts"),
        "server": ("host", "hostname", "servidor"),
        "cpu": ("cpu_percent", "cpu_usage", "%cpu", "percentcpu", "cpu_pct"),
        "ram": ("mem", "memory_percent", "ram_percent", "ram_pct", "memoria"),
        "iowait": ("io_wait", "iowait_percent"),
        "net": ("network", "net_mbps", "network_mbps"),
        "load1": ("load", "load_1"),
        "req": ("requests", "rpm", "peticiones"),
    },
)

ACCOUNTS = Schema("accounts", ("account_id", "server", "plan"), ("alta", "sector", "archetype"),
                  {"account_id": "string", "server": "string", "plan": "string",
                   "alta": "datetime", "sector": "string", "archetype": "string"}, {}, {
                      "account_id": ("account", "cliente", "customer_id"),
                      "server": ("host", "hostname", "servidor"),
                      "plan": ("tier", "paquete"),
                  })
USAGE = Schema("usage", ("timestamp", "account_id", "cpu", "ram", "io", "procesos"),
               ("minutos_limitado",), {"timestamp": "datetime", "account_id": "string",
               "cpu": "number", "ram": "number", "io": "number", "procesos": "number",
               "minutos_limitado": "number"}, {}, {})
PLANS = Schema("plans", ("plan", "precio_mes", "cpu_cores", "ram_mb", "io_mbps", "procesos"),
               (), {"plan": "string", "precio_mes": "number", "cpu_cores": "number",
                    "ram_mb": "number", "io_mbps": "number", "procesos": "number"}, {}, {})
