"""Ingeniería de variables (todas causales: solo usan el pasado de cada servidor)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_COLS = ["cpu", "ram", "iowait", "net", "load1", "req"]
_ABS_FLOOR = {"cpu": 1.0, "ram": 0.7, "iowait": 0.4, "load1": 0.3}


def _how(ts: pd.Series) -> np.ndarray:
    return (ts.dt.dayofweek * 24 + ts.dt.hour).to_numpy()


class FeatureBuilder:
    """Aprende el comportamiento 'normal' por servidor y hora-de-la-semana y lo usa para normalizar.

    * baseline: mediana de cada métrica por (servidor, hora de la semana) -> captura ciclos diarios,
      el fin de semana y el backup nocturno como comportamiento normal.
    * escala: MAD de los residuos (robusta frente a los incidentes presentes en el entrenamiento).
    * recta cpu~req por servidor: "CPU no explicada por el tráfico" (firma del vecino ruidoso).
    * perfil de flota (mediana entre servidores) para puntuar servidores que no estaban en el entrenamiento.
    """

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        d = df.assign(how=_how(df["timestamp"]))
        self.baseline_ = d.groupby(["server", "how"])[BASE_COLS].median()
        self.scale_: dict[str, np.ndarray] = {}
        self.cpu_line_: dict[str, tuple[float, float, float]] = {}
        for server, g in d.groupby("server"):
            base = self.baseline_.loc[server].reindex(g["how"]).to_numpy()
            res = g[BASE_COLS].to_numpy() - base
            mad = np.median(np.abs(res - np.median(res, axis=0)), axis=0) * 1.4826
            floor = np.array(
                [_ABS_FLOOR.get(c, 0.05 * float(np.median(g[c]))) for c in BASE_COLS]
            )
            self.scale_[server] = np.maximum(mad, floor)
            req, cpu = g["req"].to_numpy(), g["cpu"].to_numpy()
            lo, hi = np.percentile(req, [5, 95])
            m = (req >= lo) & (req <= hi)
            a, b = np.polyfit(req[m], cpu[m], 1)
            r = cpu - (a * req + b)
            s = float(np.median(np.abs(r - np.median(r))) * 1.4826)
            self.cpu_line_[server] = (float(a), float(b), max(s, 0.5))
        self.fleet_baseline_ = self.baseline_.groupby(level="how").median()
        self.fleet_scale_ = np.median(np.vstack(list(self.scale_.values())), axis=0)
        self.fleet_cpu_line_ = tuple(float(v) for v in np.median(np.array(list(self.cpu_line_.values())), axis=0))
        return self

    def unknown_servers(self, df: pd.DataFrame) -> list[str]:
        return sorted(set(df["server"].unique()) - set(self.scale_))

    def _params(self, server: str) -> tuple[pd.DataFrame, np.ndarray, tuple[float, float, float]]:
        """Parámetros del servidor; si no se vio al entrenar, los de la flota (aproximación, menos precisa)."""
        if server in self.scale_:
            return self.baseline_.loc[server], self.scale_[server], self.cpu_line_[server]
        return self.fleet_baseline_, self.fleet_scale_, self.fleet_cpu_line_

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for server, g in df.groupby("server", sort=False):
            how = _how(g["timestamp"])
            baseline, scale, (a, b, s) = self._params(server)
            base = baseline.reindex(how).to_numpy()
            z = (g[BASE_COLS].to_numpy() - base) / scale
            f = pd.DataFrame(z, columns=[f"z_{c}" for c in BASE_COLS], index=g.index)
            f["cpu_unexplained"] = (g["cpu"] - (a * g["req"] + b)) / s
            for c in ("cpu", "ram", "iowait", "req"):
                f[f"z_{c}_m6"] = f[f"z_{c}"].rolling(6, min_periods=1).mean()
            f["ram_slope"] = ((g["ram"] - g["ram"].shift(12)) / 12).fillna(0.0)
            cpu3, io3 = g["cpu"].rolling(3, min_periods=1).mean(), g["iowait"].rolling(3, min_periods=1).mean()
            f["cpu_slope"] = ((cpu3 - cpu3.shift(6)) / 6).fillna(0.0)
            f["io_slope"] = ((io3 - io3.shift(6)) / 6).fillna(0.0)
            f["ram_eta"] = np.clip((95.0 - g["ram"]) / np.maximum(f["ram_slope"], 0.01), 0, 600)
            f["cpu_d3"] = ((g["cpu"] - g["cpu"].shift(3)) / 3).fillna(0.0)
            f["io_d3"] = ((g["iowait"] - g["iowait"].shift(3)) / 3).fillna(0.0)
            f["req_d3"] = (f["z_req"] - f["z_req"].shift(3)).fillna(0.0)
            f["headroom_min"] = np.minimum.reduce(
                [(90.0 - g["cpu"]) / 90.0, (95.0 - g["ram"]) / 95.0, (30.0 - g["iowait"]) / 30.0])
            f["cpu_max6"] = g["cpu"].rolling(6, min_periods=1).max()
            f["io_max6"] = g["iowait"].rolling(6, min_periods=1).max()
            for c in ("cpu", "ram", "iowait", "load1", "net"):
                f[f"raw_{c}"] = g[c]
            h = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60
            f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
            f["weekend"] = (g["timestamp"].dt.dayofweek >= 5).astype(float)
            out.append(f)
        return pd.concat(out).loc[df.index]


ANOMALY_FEATURES = (
    [f"z_{c}" for c in BASE_COLS]
    + ["cpu_unexplained", "z_cpu_m6", "z_ram_m6", "z_iowait_m6", "z_req_m6", "ram_slope", "io_slope"]
)
FORECAST_FEATURES = ANOMALY_FEATURES + [
    "cpu_slope", "ram_eta", "cpu_max6", "io_max6", "cpu_d3", "io_d3", "req_d3", "headroom_min",
    "raw_cpu", "raw_ram", "raw_iowait", "raw_load1", "raw_net",
    "hour_sin", "hour_cos", "weekend",
]


def future_saturation(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    """1 si habrá saturación en alguna de las próximas `horizon` muestras (30 min con horizon=6)."""
    parts = []
    for _, g in df.groupby("server", sort=False):
        s = g["sat_now"].astype(float)
        fut = s.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1).fillna(0.0)
        parts.append(fut)
    return pd.concat(parts).loc[df.index].astype(int).rename("sat_future")
