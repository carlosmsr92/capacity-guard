"""Pipeline completo: simular -> features -> entrenar -> calibrar umbrales -> evaluar en test."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .evaluation import alerts_from_score, evaluate, incident_windows, saturation_windows, tune_threshold
from .features import FeatureBuilder, future_saturation
from .models import IsolationDetector, SaturationForecaster, SeasonalZ, ThresholdDetector
from .simulator import STEP_MIN, SimConfig, simulate

TASKS = {
    "A": {"title": "Detección de incidentes", "pre": 0},
    "B": {"title": "Aviso temprano de saturación (30 min)", "pre": 12},
}


@dataclass
class Artifacts:
    cfg: SimConfig
    df: pd.DataFrame
    feats: pd.DataFrame
    splits: dict
    scores: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    alerts: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    windows: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    alerts_all_a: pd.Series | None = None


def data_days(df: pd.DataFrame) -> float:
    """Duración real de los datos en días (incluye el último intervalo de 5 min)."""
    span = df["timestamp"].max() - df["timestamp"].min() + pd.Timedelta(minutes=STEP_MIN)
    return span / pd.Timedelta(days=1)


def split_masks(df: pd.DataFrame, bounds: tuple | None = None) -> dict[str, pd.Series]:
    """Partición temporal. bounds=(fin_train, fin_val, fin_test) en días desde el inicio (fin_test opcional).

    Por defecto 60 % / 15 % / 25 % de la duración real de los datos (no de la configuración del simulador).
    """
    t0 = df["timestamp"].min()
    ts = df["timestamp"]
    if bounds is None:
        days = data_days(df)
        bounds = (days * 0.60, days * 0.75, None)
    t_tr, t_va = t0 + pd.Timedelta(days=bounds[0]), t0 + pd.Timedelta(days=bounds[1])
    test = ts >= t_va
    if len(bounds) > 2 and bounds[2] is not None:
        test &= ts < t0 + pd.Timedelta(days=bounds[2])
    return {"train": ts < t_tr, "val": (ts >= t_tr) & (ts < t_va), "test": test}


def run(cfg: SimConfig | None = None, budget: float = 0.5, horizon: int = 6, k: int = 2,
        df: pd.DataFrame | None = None, bounds: tuple | None = None) -> Artifacts:
    cfg = cfg or SimConfig()
    df = simulate(cfg) if df is None else df
    df = df.assign(sat_future=future_saturation(df, horizon))
    masks = split_masks(df, bounds)

    fb = FeatureBuilder().fit(df[masks["train"]])
    feats = fb.transform(df)

    iso = IsolationDetector().fit(feats[masks["train"]])
    gbm = SaturationForecaster().fit(feats[masks["train"]], df.loc[masks["train"], "sat_future"],
                                     df.loc[masks["train"], "sat_now"])
    methods = {"threshold": ThresholdDetector(), "zscore": SeasonalZ(), "iforest": iso, "gbm": gbm}

    art = Artifacts(cfg, df, feats, {n: m for n, m in masks.items()})
    art.models = {"feature_builder": fb, **methods}
    parts = {n: df[m] for n, m in masks.items()}
    art.windows = {
        (task, n): (incident_windows if task == "A" else saturation_windows)(parts[n])
        for task in TASKS for n in ("val", "test")
    }
    for name, m in methods.items():
        art.scores[name] = pd.Series(m.score(df, feats), index=df.index)

    results: dict = {"config": cfg.__dict__ | {"budget_fa_per_server_day": budget, "horizon_min": horizon * 5},
                     "tasks": {}}
    for task, spec in TASKS.items():
        rows = {}
        for name, m in methods.items():
            sc_val, sc_test = art.scores[name][masks["val"]], art.scores[name][masks["test"]]
            if name == "threshold":
                thr = ThresholdDetector.fixed_thr
            else:
                thr = tune_threshold(parts["val"], sc_val, art.windows[(task, "val")], spec["pre"], budget, k)
            alert = alerts_from_score(parts["test"], sc_test, thr, k)
            art.thresholds[(task, name)], art.alerts[(task, name)] = thr, alert
            rows[name] = evaluate(parts["test"], alert, art.windows[(task, "test")], spec["pre"]) | {
                "method": m.name, "threshold": thr}
        results["tasks"][task] = {"title": spec["title"], "methods": rows}

    # Calidad por muestra del pronosticador (referencia clásica)
    from sklearn.metrics import average_precision_score, roc_auc_score
    t = parts["test"]
    ok = ~t["sat_now"].to_numpy()
    y, p = t["sat_future"].to_numpy()[ok], art.scores["gbm"][masks["test"]].to_numpy()[ok]
    results["gbm_sample_metrics"] = {"roc_auc": float(roc_auc_score(y, p)),
                                     "average_precision": float(average_precision_score(y, p)),
                                     "positive_rate": float(y.mean())}
    results["dataset"] = {
        "rows": int(len(df)), "servers": int(df["server"].nunique()), "days": round(data_days(df), 2),
        "incidents_test": results["tasks"]["A"]["methods"]["gbm"]["n_events"],
        "saturation_episodes_test": results["tasks"]["B"]["methods"]["gbm"]["n_events"],
    }
    art.alerts_all_a = alerts_from_score(df, art.scores["iforest"], art.thresholds[("A", "iforest")], k)
    art.results = results
    return art


def by_kind(art: Artifacts, task: str, method: str) -> pd.DataFrame:
    """Recall / aviso temprano / adelanto mediano por tipo de evento que originó el episodio."""
    te = art.df[art.splits["test"]]
    groups = {s: g.reset_index(drop=True) for s, g in te.groupby("server", sort=False)}
    rows = []
    for server, start, lead in art.results["tasks"][task]["methods"][method]["per_event"]:
        ev = groups[server]["event"]
        recent = ev.iloc[max(0, start - 12): start + 1]
        recent = recent[recent != ""]
        rows.append((recent.iloc[-1] if len(recent) else "sin_causa_etiquetada", lead))
    d = pd.DataFrame(rows, columns=["kind", "lead"])
    return d.groupby("kind").apply(
        lambda x: pd.Series({"n": len(x), "recall": x["lead"].notna().mean(),
                             "early_rate": (x["lead"] >= 10).mean(), "median_lead_min": x["lead"].median()}),
        include_groups=False)


def benchmark(seeds: list[int], **kw) -> pd.DataFrame:
    """Repite todo el experimento con varios 'mundos' simulados distintos y agrega media y desviación."""
    rows = []
    for sd in seeds:
        art = run(SimConfig(seed=sd), **kw)
        for task, v in art.results["tasks"].items():
            for m in v["methods"].values():
                rows.append({"seed": sd, "task": task, "method": m["method"], "recall": m["recall"],
                             "early_rate": m["early_rate"], "median_lead_min": m["median_lead_min"],
                             "false_alarms_per_server_day": m["false_alarms_per_server_day"]})
    d = pd.DataFrame(rows)
    return d.groupby(["task", "method"], sort=False).agg(["mean", "std"]).drop(columns="seed")
