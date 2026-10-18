"""Ingesta: simula un feed sucio, lo valida, lo repara y demuestra con la verdad-terreno que la reparación es fiable."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import METRICS, repair, validate
from .simulator import add_labels


def make_dirty_feed(df: pd.DataFrame, seed: int = 0, rate: float = 0.003) -> pd.DataFrame:
    """Inyecta fallos típicos de un agente de monitorización: duplicados, nulos, valores imposibles y huecos cortos."""
    rng = np.random.default_rng(seed)
    d = df[["timestamp", "server", *METRICS]].copy().reset_index(drop=True)
    n = len(d)
    interior = np.flatnonzero(d.groupby("server").cumcount(ascending=False).to_numpy() > 5)
    interior = interior[d.groupby("server").cumcount().to_numpy()[interior] > 5]
    drop = set()
    for s in rng.choice(interior, int(n * rate / 3), replace=False):
        drop.update(range(s, s + int(rng.integers(1, 4))))
    d.loc[rng.choice(interior, int(n * rate), replace=False), rng.choice(METRICS)] = np.nan
    bad = rng.choice(interior, int(n * rate / 2), replace=False)
    d.loc[bad, "cpu"] = rng.choice([-5.0, 130.0], len(bad))
    dups = d.iloc[rng.choice(interior, int(n * rate), replace=False)]
    d = d.drop(index=sorted(drop & set(d.index)))
    return pd.concat([d, dups], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ingest(truth: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    dirty = make_dirty_feed(truth, seed)
    before = validate(dirty)
    fixed, stats = repair(dirty)
    after = validate(fixed, long_gap_severity="warning")
    labels = truth[["server", "timestamp", *[c for c in ("event", "event_account") if c in truth]]]
    clean = fixed.drop(columns=["imputed"]).merge(labels, on=["server", "timestamp"], how="left")
    for c in ("event", "event_account"):
        if c in clean:
            clean[c] = clean[c].fillna("")
    imputed = fixed.set_index(["server", "timestamp"])["imputed"]
    clean = add_labels(clean.sort_values(["server", "timestamp"]).reset_index(drop=True))
    # ¿cuánto se equivoca la imputación? comparación con los valores verdaderos que ocultamos
    key = ["server", "timestamp"]
    t = truth.set_index(key)[METRICS]
    c = clean.set_index(key)[METRICS]
    mask = imputed.reindex(c.index).fillna(False).to_numpy(dtype=bool)
    mae = {m: float(np.abs(c.loc[mask, m].to_numpy() - t.reindex(c.index).loc[mask, m].to_numpy()).mean())
           for m in METRICS} if mask.any() else {}
    info = {"rows_truth": int(len(truth)), "rows_dirty": int(len(dirty)), "rows_clean": int(len(clean)),
            "quality_before": before.to_dict(), "quality_after": after.to_dict(), "repair": stats,
            "imputed_rows": int(mask.sum()), "imputation_mae": mae,
            "imputation_mae_vs_metric_std": {m: mae[m] / float(truth[m].std()) for m in mae}}
    return clean, info
