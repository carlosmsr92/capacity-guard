import numpy as np
import pandas as pd
import pytest

from capguard.ingest import ingest
from capguard.quality import repair, validate
from capguard.simulator import SimConfig, simulate

CFG = SimConfig(days=10, n_servers=3, seed=1)


@pytest.fixture(scope="module")
def clean():
    return simulate(CFG)


def failed(df, **kw):
    return {c.name for c in validate(df, **kw).failed()}


def test_clean_data_passes(clean):
    assert validate(clean).ok and not failed(clean)


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: pd.concat([d, d.iloc[:4]]), "sin_duplicados_servidor_timestamp"),
    (lambda d: d.assign(cpu=d["cpu"].where(d.index != 50, np.nan)), "sin_nulos_ni_infinitos"),
    (lambda d: d.assign(ram=d["ram"].where(d.index != 50, 140.0)), "valores_en_rango"),
    (lambda d: d.assign(iowait=d["iowait"].where(d.index != 50, -3.0)), "valores_en_rango"),
    (lambda d: d.drop(index=d.index[300:302]), "huecos_cortos_reparables"),
    (lambda d: d.drop(index=d.index[300:310]), "huecos_largos"),
    (lambda d: d.sample(frac=1.0, random_state=0), "orden_temporal_por_servidor"),
    (lambda d: d.assign(cpu=np.where((d["server"] == "web-01") & (d.index < 200), 33.3, d["cpu"])), "sensor_atascado_cpu"),
])
def test_each_corruption_is_detected(clean, mutate, expected):
    assert expected in failed(mutate(clean.copy()))


def test_missing_columns_is_an_error(clean):
    rep = validate(clean.drop(columns=["cpu"]))
    assert not rep.ok
    with pytest.raises(ValueError):
        rep.raise_if_failed()


def test_cpu_pegged_at_100_is_not_a_broken_sensor(clean):
    d = clean.copy()
    d.loc[d.index[:60], "cpu"] = 100.0
    assert "sensor_atascado_cpu" not in failed(d)


def test_repair_fixes_short_problems_but_never_invents_long_gaps(clean):
    bad = pd.concat([clean, clean.iloc[:5]])
    bad.loc[bad.index[100], "cpu"] = np.nan
    bad = bad.drop(index=bad.index[1000:1010])  # hueco largo (10 muestras)
    fixed, stats = repair(bad)
    assert stats["duplicates_removed"] == 5
    assert "huecos_largos" in failed(fixed, long_gap_severity="warning")  # sigue ahí: no se rellena
    assert validate(fixed, long_gap_severity="warning").ok
    assert fixed["imputed"].sum() >= 1
    assert fixed[["cpu", "ram"]].notna().all().all()


def test_ingest_end_to_end_is_accurate(clean):
    out, info = ingest(clean, seed=0)
    assert not info["quality_before"]["ok"] and info["quality_after"]["ok"]
    assert info["rows_truth"] - info["rows_clean"] <= 0.001 * info["rows_truth"]  # ≤0,1 % irrecuperable
    assert all(v < 0.25 for v in info["imputation_mae_vs_metric_std"].values())  # imputación < 25 % de la std
    assert out["sat_now"].dtype == bool and "event" in out.columns
    # fuera de lo imputado, los valores coinciden con la verdad
    m = clean.merge(out, on=["server", "timestamp"], suffixes=("", "_c"))
    assert (m["cpu"] == m["cpu_c"]).mean() > 0.99
