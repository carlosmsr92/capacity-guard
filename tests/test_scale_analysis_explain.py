from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capguard import analysis, explain, scale
from capguard.evaluation import incident_windows
from capguard.features import FeatureBuilder
from capguard.simulator import SimConfig, simulate


def test_pandas_duckdb_and_featurebuilder_agree_exactly(tmp_path):
    df = simulate(SimConfig(days=14, n_servers=3, seed=2)).sort_values(["server", "timestamp"]).reset_index(drop=True)
    cut = df["timestamp"].min() + pd.Timedelta(days=8)
    ref = FeatureBuilder().fit(df[df["timestamp"] < cut]).transform(df)
    ref.index = pd.MultiIndex.from_frame(df[["server", "timestamp"]])
    pf = scale.pandas_features(df, cut)
    d = tmp_path
    df[["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"]].to_parquet(f"{d}/part-0000.parquet")
    con, sql = scale.duckdb_features(f"{d}/part-*.parquet", str(cut))
    dk = con.execute(sql).df().set_index(["server", "timestamp"]).sort_index()
    for col in scale.FEATURES:
        a = ref[col].sort_index()
        assert a.notna().sum() > 1000  # no comparamos vacíos con vacíos
        np.testing.assert_allclose(pf[col].reindex(a.index), a, atol=1e-9, equal_nan=True)
        np.testing.assert_allclose(dk[col].reindex(a.index), a, atol=1e-9, equal_nan=True)


def test_scale_benchmark_runs_in_isolated_processes(tmp_path):
    d = tmp_path
    assert scale.generate_parquet_dataset(d, 5, 10, chunk=3) == 5 * 10 * 288
    res = scale.benchmark(d, "2026-06-07")
    assert all("error" not in r for r in res)
    assert res[0]["rows"] == res[1]["rows"] and res[0]["checksum"] == pytest.approx(res[1]["checksum"], rel=1e-6)


METHODS = {  # 10 saturaciones, 2 días-servidor; resultados conocidos para comprobar la contabilidad
    "threshold": {"method": "T", "n_events": 10, "leads_min": [0, 5, 20], "false_alarm_episodes": 1, "server_days": 2.0},
    "gbm": {"method": "G", "n_events": 10, "leads_min": [10, 15, 20, 30, 40, 5], "false_alarm_episodes": 4, "server_days": 2.0},
}


def test_cost_table_arithmetic():
    t = analysis.cost_table(METHODS, {"c_miss": 100, "c_fa": 10, "c_mitigated": 1}).set_index("key")
    assert t.loc["threshold", ["avisadas_a_tiempo", "no_avisadas", "falsas_alarmas"]].tolist() == [1, 9, 1]
    assert t.loc["threshold", "coste_total"] == 1 * 1 + 9 * 100 + 1 * 10
    assert t.loc["gbm", "coste_total"] == 5 * 1 + 5 * 100 + 4 * 10
    assert t.loc["gbm", "ahorro_vs_umbral_pct"] == pytest.approx((1 - 545 / 911) * 100, abs=0.1)


def test_sensitivity_flips_when_false_alarms_get_expensive():
    s = analysis.sensitivity(METHODS, (100,), (1, 500)).set_index("c_fa")["key"]
    assert s[1] == "gbm" and s[500] == "threshold"


def test_eda_counts_match_ground_truth():
    df = simulate(SimConfig(days=14, n_servers=3, seed=4))
    e = analysis.eda(df)
    expected = sum(len(w) for server, g in df.groupby("server") for w in [incident_windows(g.reset_index(drop=True))[server]])
    assert e["n_incidents"] == expected == sum(r["incidentes"] for r in e["by_kind"]) == sum(e["by_hour"])


EV = {"server": "web-01", "timestamp": "2026-07-04 18:05", "risk_30min_pct": 42.7,
      "now": {"cpu": 94.0, "ram": 50.2}, "one_hour_ago": {"cpu": 91.6, "ram": 49.6},
      "diagnosis": {"probability_pct": 87.0}}


@pytest.mark.parametrize("text,ok", [
    ("Riesgo del 43 % y CPU 94.0 en web-01 a las 2026-07-04 18:05", True),
    ("riesgo 42.7", True),
    ("CPU al 99.7 %", False),            # cifra inventada muy parecida a una real
    ("riesgo 42 %", False),              # truncado, no redondeado
    ("412 conexiones", False),
    ("sin números", True),
])
def test_number_guardrail(text, ok):
    assert explain.check_numbers(text, EV)[0] is ok


def _client(text=None, exc=None, stop_reason="end_turn"):
    def create(**kw):
        if exc:
            raise exc
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


FULL_EV = {**EV, "largest_deviations": [{"signal": "iowait %", "mad_units": 3.1}],
           "now": {"cpu": 94.0, "ram": 50.2, "iowait": 16.7, "req": 1880.0}, "one_hour_ago": {"cpu": 91.6, "ram": 49.6, "iowait": 17.4, "req": 1700.0},
           "diagnosis": {"most_likely": "noisy_neighbor", "probability_pct": 87.0, "alternative": "brute_force", "alternative_pct": 9.0},
           "suggested_actions": explain.RUNBOOK["noisy_neighbor"]}


def test_explain_uses_llm_only_when_guardrail_passes_and_never_raises():
    assert explain.explain(FULL_EV)["source"] == "plantilla"
    good = explain.explain(FULL_EV, _client("Riesgo del 43 % en web-01; CPU 94.0 % frente a 91.6 % hace 1 h."))
    assert good["source"] == "llm" and good["guardrail"] == "ok"
    bad = explain.explain(FULL_EV, _client("CPU al 99.9 % y 412 conexiones"))
    assert bad["source"] == "plantilla" and "rechazado" in bad["guardrail"]
    down = explain.explain(FULL_EV, _client(exc=RuntimeError("red caída")))
    assert down["source"] == "plantilla" and "no disponible" in down["guardrail"]
    cut = explain.explain(FULL_EV, _client("Riesgo del 43 % en", stop_reason="max_tokens"))  # texto cortado
    assert cut["source"] == "plantilla" and "no disponible" in cut["guardrail"]
    assert explain.check_numbers(explain.render_template(FULL_EV), FULL_EV)[0]  # la plantilla cumple su propia regla


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="VmHWM solo existe en Linux")
def test_engine_memory_is_measured_per_process_not_inherited_from_the_parent(tmp_path):
    """ru_maxrss arrastra el pico del padre a través del fork: con un padre de ~1 GB ambos motores marcaban ~1 GB."""
    scale.generate_parquet_dataset(tmp_path, 5, 3)
    ballast = np.ones(130_000_000)  # el proceso que lanza la prueba ocupa ~1 GB
    res = scale.benchmark(tmp_path, "2026-06-03")
    assert ballast.size and all(r["peak_rss_mb"] < 700 for r in res), res
