"""Prueba de humo del proyecto completo en pequeño: orquestador, artefactos, CLI de puntuación y dashboard."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from capguard import store
from capguard.orchestrate import build_all
from capguard.simulator import SimConfig

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    x = build_all(out, SimConfig(days=45, n_servers=4, seed=3), bench_seeds=0, real=False,
                  scale_servers=10, scale_days=14, log=lambda *_: None)
    return out, x


def test_all_artifacts_are_written(built):
    out, x = built
    for f in ("report.html", "results.json", "model.joblib", "data/metrics.parquet", "data/scores.parquet",
              "data/alerts_explained.json"):
        assert (out / f).stat().st_size > 1000, f
    assert x["mlflow_run_id"] and (out / "mlruns").exists()
    res = json.loads((out / "results.json").read_text())
    assert {"ingest", "main", "eda", "costs", "walk_forward", "calibration", "importance", "deep", "diagnosis",
            "explanations", "scale"} <= set(res)


def test_report_contains_every_discipline(built):
    html = (built[0] / "report.html").read_text()
    for h in ("Ingeniería de datos", "Big data", "Analista de datos y BI", "Machine learning", "Deep learning",
              "Diagnóstico de causa", "IA ·", "Límites"):
        assert h in html, h
    assert "None" not in html.split("</style>")[1]  # ningún valor vacío sin formatear en el informe


def test_numbers_are_internally_consistent(built):
    _, x = built
    m = x["main"]["tasks"]["B"]["methods"]
    for k, v in m.items():
        assert 0 <= v["recall"] <= 1 and 0 <= v["early_rate"] <= v["recall"] + 1e-9, k
        assert v["false_alarm_episodes"] <= v["alert_episodes"]
    assert x["ingest"]["quality_after"]["ok"] and not x["ingest"]["quality_before"]["ok"]
    dg, sh = x["diagnosis"]["by_lag"], x["diagnosis_shuffled"]["by_lag"]
    k = "al saltar la alarma"
    assert dg[k]["accuracy"] > sh[k]["accuracy"] + 0.2  # etiquetas barajadas => rendimiento de azar
    assert x["eda"]["n_incidents"] == sum(r["incidentes"] for r in x["eda"]["by_kind"])


def test_data_is_queryable_with_duckdb_and_explanations_obey_guardrail(built):
    out, x = built
    con = store.connect(out / "data" / "metrics.parquet")
    n, servers = con.execute("SELECT count(*), count(DISTINCT server) FROM metrics").fetchone()
    assert servers == 4 and n > 50000
    from capguard.explain import check_numbers
    assert x["explanations"] and all(check_numbers(e["text"], e["evidence"])[0] for e in x["explanations"])


def test_score_cli_on_new_metrics(built, tmp_path):
    out, _ = built
    d = pd.read_parquet(out / "data" / "metrics.parquet")
    d = d[d["server"] == "web-02"].iloc[-1500:][["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"]]
    inp, res = tmp_path / "in.csv", tmp_path / "alerts.csv"
    d.to_csv(inp, index=False)
    p = subprocess.run([sys.executable, "-m", "capguard", "score", "--model", str(out / "model.joblib"),
                        "--input", str(inp), "--out", str(res)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-500:]
    r = pd.read_csv(res)
    assert len(r) == 1500 and r["risk_30min"].between(0, 1).all() and r["alert"].dtype == bool


def test_score_cli_on_unseen_server_and_dirty_feed(built, tmp_path):
    """Caso real: un servidor que el modelo nunca vio, con una fila duplicada y un hueco corto."""
    out, _ = built
    d = pd.read_parquet(out / "data" / "metrics.parquet")
    d = d[d["server"] == "web-01"].iloc[-600:][["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"]]
    d = pd.concat([d.drop(index=d.index[100:102]), d.iloc[[50]]]).assign(server="mi-servidor-01")
    inp, res = tmp_path / "in.csv", tmp_path / "alerts.csv"
    d.to_csv(inp, index=False)
    p = subprocess.run([sys.executable, "-m", "capguard", "score", "--model", str(out / "model.joblib"),
                        "--input", str(inp), "--out", str(res)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-500:]
    assert "no vistos al entrenar" in p.stdout and "reparados" in p.stdout
    r = pd.read_csv(res)
    assert len(r) == 600 and r["risk_30min"].between(0, 1).all() and (r["server"] == "mi-servidor-01").all()


def test_model_bundle_records_library_versions(built):
    from capguard.bundle import load_model
    b = load_model(built[0] / "model.joblib")
    assert {"scikit-learn", "pandas", "numpy"} <= set(b["versions"]) and b["threshold"] > 0


def _cheapest(x, c_miss, c_fa):
    """Método más barato según la tabla de costes: lo que el dashboard debe recomendar con esos costes."""
    from capguard.analysis import cost_table
    ct = cost_table(x["main"]["tasks"]["B"]["methods"], {"c_miss": c_miss, "c_fa": c_fa, "c_mitigated": 20})
    return ct.loc[ct["coste_total"].idxmin(), "method"]


def test_dashboard_renders_and_reacts(built, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CAPGUARD_OUT", str(built[0]))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception and len(at.tabs) == 7 and len(at.metric) >= 8
    at.selectbox(key="server").set_value("web-03").run()
    at.slider(key="c_fa").set_value(250).run()
    at.slider(key="c_miss").set_value(50).run()
    assert not at.exception
    assert _cheapest(built[1], 50, 250) in at.success[0].value
    at.slider(key="c_fa").set_value(1).run()
    at.slider(key="c_miss").set_value(2000).run()
    assert not at.exception and _cheapest(built[1], 2000, 1) in at.success[0].value
    at.selectbox(key="alert_pick").set_value(2).run()
    assert not at.exception
    at.selectbox(key="customer_offer").set_value("upgrade").run()  # pestaña Clientes
    assert not at.exception and any("Pendiente de revisión" in w.value for w in at.warning)
    picks = at.selectbox(key="customer_account").options
    at.selectbox(key="customer_account").set_value(picks[-1]).run()
    assert not at.exception


def test_export_demo_produces_a_small_snapshot_the_dashboard_can_read(built, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from capguard.demo import export_demo
    out, _ = built
    sizes = export_demo(out, tmp_path / "demo")
    full = sum(p.stat().st_size for p in (out / "data").glob("*.parquet"))
    assert sum(v for k, v in sizes.items() if k.endswith(".parquet")) < full / 2
    assert not any(k.endswith(".joblib") for k in sizes)
    monkeypatch.setenv("CAPGUARD_OUT", str(tmp_path / "demo"))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception and len(at.tabs) == 7
