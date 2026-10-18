"""Capa de ingesta: normalización de CSV reales y ejecución de extremo a extremo con `all --source csv`."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from capguard.simulator import SimConfig, simulate
from capguard.sources import IngestError, load_source, normalize, read_csv_robust

ROOT = Path(__file__).resolve().parent.parent
CORE = ["timestamp", "server", "cpu", "ram", "iowait", "net", "load1", "req"]


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "capguard", *map(str, args)], capture_output=True, text=True, cwd=ROOT)


def _dirty_csv(path: Path, days: int, seed: int = 3) -> pd.DataFrame:
    """CSV como lo exportaría otra herramienta: sin etiquetas, `;`, coma decimal, latin-1, fracciones y alias."""
    df = simulate(SimConfig(days=days, n_servers=3, seed=seed))[CORE]
    d = df.copy()
    for c in ("cpu", "ram", "iowait"):
        d[c] = d[c] / 100
    d["sat_future"] = True  # columna ajena que no debe llegar al modelo
    d = d.rename(columns={"timestamp": "Fecha", "server": "Servidor", "cpu": "CPU %", "ram": "Memoria",
                          "req": "Peticiones"})
    d.to_csv(path, sep=";", decimal=",", index=False, encoding="latin-1")
    return df


@pytest.fixture(scope="module")
def csv_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("csv")
    src = tmp / "metricas.csv"
    _dirty_csv(src, days=30)
    out = tmp / "out"
    p = _cli("all", "--source", "csv", "--metrics", src, "--bench-seeds", 0, "--scale-servers", 4,
             "--scale-days", 7, "--out", out)
    return src, out, p


def test_all_source_csv_runs_end_to_end_and_skips_what_needs_labels(csv_run):
    src, out, p = csv_run
    assert p.returncode == 0, p.stderr[-800:]
    assert "se omite task_a" in p.stdout and "se omite diagnosis" in p.stdout
    r = json.loads((out / "results.json").read_text())
    assert r["source"] == "csv" and r["capabilities"]["task_a"] is False and r["diagnosis"] is None
    assert r["ingest"]["unit_conversions"] and "sat_future" in r["ingest"]["ignored_columns"]
    assert r["main"]["dataset"]["days"] == pytest.approx(30, abs=0.01) and r["main"]["dataset"]["servers"] == 3
    assert 0.6 < r["main"]["gbm_sample_metrics"]["roc_auc"] <= 1
    html = (out / "report.html").read_text()
    assert "Ingesta y origen de los datos" in html and "Se omite" in html
    body = html.split("</style>")[1]
    assert "None" not in body and "nan %" not in body
    metrics = pd.read_parquet(out / "data" / "metrics.parquet")
    assert "sat_future" not in metrics.columns and metrics["cpu"].max() > 1  # fracción -> porcentaje
    assert (out / "RESULTS.md").read_text().count("se omite") == 1


def test_score_and_validate_accept_the_same_dirty_csv(csv_run, tmp_path):
    src, out, _ = csv_run
    p = _cli("score", "--model", out / "model.joblib", "--input", src, "--out", tmp_path / "alerts.csv")
    assert p.returncode == 0, p.stderr[-500:]
    assert pd.read_csv(tmp_path / "alerts.csv")["risk_30min"].between(0, 1).all()
    p = _cli("validate-csv", "--metrics", src, "--out", tmp_path / "v")
    assert p.returncode == 0, p.stderr[-500:]
    assert (tmp_path / "v" / "metrics.parquet").exists() and "[aviso]" not in p.stdout


def test_dashboard_renders_csv_results(csv_run, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CAPGUARD_OUT", str(csv_run[1]))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert any("omitidas" in w.value for w in at.warning)


def test_short_csv_validates_but_is_rejected_for_training(tmp_path):
    src = tmp_path / "corto.csv"
    _dirty_csv(src, days=7)
    p = _cli("validate-csv", "--metrics", src, "--out", tmp_path / "v")
    assert p.returncode == 0 and "no basta para entrenar" in p.stdout
    with pytest.raises(IngestError, match="Datos insuficientes"):
        load_source("csv", SimConfig(), {"metrics": src})


def test_fraction_heuristic_needs_cpu_and_ram_together():
    df = simulate(SimConfig(days=2, n_servers=1, seed=1))[CORE].assign(iowait=0.4)  # iowait real ocioso, en %
    out, rep = normalize(df)
    assert out["iowait"].eq(0.4).all() and rep.unit_conversions == []
    frac = df.assign(cpu=df["cpu"] / 100, ram=df["ram"] / 100, iowait=0.004)
    out, rep = normalize(frac)
    assert out["cpu"].max() > 1 and out["iowait"].eq(0.4).all() and rep.unit_conversions


def test_non_canonical_columns_are_dropped_and_coarse_data_rejected():
    df = simulate(SimConfig(days=2, n_servers=1, seed=1))
    out, rep = normalize(df.assign(extra=1))
    assert set(out.columns) <= set(CORE) | {"event", "event_account"}
    assert {"is_incident", "sat_now", "extra"} <= set(rep.ignored_columns)
    with pytest.raises(IngestError, match="15 min"):
        normalize(df[CORE].iloc[::3])


def test_simulated_source_carries_accounts_and_capabilities():
    ds = load_source("sim", SimConfig(days=10, n_servers=2, seed=4))
    assert ds.source == "sim" and ds.capabilities["customers"] and ds.capabilities["ground_truth_archetypes"]
    assert ds.capabilities["task_a"] and len(ds.accounts) == 2 * 40 and not ds.usage.empty
    assert "event_account" in ds.metrics and ds.metrics.loc[ds.metrics["event"].ne(""), "event_account"].ne("").all()


def test_spanish_thousands_separator_and_too_many_non_numeric(tmp_path):
    df = simulate(SimConfig(days=2, n_servers=1, seed=1))[CORE]
    d = df.assign(req=df["req"] * 3)  # > 1000 peticiones/min: aparecen separadores de miles
    d["req"] = d["req"].map(lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    src = tmp_path / "miles.csv"
    d.to_csv(src, sep=";", index=False)
    out, rep = normalize(read_csv_robust(src))
    assert rep.rows_final == len(df) and rep.non_numeric["req"] == 0
    assert out["req"].to_numpy() == pytest.approx((df["req"] * 3).round(2).to_numpy())
    broken = df.astype({"cpu": object})
    broken.loc[broken.index[::10], "cpu"] = "n/d"  # 10 % de basura: es un problema de formato, no ruido
    with pytest.raises(IngestError, match="cpu tiene un 10 % de valores no numéricos"):
        normalize(broken)


def test_day_first_dates_are_detected_and_ambiguous_ones_rejected():
    df = simulate(SimConfig(days=20, n_servers=1, seed=1))[CORE]
    dmy = df.assign(timestamp=df["timestamp"].dt.strftime("%d/%m/%Y %H:%M"))
    out, rep = normalize(dmy)
    assert out["timestamp"].min() == df["timestamp"].min() and "fechas día/mes" in rep.unit_conversions
    short = dmy[df["timestamp"] < "2026-06-12"]  # todos los días ≤ 12: imposible saber si es día/mes o mes/día
    with pytest.raises(IngestError, match="ambiguas.*--date-format"):
        normalize(short)
    out, _ = normalize(short, date_format="%d/%m/%Y %H:%M")
    assert out["timestamp"].min() == df["timestamp"].min()


def test_long_gaps_are_never_filled():
    df = simulate(SimConfig(days=2, n_servers=1, seed=2))[CORE]
    for k, added in ((36, 0), (4, 0), (3, 3)):
        out, _ = normalize(df.drop(index=range(300, 300 + k)))
        assert len(out) - (len(df) - k) == added, k


def test_unknown_event_labels_count_as_incidents_and_get_default_actions():
    from capguard import explain
    from capguard.sources import _labelled
    df = simulate(SimConfig(days=2, n_servers=1, seed=2))
    d, rep = normalize(df.assign(event=df["event"].replace({"brute_force": "ddos", "legit_surge": "campaña"})),
                       mapping={"event_values": {"campaña": "legit_surge"}})
    lab = _labelled(d)
    assert "ddos" in rep.event_labels and "campaña" not in rep.event_labels
    assert lab.loc[lab["event"] == "ddos", "is_incident"].all()
    assert not lab.loc[lab["event"] == "legit_surge", "is_incident"].any()
    assert "ddos" in explain.kind_es("ddos") and explain.RUNBOOK.get("ddos", explain.DEFAULT_ACTIONS)


def test_customers_from_real_csvs_without_archetypes_or_event_labels(tmp_path):
    from capguard.orchestrate import customers_step
    from capguard.simulator import simulate_world
    w = simulate_world(SimConfig(days=30, n_servers=3, seed=5))
    paths = {"metrics": tmp_path / "m.csv", "accounts": tmp_path / "a.csv", "usage": tmp_path / "u.csv"}
    w.metrics[CORE].to_csv(paths["metrics"], index=False)
    w.accounts.drop(columns="archetype").to_csv(paths["accounts"], index=False)
    w.usage_hourly[["timestamp", "account_id", "cpu", "ram", "io", "procesos"]].to_csv(paths["usage"], index=False)
    ds = load_source("csv", SimConfig(), paths)
    assert ds.capabilities["customers"] and not ds.capabilities["ground_truth_archetypes"]
    (tmp_path / "data").mkdir()
    c = customers_step(ds, ds.metrics, tmp_path, ds.capabilities)
    assert c["available"] and c["technical_source"] == "uso_por_encima_del_limite" and c["ground_truth"] is None
    arch = w.accounts.set_index("account_id")["archetype"]
    assert c["technical_accounts"] and all(arch[a] == "causa_tecnica" for a in c["technical_accounts"])
    offers = pd.DataFrame(c["offers"])
    sales = offers[offers["oferta"].isin(["upgrade", "temporal"])]
    assert not sales["account_id"].isin(c["technical_accounts"]).any()
    assert (tmp_path / "data" / "account_usage_hourly.parquet").stat().st_size < 10_000_000


def test_report_escapes_values_that_come_from_the_csv(tmp_path):
    """Nombres de servidor y etiquetas de evento llegan al HTML: nunca deben poder inyectar código."""
    df = simulate(SimConfig(days=30, n_servers=3, seed=6))[CORE + ["event"]]
    df["server"] = df["server"].replace({"web-01": "<script>alert(1)</script>"})
    df["event"] = df["event"].replace({"brute_force": "<img src=x onerror=alert(2)>"})
    src = tmp_path / "x.csv"
    df.to_csv(src, index=False)
    p = _cli("all", "--source", "csv", "--metrics", src, "--bench-seeds", 0, "--scale-servers", 4,
             "--scale-days", 7, "--out", tmp_path / "out")
    assert p.returncode == 0, p.stderr[-800:]
    page = (tmp_path / "out" / "report.html").read_text()
    assert "<script>alert" not in page and "<img src=x" not in page
