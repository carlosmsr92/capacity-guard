"""Simulador acoplado: la carga del servidor sale de las cuentas y cada evento tiene una cuenta responsable."""
import numpy as np
import pandas as pd

from capguard.accounts import make_accounts
from capguard.simulator import STEP_MIN, SimConfig, _simulate_server, simulate_world

CFG = SimConfig(days=14, n_servers=3, seed=12)
RES = ("cpu", "ram", "io", "procesos")


def _idx(cfg):
    return pd.date_range(cfg.start, periods=cfg.days * 24 * 60 // STEP_MIN, freq=f"{STEP_MIN}min")


def test_simulate_world_is_deterministic():
    first, second = simulate_world(CFG), simulate_world(CFG)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.accounts, second.accounts)
    pd.testing.assert_frame_equal(first.usage_hourly, second.usage_hourly)
    assert len(first.accounts) == CFG.n_servers * CFG.accounts_per_server


def test_server_metrics_depend_on_the_accounts():
    """Si el acoplamiento fuera decorativo, cambiar las cuentas no movería la carga del servidor."""
    few = simulate_world(SimConfig(days=7, n_servers=2, seed=12, accounts_per_server=3), with_usage=False)
    many = simulate_world(SimConfig(days=7, n_servers=2, seed=12, accounts_per_server=40), with_usage=False)
    assert not np.allclose(few.metrics["cpu"], many.metrics["cpu"])

    cfg = SimConfig(days=7, n_servers=1, seed=5)
    accounts = make_accounts("web-01", np.random.default_rng(0), cfg.accounts_per_server)
    calm = accounts.assign(archetype="holgado")
    busy = calm.copy()
    busy.loc[:4, "archetype"] = "intensivo_constante"  # 5 cuentas pasan a usar casi todo su plan
    a = _simulate_server("web-01", _idx(cfg), cfg, np.random.default_rng(1), calm)
    b = _simulate_server("web-01", _idx(cfg), cfg, np.random.default_rng(1), busy)
    assert b.usage["cpu"].sum() > a.usage["cpu"].sum() * 1.1
    assert not np.allclose(a.metrics["cpu"], b.metrics["cpu"])


def test_account_usage_explains_server_load_outside_events():
    world = simulate_world(CFG)
    m, u = world.metrics, world.usage_hourly
    for server in m["server"].unique():
        mm = m[m["server"] == server].set_index("timestamp")
        quiet = mm["event"].eq("").resample("1h").min()  # horas sin ningún evento en el servidor
        accounts = u[u["server"] == server].groupby("timestamp")["cpu"].sum()
        cpu = mm["cpu"].resample("1h").mean()
        assert np.corrcoef(accounts[quiet.to_numpy()], cpu[quiet])[0, 1] > 0.9, server


def test_plan_limits_are_real_and_respected():
    u = simulate_world(CFG).usage_hourly
    normal = u[u["incidente"] == ""]
    for r in RES:
        assert normal[r].le(normal[f"{r}_limit"] * (1 + 1e-9)).all(), r
    assert set(u["cpu_limit"]) <= {1.0, 2.0, 3.0, 4.0, 8.0} and u["ram_limit"].min() >= 1024
    arch = simulate_world(CFG).accounts.set_index("account_id")["archetype"]
    util = normal.assign(util=np.max([normal[r] / normal[f"{r}_limit"] for r in RES], axis=0))
    p95 = util.groupby("account_id")["util"].quantile(0.95)
    by_arch = p95.groupby(p95.index.map(arch)).median()
    assert by_arch["intensivo_constante"] > 0.85 and by_arch["holgado"] < 0.6
    throttled = normal.groupby("account_id")["minutos_limitado"].sum()
    assert throttled[throttled.index.map(arch) == "holgado"].eq(0).mean() > 0.9
    assert throttled[throttled.index.map(arch) == "intensivo_constante"].gt(0).mean() > 0.5


def test_every_event_is_attributed_per_event_and_leaves_a_footprint():
    world = simulate_world(SimConfig(days=30, n_servers=3, seed=21, accounts_per_server=80))
    m, u = world.metrics, world.usage_hourly
    arch = world.accounts.set_index("account_id")["archetype"]
    events = m[m["event"].ne("")]
    assert events["event_account"].ne("").all()
    culprits = events["event_account"].map(arch)
    assert culprits[events["is_incident"]].eq("causa_tecnica").all()
    assert culprits[events["is_benign_event"]].eq("estacional_campanas").all()
    # con varias cuentas técnicas en un servidor, eventos distintos recaen en cuentas distintas
    assert events.groupby(["server", "event"])["event_account"].nunique().max() > 1
    # la cuenta culpable usa muchos más recursos (respecto a su plan) durante su incidente que fuera de él
    for account in events.loc[events["is_incident"], "event_account"].unique()[:5]:
        a = u[u["account_id"] == account]
        load = sum(a[r] / a[f"{r}_limit"] for r in RES)
        assert load[a["incidente"] != ""].mean() > 1.5 * load[a["incidente"] == ""].mean(), account


def test_more_demanding_customers_raise_the_server_load_level():
    """Escala fija de flota: más demanda de los clientes = servidor más cargado (no solo otra forma de curva)."""
    cfg = SimConfig(days=7, n_servers=1, seed=5)
    accounts = make_accounts("web-01", np.random.default_rng(0), cfg.accounts_per_server).assign(archetype="holgado")
    busy = accounts.copy()
    busy.loc[:9, "archetype"] = "intensivo_constante"
    calm = _simulate_server("web-01", _idx(cfg), cfg, np.random.default_rng(1), accounts).metrics
    hot = _simulate_server("web-01", _idx(cfg), cfg, np.random.default_rng(1), busy).metrics
    quiet = calm["event"].eq("") & hot["event"].eq("")
    assert hot.loc[quiet, "cpu"].mean() > calm.loc[quiet, "cpu"].mean() + 5
