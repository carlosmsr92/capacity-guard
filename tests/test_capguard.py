import numpy as np
import pandas as pd

from capguard.evaluation import episodes, evaluate, persistent
from capguard.features import FeatureBuilder, future_saturation
from capguard.pipeline import run
from capguard.simulator import SimConfig, simulate

SMALL = SimConfig(days=14, n_servers=3, seed=3)


def test_simulator_is_deterministic_and_clean():
    a, b = simulate(SMALL), simulate(SMALL)
    pd.testing.assert_frame_equal(a, b)
    assert not a.isna().any().any()
    assert a[["cpu", "ram", "iowait"]].max().max() <= 100
    assert a["is_incident"].any() and a["is_benign_event"].any()


def test_episodes_and_persistence():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert episodes(m) == [(1, 2), (5, 5), (7, 9)]
    assert episodes(m, merge_gap=2) == [(1, 5), (7, 9)] or episodes(m, merge_gap=2) == [(1, 9)]
    assert persistent(m, 2).tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 1, 1]


def test_features_are_causal():
    df = simulate(SMALL)
    train = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=8)]
    fb = FeatureBuilder().fit(train)
    f1 = fb.transform(df)
    cut = df["timestamp"].min() + pd.Timedelta(days=10)
    df2 = df.copy()
    df2.loc[df2["timestamp"] >= cut, ["cpu", "ram", "iowait", "req", "net", "load1"]] *= 3  # cambia solo el futuro
    f2 = fb.transform(df2)
    past = (df["timestamp"] < cut).to_numpy()
    pd.testing.assert_frame_equal(f1[past], f2[past])


def test_unseen_server_uses_fleet_profile():
    df = simulate(SMALL)
    fb = FeatureBuilder().fit(df[df["server"] != "web-03"])
    assert fb.unknown_servers(df) == ["web-03"]
    f = fb.transform(df)
    new = (df["server"] == "web-03").to_numpy()
    assert f.loc[new, "z_cpu"].notna().all() and np.isfinite(f.loc[new, "cpu_unexplained"]).all()
    assert f.loc[new, "z_cpu"].abs().median() < 5  # perfil de flota razonable para un servidor parecido


def test_future_saturation_label():
    df = pd.DataFrame({"server": "s", "sat_now": [False, False, False, True, False, False]})
    assert future_saturation(df, horizon=2).tolist() == [0, 1, 1, 0, 0, 0]


def test_evaluate_lead_and_false_alarms():
    df = pd.DataFrame({"server": ["s"] * 20})
    alert = pd.Series([False] * 20)
    alert[[3, 4, 15]] = True  # aviso 7 muestras antes del evento (35 min) + una falsa alarma
    res = evaluate(df, alert, {"s": [(10, 12)]}, pre=8, post=0)
    assert res["recall"] == 1 and res["median_lead_min"] == 35 and res["early_rate"] == 1
    assert res["alert_episodes"] == 2 and res["alert_precision"] == 0.5


def test_pipeline_end_to_end_beats_static_threshold_early_warning():
    art = run(SimConfig(days=30, n_servers=4, seed=5))
    b = art.results["tasks"]["B"]["methods"]
    assert b["gbm"]["early_rate"] >= b["threshold"]["early_rate"]
    assert 0.5 < art.results["gbm_sample_metrics"]["roc_auc"] <= 1.0
