"""Clientes cerca de su límite: regla en utilización, segmentación interpretable, guardas y ofertas en borrador."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capguard.offers import DEFAULT_OFFER_PARAMS, generate_offers, offer_text, offer_value
from capguard.segments import (
    account_features,
    cluster_accounts,
    flag_near_limit,
    forward_validation,
    name_clusters,
    technical_cause,
    utilization,
    vs_ground_truth,
)
from capguard.simulator import SimConfig, simulate_world


@pytest.fixture(scope="module")
def world():
    w = simulate_world(SimConfig(seed=7))
    util = utilization(w.usage_hourly, w.accounts, w.plans)
    flagged = flag_near_limit(account_features(util, w.accounts))
    return w, util, flagged


def _client(text):
    def create(**kw):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_rule_flags_tight_accounts_and_not_comfortable_ones(world):
    _, _, f = world
    share = f.groupby("archetype")["near_limit"].mean()
    assert share["holgado"] < 0.05 and share["causa_tecnica"] < 0.2
    assert share["intensivo_constante"] > 0.9
    assert 0.05 < f["near_limit"].mean() < 0.4  # marcar a casi todos (o a nadie) sería una regla inútil


def test_rule_is_measured_against_the_plan_not_in_absolute_units():
    """900 MB de RAM es estar al límite en un plan de 1 GB y estar holgado en un VPS de 8 GB."""
    ts = pd.date_range("2026-06-01", periods=24 * 14, freq="1h")
    accounts = pd.DataFrame({"account_id": ["a", "b"], "plan": ["basico", "vps"], "server": "s"})
    usage = pd.concat([pd.DataFrame({"timestamp": ts, "account_id": a, "cpu": 0.1, "ram": 900.0, "io": 1.0,
                                     "procesos": 5.0}) for a in ("a", "b")])
    f = flag_near_limit(account_features(utilization(usage, accounts), accounts)).set_index("account_id")
    assert f.loc["a", "near_limit"] and not f.loc["b", "near_limit"]
    assert f.loc["a", "p95_ram"] == pytest.approx(900 / 1024) and f.loc["b", "p95_ram"] == pytest.approx(900 / 8192)
    with pytest.raises(ValueError, match="plan desconocido"):
        utilization(usage, accounts.assign(plan=["basico", "oro"]))


def test_segmentation_is_stable_interpretable_and_recovers_archetypes(world):
    w, _, f = world
    technical, _ = technical_cause(w.accounts, w.metrics)
    c = cluster_accounts(f[~f["account_id"].isin(technical)])
    assert c["available"] and c["stability"] >= 0.8 and c["tree_fidelity"] > 0.85
    assert vs_ground_truth(c, f)["ari"] > 0.5
    names = list(c["cluster_names"].values())
    assert len(set(names)) == len(names)  # nunca dos grupos distintos con el mismo nombre
    assert "intensivo" in names


def test_cluster_names_come_from_fixed_rules():
    cents = pd.DataFrame([
        {"p95_max": 0.97, "tendencia": 0.0, "rafagas": 0, "concentracion_picos": 0.02},
        {"p95_max": 0.70, "tendencia": 0.8, "rafagas": 0, "concentracion_picos": 0.10},
        {"p95_max": 0.80, "tendencia": 0.5, "rafagas": 0, "concentracion_picos": 0.30},
        {"p95_max": 0.45, "tendencia": 0.0, "rafagas": 15, "concentracion_picos": 0.10},
        {"p95_max": 0.95, "tendencia": 0.0, "rafagas": 0, "concentracion_picos": 0.01},
        {"p95_max": 0.40, "tendencia": 0.0, "rafagas": 0, "concentracion_picos": 0.05},
    ])
    assert name_clusters(cents) == {0: "intensivo", 1: "crecimiento", 2: "campanas", 3: "esporadico",
                                    4: "intensivo_2", 5: "moderado"}


def test_forward_validation_compares_against_the_future(world):
    w, util, _ = world
    fv = forward_validation(util, w.accounts)
    assert fv["available"] and fv["positivos_futuros"] > 0
    for rule in ("regla", "ingenua"):
        assert 0 <= fv[rule]["precision"] <= 1 and 0 <= fv[rule]["cobertura"] <= 1
    assert fv["regla"]["precision"] >= fv["ingenua"]["precision"]  # la regla marca menos y acierta más


def test_technical_accounts_never_receive_a_sale_with_or_without_labels(world):
    w, util, f = world
    arch = w.accounts.set_index("account_id")["archetype"]
    labelled, src = technical_cause(w.accounts, w.metrics, util)
    unlabelled, src2 = technical_cause(w.accounts, w.metrics.assign(is_incident=False), util)
    assert src == "incidentes_etiquetados" and src2 == "uso_por_encima_del_limite"
    for technical in (labelled, unlabelled):
        assert technical and all(arch[a] == "causa_tecnica" for a in technical)
        forced = f.assign(near_limit=f["near_limit"] | f["account_id"].isin(technical), cluster="intensivo")
        offers = pd.DataFrame(generate_offers(forced, technical, reference_date="2026-07-15"))
        mine = offers[offers["account_id"].isin(technical)]
        assert len(mine) == len(technical) and mine["oferta"].eq("guardia_tecnica").all()
        assert offers["estado"].eq("pendiente_revision").all()


def _row(**kw):
    base = {"account_id": "c1", "plan": "basico", "near_limit": True, "cluster": "crecimiento", "alta": "2026-01-01",
            "p95_max": 0.93, "dias_cerca_limite": 9, "dias_limitado": 4}
    return pd.DataFrame([{**base, **kw}])


def test_offer_policy_targets_and_guards():
    ref = "2026-07-15"
    up = generate_offers(_row(), reference_date=ref)[0]
    assert (up["oferta"], up["plan_propuesto"]) == ("upgrade", "plus")
    assert up["precio_propuesto"] == pytest.approx(18 * 0.85) and "plus" in up["texto"]
    inten = generate_offers(_row(cluster="intensivo_2"), reference_date=ref)[0]
    assert inten["plan_propuesto"] == "plus_intermedio"
    assert generate_offers(_row(plan="pro", cluster="intensivo"), reference_date=ref)[0]["plan_propuesto"] == "vps"
    assert generate_offers(_row(plan="vps"), reference_date=ref)[0]["oferta"] == "plan_maximo"
    temp = generate_offers(_row(cluster="campanas"), reference_date=ref)[0]
    assert temp["oferta"] == "temporal" and 9 < temp["precio_propuesto"] < 18  # suplemento, no el plan entero
    assert generate_offers(_row(alta="2026-07-01"), reference_date=ref)[0]["oferta"] == "sin_oferta"
    cooled = generate_offers(_row(), reference_date=ref, history={"c1": pd.Timestamp("2026-06-01")})[0]
    assert cooled["oferta"] == "sin_oferta" and "enfriamiento" not in cooled["texto"]
    capped = generate_offers(_row(), reference_date=ref, params={"max_discount": 0.05})[0]
    assert capped["precio_propuesto"] == pytest.approx(18 * 0.95) and "5 %" in capped["texto"]
    assert DEFAULT_OFFER_PARAMS["max_discount"] <= 0.2


def test_offer_texts_pass_the_guardrail_and_an_invented_price_falls_back(world):
    _, _, f = world
    offers = generate_offers(f.assign(cluster="crecimiento"), reference_date="2026-07-15")
    assert offers and all(o["guardarrail"] == "ok" for o in offers)
    ev = offers[0]["evidencia"]
    bad = offer_text(ev, _client("Le ofrecemos el plan superior por solo 4,99 € al mes."))
    assert bad["source"] == "plantilla" and "rechazado" in bad["guardrail"]
    good = offer_text(ev, _client(f"Proponemos el plan {ev['plan_propuesto']} por {ev['precio_propuesto']:.2f} € al mes."))
    assert good["source"] == "llm"


def test_offer_value_is_explicitly_illustrative():
    offers = [{"oferta": "upgrade", "precio_actual": 9.0, "precio_propuesto": 15.3},
              {"oferta": "guardia_tecnica", "precio_actual": 18.0, "precio_propuesto": 18.0}]
    v = offer_value(offers)
    assert v["illustrative"] and "supuesto" in v["note"] and v["commercial"] == 1
    assert [s["expected_monthly_value"] for s in v["sensitivity"]] == pytest.approx([0.63, 1.575, 2.52], abs=0.01)
    assert np.all(np.diff([s["expected_monthly_value"] for s in v["sensitivity"]]) > 0)
