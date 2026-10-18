"""Orquestación de extremo a extremo: ingesta -> calidad -> almacenamiento -> modelos -> análisis -> informe."""
from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, deep, diagnosis, explain, scale, store, tracking, validation
from .bundle import save_model
from .features import FORECAST_FEATURES
from .offers import generate_offers, offer_value
from .pipeline import benchmark, run
from .report import build_report
from .report_extras import extra_sections
from .segments import (
    account_features,
    cluster_accounts,
    flag_near_limit,
    forward_validation,
    sensitivity_flag,
    technical_cause,
    utilization,
    vs_ground_truth,
)
from .simulator import SimConfig
from .sources import Dataset, load_source

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Maximum iterations.*")


def _json(o):
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _strip(d):
    """Quita listas largas por evento antes de guardar resultados en JSON."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in ("per_event", "model", "scores", "calibrator")}
    return d


def customers_step(dataset: Dataset, metrics: pd.DataFrame, out: Path, cap: dict, client=None) -> dict:
    """Utilización por plan -> regla -> guarda técnica -> segmentación -> borradores de oferta."""
    try:
        util = utilization(dataset.usage, dataset.accounts, dataset.plans)
    except ValueError as e:
        return {"available": False, "reason": str(e)}
    flagged = flag_near_limit(account_features(util, dataset.accounts))
    technical, technical_source = technical_cause(dataset.accounts, metrics, util)
    clustered = cluster_accounts(flagged[~flagged["account_id"].isin(technical)])
    flagged["cluster"] = ""
    if clustered.get("available"):
        names = clustered["features"].set_index("account_id")["cluster_id"].map(clustered["cluster_names"])
        flagged["cluster"] = flagged["account_id"].map(names).fillna("")
    flagged.loc[flagged["account_id"].isin(technical), "cluster"] = "causa_tecnica"
    reference = util["timestamp"].max().normalize()
    offers = generate_offers(flagged, technical, reference_date=reference, client=client)
    store.write_parquet(dataset.accounts, out / "data" / "accounts.parquet")
    store.write_usage_parquet(dataset.usage, out / "data" / "account_usage_hourly.parquet")
    (out / "data" / "offers.json").write_text(json.dumps(offers, ensure_ascii=False, indent=2, default=_json))
    seg = {k: v for k, v in clustered.items() if k not in ("features", "centroids")}
    if clustered.get("available"):
        seg["centroids"] = clustered["centroids"].round(3).reset_index().to_dict("records")
    return {"available": True, "n_accounts": int(len(flagged)), "n_flagged": int(flagged["near_limit"].sum()),
            "features": flagged.to_dict("records"), "segmentation": seg,
            "rule_sensitivity": sensitivity_flag(flagged), "forward_validation": forward_validation(util, dataset.accounts),
            "ground_truth": vs_ground_truth(clustered, flagged) if cap.get("ground_truth_archetypes") else None,
            "technical_accounts": sorted(technical), "technical_source": technical_source,
            "offers": offers, "offer_value": offer_value(offers)}


def build_all(out: str | Path = "outputs", cfg: SimConfig | None = None, budget: float = 0.5, bench_seeds: int = 3,
              real: bool = False, scale_servers: int = 100, scale_days: int = 30, llm: bool = False,
              log=print, dataset: Dataset | None = None) -> dict:
    out = Path(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    cfg = cfg or SimConfig()
    x: dict = {}

    log("1/10 Ingesta, calidad y reparación…")
    dataset = dataset or load_source("sim", cfg)
    df, x["ingest"] = dataset.metrics, dataset.report
    cap = dataset.capabilities
    x["source"], x["capabilities"] = dataset.source, cap
    for name, ok in cap.items():
        if ok is False and f"{name}_reason" in cap:
            log(f"[aviso] se omite {name}: {cap[name + '_reason']}")
    if not x["ingest"]["quality_after"]["ok"]:
        raise RuntimeError("Los datos reparados siguen sin cumplir el contrato de calidad")
    store.write_parquet(df, out / "data" / "metrics.parquet")

    log("2/10 Entrenamiento y evaluación principal…")
    art = run(cfg, budget=budget, df=df)
    x["main"] = art.results

    log("3/10 Análisis de negocio…")
    x["eda"] = analysis.eda(art.df) if cap["incident_eda"] else {
        "days": round(art.results["dataset"]["days"], 1), "saturated_share_of_time": float(art.df["sat_now"].mean())}
    b_methods = art.results["tasks"]["B"]["methods"]
    x["costs"] = analysis.cost_table(b_methods).to_dict("records")
    x["sensitivity"] = analysis.sensitivity(b_methods).to_dict("records")

    log("4/10 Validación walk-forward, calibración e interpretabilidad…")
    x["walk_forward"] = validation.walk_forward(art.df, cfg, budget=budget).to_dict("records")
    cal = validation.calibration(art)
    x["calibration"] = {k: v for k, v in cal.items() if k != "calibrator"}
    x["importance"] = validation.importance(art)

    log("5/10 Deep learning (autoencoder)…")
    ae = deep.evaluate_autoencoder(art, budget)
    x["deep"] = ae["results"]

    log("6/10 Diagnóstico de causa…")
    if cap["diagnosis"]:
        dg = diagnosis.evaluate_diagnosis(art)
        x["diagnosis"], dx = dg["summary"], dg["model"]
        x["diagnosis_shuffled"] = diagnosis.evaluate_diagnosis(art, shuffle_labels=True)["summary"]
    else:
        x["diagnosis"] = x["diagnosis_shuffled"] = dx = None

    log("7/10 Explicaciones de alarmas (IA)…")
    client = explain.default_client() if llm else None
    al = art.alerts[("B", "gbm")]
    firsts = al[al & ~al.groupby(art.df["server"]).shift(1, fill_value=False)].index[:150]
    expl = []
    for ix in firsts:
        ev = explain.build_evidence(art.df, art.feats, ix, art.scores["gbm"].loc[ix],
                                    dx.predict_proba(art.feats.loc[ix]) if dx else None)
        expl.append({"index": int(ix), "server": ev["server"], "timestamp": ev["timestamp"], "evidence": ev,
                     **explain.explain(ev, client)})
    x["explanations"] = expl
    (out / "data" / "alerts_explained.json").write_text(json.dumps(expl, ensure_ascii=False, default=_json))

    log("8/10 Clientes cerca de su límite, segmentación y ofertas…")
    x["customers"] = {"available": False, "reason": cap.get("customers_reason", "faltan datos de cuentas o uso")}
    if cap.get("customers"):
        x["customers"] = customers_step(dataset, df, out, cap, client)

    log("9/10 Prueba de escala (big data)…")
    tmp = tempfile.mkdtemp()
    try:
        n = scale.generate_parquet_dataset(tmp, scale_servers, scale_days)
        cutoff = str((pd.Timestamp(cfg.start) + pd.Timedelta(days=scale_days * 0.6)).date())
        x["scale"] = {"quick": {"servers": scale_servers, "days": scale_days, "rows": n,
                                "machine": scale.machine(), "results": scale.benchmark(tmp, cutoff)}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    full = Path(__file__).resolve().parent.parent / "docs" / "scale_benchmark.json"
    if full.exists():
        x["scale"] |= {f"full_{k}": v for k, v in json.loads(full.read_text()).items()}

    log("10/10 Informe, dashboard y registro de experimento…")
    sc = pd.DataFrame({"timestamp": art.df["timestamp"], "server": art.df["server"],
                       "split": np.select([art.splits["train"], art.splits["val"], art.splits["test"]],
                                          ["train", "val", "test"], "fuera"),
                       "risk_gbm": art.scores["gbm"], "score_iforest": art.scores["iforest"],
                       "score_zscore": art.scores["zscore"], "score_ae": ae["scores"]})
    for name in ("threshold", "zscore", "iforest", "gbm"):
        sc[f"alert_B_{name}"] = False
        sc.loc[art.alerts[("B", name)].index, f"alert_B_{name}"] = art.alerts[("B", name)]
    sc.to_parquet(out / "data" / "scores.parquet", compression="zstd", index=False)

    # el benchmark repite el experimento con otros mundos simulados: no aplica a datos reales
    bench = (benchmark([cfg.seed + i + 1 for i in range(bench_seeds)], budget=budget)
             if bench_seeds and dataset.source == "sim" else None)
    if bench is not None:
        flat = bench.copy()
        flat.columns = [f"{a}_{s}" for a, s in flat.columns]
        x["benchmark"] = {"seeds": [cfg.seed + i + 1 for i in range(bench_seeds)],
                          "rows": flat.reset_index().to_dict("records")}
    note = ""
    if real:
        try:
            from .real import Z_THR, run_real, summary_html
            res = run_real(out / "real_cache")
            res.to_csv(out / "real_results.csv", index=False)
            x["real"] = res.to_dict("records")
            note = summary_html(res, Z_THR)
        except Exception as e:
            log(f"[aviso] sin validación con datos reales: {e}")
    (out / "report.html").write_text(build_report(art, bench, note, extra_sections(x), source=dataset.source), encoding="utf-8")

    save_model(out / "model.joblib", feature_builder=art.models["feature_builder"], forecaster=art.models["gbm"],
               threshold=art.thresholds[("B", "gbm")], persistence=2, diagnoser=dx, features=FORECAST_FEATURES)
    g = art.results["tasks"]["B"]["methods"]["gbm"]
    run_id = tracking.log_run("capacity-guard-all", {"seed": cfg.seed, "days": cfg.days, "servers": cfg.n_servers,
                                                     "budget": budget, "source": dataset.source},
                              {"early_rate": g["early_rate"], "fa_per_server_day": g["false_alarms_per_server_day"],
                               "auc": art.results["gbm_sample_metrics"]["roc_auc"],
                               **({"diagnosis_acc": x["diagnosis"]["by_lag"]["al saltar la alarma"]["accuracy"]}
                                  if x["diagnosis"] else {})},
                              [out / "report.html"], out / "mlruns")
    x["mlflow_run_id"] = run_id
    (out / "results.json").write_text(json.dumps(_strip(x), indent=1, ensure_ascii=False, default=_json))
    from .summary import results_markdown
    (out / "RESULTS.md").write_text(results_markdown(json.loads((out / "results.json").read_text())), encoding="utf-8")
    log(f"Listo: {out/'report.html'}")
    return x
