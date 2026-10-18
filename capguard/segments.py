"""Clientes cerca de su límite: utilización por plan, regla auditable, validación hacia delante y segmentación.

Todo se calcula sobre la **utilización** (uso / límite del plan, 0-1), nunca sobre el uso absoluto: 200 MB de RAM no
dicen nada sin saber si el plan tiene 1 GB u 8 GB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .accounts import LIMIT_FIELD, RESOURCES, plans_frame

NEAR = 0.85  # «cerca del límite»: ≥ 85 % del límite del plan
LIMITED = 0.999  # «limitado»: en el límite (el resto de la demanda se estrangula)
OVER = 1.05  # por encima del límite: solo posible si algo escapa al aislamiento por cuenta (causa técnica)
CLUSTER_FEATURES = ("p95_max", "tendencia", "rafagas", "concentracion_picos", "ratio_finde")


def utilization(usage: pd.DataFrame, accounts: pd.DataFrame, plans: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uso por cuenta y hora como fracción del límite de su plan.

    Los límites salen de las columnas `<recurso>_limit` si existen; si no, del plan de la cuenta y del catálogo.
    """
    u = usage.copy()
    if not all(f"{r}_limit" in u for r in RESOURCES):
        catalog = (plans if plans is not None else plans_frame()).set_index("plan")
        plan = u["account_id"].map(accounts.drop_duplicates("account_id").set_index("account_id")["plan"])
        unknown = sorted(set(plan.dropna()) - set(catalog.index))
        if plan.isna().any() or unknown:
            raise ValueError(f"No se conocen los límites de algunas cuentas (plan desconocido o cuenta sin plan): "
                             f"{unknown[:5] or 'cuentas sin plan'}. Aporta --plans o columnas <recurso>_limit.")
        for r in RESOURCES:
            u[f"{r}_limit"] = plan.map(catalog[LIMIT_FIELD[r]]).to_numpy(dtype=float)
    for r in RESOURCES:
        u[r] = u[r].astype(float) / u[f"{r}_limit"].astype(float)
    return u


def account_features(util: pd.DataFrame, accounts: pd.DataFrame, end=None, window_days: int = 14) -> pd.DataFrame:
    """Variables por cuenta a partir de la utilización horaria, usando solo datos hasta `end` (causal).

    Los días cerca del límite y limitados se cuentan en los últimos `window_days` días («repetidamente» = reciente);
    el resto de variables describen el patrón de todo el periodo y alimentan la segmentación.
    """
    d = util if end is None else util[util["timestamp"] < pd.Timestamp(end)]
    if "incidente" in d:  # la huella de un incidente atribuido no es demanda del cliente
        d = d[d["incidente"].fillna("").eq("")]
    d = d.assign(u=d[list(RESOURCES)].max(axis=1), day=d["timestamp"].dt.floor("D"))
    window_start = d["timestamp"].max() - pd.Timedelta(days=window_days)
    rows = []
    for account_id, g in d.groupby("account_id", sort=False):
        row = {"account_id": account_id}
        recent = g[g["timestamp"] > window_start]
        limited_hour = (recent["minutos_limitado"] > 0) if "minutos_limitado" in recent else recent["u"] >= LIMITED
        for r in RESOURCES:
            row[f"p95_{r}"] = float(g[r].quantile(0.95))
            daily_max = recent.groupby("day")[r].max()
            row[f"dias_cerca_{r}"] = int((daily_max >= NEAR).sum())
            row[f"dias_limitado_{r}"] = int((daily_max >= LIMITED).sum())
        row["p95_max"] = max(row[f"p95_{r}"] for r in RESOURCES)  # el recurso que más aprieta a esta cuenta
        row["dias_limitado_total"] = int(limited_hour.groupby(recent["day"]).any().sum())
        row["horas_limitado"] = int(limited_hour.sum())
        daily = g.groupby("day")["u"].mean()
        mean = max(float(daily.mean()), 1e-9)
        # crecimiento relativo a lo largo del periodo (pendiente diaria × días / nivel medio)
        days = np.arange(len(daily))
        slope, intercept = np.polyfit(days, daily.to_numpy(), 1) if len(daily) > 2 else (0.0, mean)
        row["tendencia"] = float(slope * len(daily) / mean)
        row["rafagas"] = int((g["u"].diff() > 0.3).sum())  # subidas bruscas de una hora a la siguiente
        # días muy distintos entre sí una vez quitada la tendencia: campañas (el crecimiento no cuenta aquí)
        row["concentracion_picos"] = float((daily.to_numpy() - (slope * days + intercept)).std() / mean)
        weekend = g["timestamp"].dt.dayofweek >= 5
        row["ratio_finde"] = float(g.loc[weekend, "u"].mean() / max(float(g.loc[~weekend, "u"].mean()), 1e-9))
        row["recurso_dominante"] = max(RESOURCES, key=lambda r: row[f"p95_{r}"])
        rows.append(row)
    cols = [c for c in ("account_id", "server", "plan", "alta", "archetype") if c in accounts]
    return pd.DataFrame(rows).merge(accounts[cols].drop_duplicates("account_id"), on="account_id", how="left")


def flag_near_limit(feats: pd.DataFrame, near_days: int = 5, throttle_days: int = 3,
                    window_days: int = 14) -> pd.DataFrame:
    """Regla auditable: ≥ near_days días cerca del límite o ≥ throttle_days días limitado en la ventana."""
    out = feats.copy()
    out["dias_cerca_limite"] = out[[f"dias_cerca_{r}" for r in RESOURCES]].max(axis=1)
    out["dias_limitado"] = out[[f"dias_limitado_{r}" for r in RESOURCES] + ["dias_limitado_total"]].max(axis=1)
    out["near_limit"] = (out["dias_cerca_limite"] >= near_days) | (out["dias_limitado"] >= throttle_days)
    out["rule_window_days"] = window_days
    return out


def sensitivity_flag(feats: pd.DataFrame, values=((3, 2), (5, 3), (7, 4))) -> list[dict]:
    """Cuántas cuentas marca la regla con umbrales más laxos o más estrictos."""
    rows = []
    for near, throttle in values:
        flagged = flag_near_limit(feats, near, throttle)
        rows.append({"near_days": near, "throttle_days": throttle, "flagged": int(flagged["near_limit"].sum()),
                     "share": float(flagged["near_limit"].mean())})
    return rows


def forward_validation(util: pd.DataFrame, accounts: pd.DataFrame, horizon_days: int = 14, window_days: int = 14,
                       near_days: int = 5, throttle_days: int = 3) -> dict:
    """¿La regla anticipa quién seguirá limitado? Se marca con datos hasta un corte y se mira el periodo siguiente.

    «Positivo» = la cuenta vuelve a cumplir la regla en los `horizon_days` siguientes. Se compara con la regla
    ingenua «≥1 día limitado en la ventana». No necesita etiquetas: funciona igual con datos reales.
    """
    t_end = util["timestamp"].max()
    cut = t_end - pd.Timedelta(days=horizon_days)
    if cut - util["timestamp"].min() < pd.Timedelta(days=window_days):
        return {"available": False, "reason": f"hacen falta al menos {window_days + horizon_days} días de uso"}
    past = flag_near_limit(account_features(util, accounts, end=cut, window_days=window_days), near_days, throttle_days)
    future = flag_near_limit(account_features(util[util["timestamp"] >= cut], accounts, window_days=horizon_days),
                             near_days, throttle_days).set_index("account_id")["near_limit"]
    truth = past["account_id"].map(future).fillna(False).astype(bool).to_numpy()

    def score(pred):
        pred = np.asarray(pred, dtype=bool)
        tp = int((pred & truth).sum())
        return {"marcadas": int(pred.sum()), "precision": tp / pred.sum() if pred.sum() else float("nan"),
                "cobertura": tp / truth.sum() if truth.sum() else float("nan")}

    return {"available": True, "corte": str(cut.date()), "positivos_futuros": int(truth.sum()),
            "regla": score(past["near_limit"]), "ingenua": score(past["dias_limitado"] >= 1)}


def technical_cause(accounts: pd.DataFrame, metrics: pd.DataFrame, util: pd.DataFrame | None = None,
                    start=None, end=None) -> tuple[set[str], str]:
    """Cuentas cuyo uso alto tiene una causa técnica: a ellas nunca se les ofrece una venta.

    Con incidentes etiquetados y atribuidos (`event_account`) se usan esas etiquetas. Sin etiquetas (CSV real sin
    `event`), se marcan las cuentas que superan el límite de su plan: con aislamiento por cuenta (LVE/cgroups) eso
    solo pasa cuando algo escapa al límite, que es justo la firma de un incidente y no de demanda comercial.
    """
    d = metrics
    if start is not None:
        d = d[d["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d["timestamp"] <= pd.Timestamp(end)]
    if "event_account" in d and "is_incident" in d and d["is_incident"].any():
        return set(d.loc[d["is_incident"] & d["event_account"].ne(""), "event_account"]), "incidentes_etiquetados"
    if util is None:
        return set(), "sin_datos"
    over = util[list(RESOURCES)].max(axis=1) > OVER
    return set(util.loc[over, "account_id"]), "uso_por_encima_del_limite"


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Nombre de cada cluster por reglas deterministas sobre su centroide (en utilización, 0-1).

    Orden de prioridad: campañas > crecimiento > esporádico > intensivo > moderado. Si dos clusters reciben el
    mismo nombre se numeran («intensivo», «intensivo_2»), para que nadie confunda dos grupos distintos.
    """
    names: dict[int, str] = {}
    for idx, row in centroids.iterrows():
        p95 = row.get("p95_max", max(row.get(f"p95_{r}", 0.0) for r in RESOURCES))
        if row.get("concentracion_picos", 0.0) > 0.2:  # antes que la tendencia: una campaña final parece tendencia
            name = "campanas"
        elif row.get("tendencia", 0.0) > 0.35:
            name = "crecimiento"
        elif row.get("rafagas", 0.0) > 8:
            name = "esporadico"
        elif p95 > 0.8:
            name = "intensivo"
        else:
            name = "moderado"
        used = sum(1 for v in names.values() if v == name or v.startswith(name + "_"))
        names[int(idx)] = name if used == 0 else f"{name}_{used + 1}"
    return names


def _transform(d: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = d[columns].astype(float).copy()
    if "rafagas" in x:
        x["rafagas"] = np.log1p(x["rafagas"])
    return StandardScaler().fit_transform(x)


def cluster_accounts(feats_flagged: pd.DataFrame, k_range=range(3, 7), seeds: int = 20,
                     min_accounts: int = 15) -> dict:
    """KMeans sobre las cuentas marcadas; k por silueta entre los k estables (ARI medio ≥ 0,8 entre semillas)."""
    d = feats_flagged[feats_flagged["near_limit"]].copy()
    if len(d) < min_accounts:
        return {"available": False, "reason": f"solo {len(d)} cuentas marcadas (mínimo {min_accounts})", "features": d}
    columns = [c for c in CLUSTER_FEATURES if c in d]
    x = _transform(d, columns)
    candidates = {}
    for k in k_range:
        if k >= len(d):
            continue
        fits = [KMeans(n_clusters=k, random_state=s, n_init=10).fit(x) for s in range(seeds)]
        ref = fits[0].labels_
        stability = float(np.mean([adjusted_rand_score(ref, f.labels_) for f in fits[1:]]))
        candidates[k] = {"silhouette": float(silhouette_score(x, ref)), "stability": stability, "labels": ref}
    if not candidates:
        return {"available": False, "reason": "no hay suficientes cuentas para ningún k", "features": d}
    stable = {k: v for k, v in candidates.items() if v["stability"] >= 0.8} or candidates
    k = max(stable, key=lambda k: stable[k]["silhouette"])
    labels = candidates[k]["labels"]
    d["cluster_id"] = labels
    centroids = d.groupby("cluster_id")[columns].mean()  # en unidades originales: legible para negocio
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(d[columns], labels)
    return {"available": True, "k": k,
            "silhouette": {k_: round(v["silhouette"], 3) for k_, v in candidates.items()},
            "stability_by_k": {k_: round(v["stability"], 3) for k_, v in candidates.items()},
            "stability": candidates[k]["stability"], "features": d, "centroids": centroids,
            "cluster_names": name_clusters(centroids),
            "tree_rules": export_text(tree, feature_names=columns),
            "tree_fidelity": float(tree.score(d[columns], labels)), "columns": columns}


def vs_ground_truth(result: dict, feats: pd.DataFrame) -> dict | None:
    """Solo con datos simulados: ¿los clusters recuperan los arquetipos reales?"""
    if not result.get("available") or "archetype" not in feats:
        return None
    d = result["features"][["account_id", "cluster_id"]].merge(feats[["account_id", "archetype"]], on="account_id")
    return {"ari": float(adjusted_rand_score(d["archetype"], d["cluster_id"])),
            "cross_tab": {str(k): v for k, v in pd.crosstab(d["archetype"], d["cluster_id"]).to_dict().items()}}
