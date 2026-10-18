"""Seguimiento de experimentos con MLflow (MLOps). Si MLflow no está disponible, el pipeline sigue sin él."""
from __future__ import annotations

import os
import warnings
from pathlib import Path


def log_run(run_name: str, params: dict, metrics: dict, artifacts: list[str | Path] = (),
            tracking_dir: str | Path = "outputs/mlruns", experiment: str = "capacity-guard") -> str | None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # almacén local en ficheros, sin servidor
    try:
        import mlflow
    except ImportError as e:  # p. ej. mlflow no instalado o una versión antigua incompatible con este Python
        print(f"[aviso] MLflow no disponible, la ejecución no se registra: {e}")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and v == v})
                for a in artifacts:
                    if Path(a).exists():
                        mlflow.log_artifact(str(a))
                return run.info.run_id
    except Exception as e:  # el seguimiento nunca debe romper el entrenamiento
        print(f"[aviso] MLflow no registró la ejecución: {e}")
        return None
