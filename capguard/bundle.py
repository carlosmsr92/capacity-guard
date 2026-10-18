"""Guardado y carga del modelo entrenado (joblib) con las versiones de las librerías con que se entrenó.

Seguridad: el bundle es un pickle (joblib), y cargar un pickle ejecuta código. Carga solo modelos generados por ti
o por tu propio pipeline, nunca un model.joblib descargado de una fuente no confiable.

Un pickle de scikit-learn/pandas solo es fiable con las mismas versiones; si no coinciden se avisa con un mensaje
claro en lugar de un error críptico al deserializar.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import joblib

from . import __version__

PINNED = ("scikit-learn", "pandas", "numpy")


def _versions() -> dict[str, str]:
    return {p: version(p) for p in PINNED}


def save_model(path: str | Path, **parts) -> Path:
    path = Path(path)
    joblib.dump({**parts, "capguard_version": __version__, "versions": _versions()}, path)
    return path


def load_model(path: str | Path) -> dict:
    try:
        bundle = joblib.load(path)
    except Exception as e:
        raise SystemExit(f"No se pudo cargar {path} ({type(e).__name__}: {e}). Suele deberse a versiones distintas de "
                         f"scikit-learn/pandas: reentrena con `python -m capguard all` en este entorno.") from e
    trained = bundle.get("versions", {})
    diff = {p: (v, trained[p]) for p, v in _versions().items() if p in trained and trained[p] != v}
    if diff:
        print("[aviso] versiones distintas a las del entrenamiento: "
              + ", ".join(f"{p} {now} (entrenado con {then})" for p, (now, then) in diff.items()))
    return bundle
