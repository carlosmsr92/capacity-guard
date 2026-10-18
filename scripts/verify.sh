#!/usr/bin/env bash
# Verificación desde cero: entorno limpio -> instalar -> tests -> pipeline completo -> comprobaciones sobre los artefactos.
set -euo pipefail
cd "$(dirname "$0")/.."
V="$(mktemp -d)/venv"
python3 -m venv "$V" && source "$V/bin/activate"
pip install -q -r requirements.lock pytest ruff
echo "== lint =="; ruff check .
echo "== tests =="; python -m pytest -q
OUT="$(mktemp -d)"
echo "== pipeline completo =="; python -m capguard all --bench-seeds 0 --out "$OUT"
echo "== comprobaciones sobre los artefactos =="
python - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); r = json.loads((out / "results.json").read_text())
assert r["ingest"]["quality_after"]["ok"], "calidad tras reparar"
for k in ("report.html", "model.joblib", "RESULTS.md", "data/metrics.parquet", "data/scores.parquet"):
    assert (out / k).stat().st_size > 1000, k
assert "Big data" in (out / "report.html").read_text()
print("OK: artefactos completos y coherentes")
PY
echo "== VERIFICACIÓN SUPERADA =="
