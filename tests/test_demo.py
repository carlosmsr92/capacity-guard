"""Instantánea versionada demo/: lo que ve el dashboard desplegado sin ejecutar el pipeline (p. ej. Streamlit Cloud)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"


def test_demo_is_complete_small_and_has_no_pickle():
    from capguard.demo import REQUIRED
    assert all((DEMO / f).exists() for f in REQUIRED)
    files = [p for p in DEMO.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) < 10_000_000
    assert not any(p.suffix in (".joblib", ".pkl", ".pickle") for p in files)


def test_demo_is_not_stale():
    """Si cambian los resultados y no se regenera la demo, el dashboard desplegado contradiría al README."""
    assert (DEMO / "results.json").read_bytes() == (ROOT / "outputs" / "results.json").read_bytes(), \
        "demo/ está desfasada: ejecuta `python -m capguard export-demo` tras `python -m capguard all`"


def test_dashboard_renders_the_demo_and_says_so(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CAPGUARD_OUT", str(DEMO))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception and len(at.tabs) == 7
    assert any("demostración" in i.value for i in at.info)
    at.selectbox(key="customer_offer").set_value("upgrade").run()
    at.selectbox(key="customer_account").set_value(at.selectbox(key="customer_account").options[0]).run()
    assert not at.exception
