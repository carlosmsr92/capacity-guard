FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero (capa cacheada mientras no cambie el lock)
COPY requirements.lock .
RUN pip install -r requirements.lock

RUN useradd --create-home --uid 1000 app && chown app /app
USER app

COPY --chown=app capguard ./capguard
COPY --chown=app dashboard ./dashboard
COPY --chown=app .streamlit ./.streamlit
COPY --chown=app docs ./docs

# Los artefactos (modelo, datos, informe) se generan al construir la imagen: el contenedor arranca al instante.
# Prueba de escala reducida para no alargar el build; las cifras grandes vienen de docs/scale_benchmark.json.
RUN python -m capguard all --bench-seeds 0 --scale-servers 20 --scale-days 14 --out outputs

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
