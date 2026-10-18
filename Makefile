.PHONY: install test lint all dashboard scale verify readme docker
install:   ; pip install -r requirements-dev.txt
test:      ; python -m pytest -q
lint:      ; ruff check .
all:       ; python -m capguard all --bench-seeds 3 --real --out outputs && python -m capguard readme
dashboard: ; streamlit run dashboard/app.py
scale:     ; python -m capguard scale --servers 300 --days 45
readme:    ; python -m capguard readme
verify:    ; bash scripts/verify.sh
docker:    ; docker build -t capacity-guard . && docker run --rm -p 8501:8501 capacity-guard
