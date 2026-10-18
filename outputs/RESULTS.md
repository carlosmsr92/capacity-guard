*Generado por `python -m capguard all` · semilla 7 · 8 servidores × 45 días · datos simulados.*

**Aviso temprano de saturación (30 min), periodo de test, presupuesto ≤ 0.5 falsas alarmas por servidor y día en validación:**

| Método | Saturaciones avisadas ≥10 min antes | Falsas alarmas / servidor·día |
|---|---|---|
| Umbral estático | 23 % | 0.04 |
| Z-score estacional | 41 % | 0.41 |
| Isolation Forest | 28 % | 0.50 |
| Pronosticador GBM | 33 % | 0.29 |
| Autoencoder (deep learning) | 13 % | 0.60 |

**Robustez: el mismo experimento en 3 mundos simulados distintos (semillas 8, 9, 10):**

| Método | Avisadas ≥10 min antes (media ± desv.) | Falsas alarmas / servidor·día |
|---|---|---|
| Umbral estático | 23 % ± 6 | 0.06 ± 0.04 |
| Z-score estacional | 37 % ± 6 | 0.46 ± 0.13 |
| Isolation Forest | 30 % ± 6 | 0.67 ± 0.36 |
| Pronosticador GBM | 41 % ± 9 | 0.42 ± 0.04 |

**Validación walk-forward (3 periodos futuros):**

| Método | Avisadas ≥10 min antes (media ± desv.) | Falsas alarmas / servidor·día |
|---|---|---|
| Umbral estático | 17 % ± 7 | 0.04 |
| Z-score estacional | 27 % ± 16 | 0.45 |
| Isolation Forest | 18 % ± 17 | 0.46 |
| Pronosticador GBM | 35 % ± 10 | 0.48 |

**Coste con supuestos ilustrativos (no avisar 200 €, falsa alarma 15 €, avisar a tiempo 20 €):**

| Método | Coste en test (€) | Ahorro vs umbral |
|---|---|---|
| Umbral estático | 6,240 | 0 % |
| Z-score estacional | 5,475 | 12 % |
| Isolation Forest | 6,495 | -4 % |
| Pronosticador GBM | 5,850 | 6 % |

- **Calidad de datos:** el feed sucio falla 6 comprobaciones y tras reparar pasa el contrato; 5 de 103,680 filas irrecuperables; error de imputación ≤ 13 % de la desviación típica.
- **Diagnóstico de causa:** 97 % de acierto al saltar la alarma (predecir siempre la clase mayoritaria: 22 %; con etiquetas barajadas 16 %). Las firmas simuladas son limpias: con datos reales hay que esperar menos.
- **Calibración isotónica:** Brier 0.0080 → 0.0079 (mejora).

**Prueba de escala (4 variables de producción; pandas vs DuckDB sobre Parquet):**

| Escala | Motor | Tiempo (s) | Memoria pico (MB) | Máquina | Origen |
|---|---|---|---|---|---|
| 864,000 filas | pandas | 0.52 | 432 | 32 vCPU, 47 GB RAM | esta ejecución |
| 864,000 filas | duckdb | 0.09 | 464 | 32 vCPU, 47 GB RAM | esta ejecución |
| 3,888,000 filas | pandas | 5.04 | 1055 | 1 vCPU, 3 GB RAM | medición previa (docs/scale_benchmark.json) |
| 3,888,000 filas | duckdb | 2.99 | 739 | 1 vCPU, 3 GB RAM | medición previa (docs/scale_benchmark.json) |
| 25,920,000 filas | duckdb | 20.83 | 3468 | 1 vCPU, 3 GB RAM | medición previa (docs/scale_benchmark.json) |

- **Clientes cerca de su límite:** 52 de 320 cuentas marcadas; 45 borradores comerciales, todos en `pendiente_revision` (nada se envía solo); 10 cuentas con causa técnica excluidas de cualquier venta; 6 segmentos (campanas, campanas_2, campanas_3, crecimiento, esporadico, intensivo); ARI frente a los arquetipos simulados 0.95; precisión hacia delante de la regla 80 % (ingenua 73 %). La aceptación de las ofertas es un supuesto ilustrativo.