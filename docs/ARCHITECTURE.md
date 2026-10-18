# Arquitectura

```mermaid
flowchart LR
  A[Métricas cada 5 min<br/>agente / Prometheus] --> B[Contrato de datos<br/>quality.validate]
  B -->|errores| X[Bloquea y avisa]
  B --> C[Reparación trazable<br/>quality.repair]
  C --> D[(Parquet + zstd)]
  D --> E[DuckDB<br/>consultas y escala]
  D --> F[Features causales<br/>baseline estacional MAD]
  F --> G1[Umbral / Z-score]
  F --> G2[Isolation Forest]
  F --> G3[GBM: riesgo 30 min]
  F --> G4[Autoencoder]
  G2 --> H[Diagnóstico de causa]
  G3 --> I[Alarma con presupuesto de falsas alarmas]
  D --> N[Cuentas y uso horario]
  N --> O[Proximidad al límite y clustering]
  O --> P[Oferta en borrador · revisión humana]
  H --> J[Evidencia + explicación<br/>plantilla / LLM con guardarraíl]
  I --> J
  J --> K[Técnico de guardia<br/>decide y actúa]
  D --> L[Dashboard BI]
  I --> L
  G3 --> M[MLflow]
```

## Decisiones

* **Evaluación por evento**: ¿saltó la alarma?, ¿con cuántos minutos de antelación?, ¿cuántas falsas alarmas por día?
* **Partición temporal** y variables **causales** (un test lo comprueba); walk-forward para ver la estabilidad.
* **Umbrales por presupuesto de falsas alarmas** calibrado en validación, no cifras arbitrarias.
* **Fallos visibles**: los datos malos se miden y se reparan con rastro (`imputed`), los huecos largos no se inventan,
  el LLM nunca puede publicar una cifra que no esté en la evidencia.
* **Humano en el bucle**: el sistema no actúa solo ni atiende al cliente.
* **Clientes**: las causas técnicas bloquean la venta; todo borrador queda en `pendiente_revision`.

## Datos por cuenta

La carga de cada servidor simulado es la suma del uso de CPU de sus cuentas dividida por la capacidad de un servidor
típico (escala fija para toda la flota): un servidor con clientes más intensivos está de verdad más cargado.

`segments.py` trabaja siempre en **utilización** (uso / límite del plan): marca las cuentas que en los últimos 14 días
han pasado ≥ 5 días por encima del 85 % de algún límite o ≥ 3 días limitadas, valida la regla hacia delante (¿siguen
cerca del límite las 2 semanas siguientes?, frente a una regla ingenua), agrupa las marcadas con KMeans (k por silueta
entre los k estables) y nombra cada grupo con reglas fijas sobre su centroide, más un árbol sustituto legible.

Las cuentas con causa técnica (incidente atribuido o, sin etiquetas, uso por encima del límite del plan) se excluyen
del clustering y nunca reciben una venta. `offers.py` aplica una política determinista (upgrade, ampliación temporal,
consejo, plan máximo) con tope de descuento, enfriamiento entre ofertas y antigüedad mínima; los textos pasan el mismo
guardarraíl de cifras que las alarmas y todo queda en `pendiente_revision`. Los CSV sin datos de cuentas omiten esta
capacidad sin inventar información.

## Producción (no implementado aquí)

Puntuación por lotes cada 5 min desde Prometheus con `capguard score`, reentrenamiento semanal con walk-forward como
puerta de calidad, y revisión del presupuesto de alarmas con el equipo de guardia.
