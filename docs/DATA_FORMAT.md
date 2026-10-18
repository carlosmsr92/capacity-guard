# Formato de datos

CapGuard acepta métricas de servidor en CSV y las convierte a un formato canónico
antes de ejecutar el modelo. Las columnas obligatorias son `timestamp`, `server`,
`cpu`, `ram`, `iowait`, `net`, `load1` y `req`. `event` y `event_account` son
opcionales: sin `event` no se evalúan incidentes ni diagnóstico.

Se aceptan alias habituales (`host` para `server`, `mem` para `ram`,
`requests` o `rpm` para `req`, y `fecha`, `time` o `ts` para `timestamp`).
Las cabeceras se normalizan quitando acentos, espacios y diferencias entre
mayúsculas y minúsculas. Las columnas que no forman parte del esquema se
descartan (y se listan en el informe), para que ninguna etiqueta ajena llegue
al modelo. CPU, RAM e iowait pueden expresarse como fracción (0–1): la
conversión a porcentaje solo se aplica si CPU **y** RAM están ambas en 0–1
(un iowait real ocioso de 0,4 % no basta para decidirlo); para forzarla en una
columna concreta: `{"iowait": {"from": "fraction"}}`. `--tz Europe/Madrid`
convierte los timestamps a esa zona; las horas sin zona se interpretan como UTC.
Los timestamps ilegibles se descartan y se cuentan (si son ≥ 1 %, error, con
ejemplos de los valores que fallan).

**Fechas `dd/mm/aaaa` frente a `mm/dd/aaaa`:** se deduce de los datos (si algún
primer número es > 12, es día/mes). Si todos los días son ≤ 12 la fecha es
ambigua y no se adivina: hay que indicar el formato con
`--date-format "%d/%m/%Y %H:%M"`.

**Números:** se aceptan coma decimal (`;` como separador de campos) y
separador de miles a la española (`1.234,56`). Si una columna obligatoria
tiene más de un 5 % de valores no numéricos se rechaza con ejemplos (casi
siempre es un problema de formato, no ruido); por debajo, esos valores se
tratan como huecos y se cuentan en el informe. Si tras reparar no queda
ninguna fila válida, la ingesta falla con un mensaje claro.

La resolución requerida es de cinco minutos. Las series de un minuto se
agregan por servidor (media, salvo `load1`, que usa máximo); una resolución
superior a cinco minutos se rechaza. Los huecos de hasta 3 muestras (15 min) se
interpolan; en los huecos más largos no se inventa ninguna muestra.

**Etiquetas de `event`:** puede traer etiquetas propias (p. ej. `ddos`); toda
etiqueta que no sea un pico legítimo (`legit_surge`) cuenta como incidente y
recibe las acciones genéricas del runbook. Para traducir etiquetas se usa la
clave `event_values` del mapping: `{"event_values": {"campaña": "legit_surge",
"bots": "brute_force"}}`.

```bash
capguard validate-csv --metrics metrics.csv --out outputs/data
capguard all --source csv --metrics metrics.csv
capguard score --input metrics.csv --model outputs/model.joblib
```

Para columnas no estándar se puede pasar un JSON con un mapeo explícito:

```json
{"fecha_lectura": "timestamp", "hostname": "server", "cpu_pct": "cpu"}
```

```bash
capguard validate-csv --metrics metrics.csv --mapping mapping.json
```

Para entrenar con `capguard all --source csv` hacen falta al menos 28 días y
3 episodios de saturación tanto en validación como en test (si no, error
«Datos insuficientes»). `validate-csv` solo avisa: sirve para revisar un CSV
corto. Los periodos de entrenamiento, validación y test (60/15/25 %) y los
tres folds del walk-forward se calculan sobre la duración real de los datos.

## Qué se calcula según los datos

| Parte del análisis | Necesita |
|---|---|
| Aviso temprano de saturación (GBM, umbrales, walk-forward, calibración, costes) | Solo las métricas: la saturación se deriva de CPU, RAM e iowait |
| Tarea A (detección de incidentes), diagnóstico de causa, análisis por tipo de incidente | Columna `event` con etiquetas de incidentes |
| Clientes cerca de su límite | Datos de cuentas (`--accounts`) y de uso por cuenta (`--usage`) |
| Comparación de segmentos con la verdad | Solo la simulación (arquetipos conocidos) |

Lo que no se puede calcular se omite con un aviso en la consola, el informe y el
dashboard; no se inventan etiquetas.

## Datos de clientes (opcionales)

| Fichero | Columnas | Notas |
|---|---|---|
| `--accounts` | `account_id`, `server`, `plan` (+ `alta`, `sector`) | Sin datos personales: el identificador basta |
| `--usage` | `timestamp` (horario), `account_id`, `cpu` (cores), `ram` (MB), `io` (MB/s), `procesos` (+ `minutos_limitado`) | Uso **absoluto**; o añade columnas `<recurso>_limit` |
| `--plans` | `plan`, `precio_mes`, `cpu_cores`, `ram_mb`, `io_mbps`, `procesos` | Sin él se usa el catálogo por defecto |

Todo el análisis de clientes trabaja en **utilización** (uso / límite del plan):
900 MB son estar al límite en un plan de 1 GB y estar holgado en uno de 8 GB.
Si una cuenta tiene un plan desconocido, el paso de clientes se omite con un
aviso que lo dice.

**Guarda técnica:** con incidentes etiquetados y atribuidos (`event_account`),
las cuentas responsables nunca reciben una venta. Sin etiquetas, se marcan como
técnicas las cuentas que superan el límite de su plan (con aislamiento por
cuenta eso solo ocurre cuando algo escapa al límite, la firma de un incidente).

En `examples/` hay CSV de ejemplo (`capguard export-sample` los regenera),
incluido uno «sucio» con `;`, coma decimal, latin-1, fracciones y fechas
día/mes. Son 7 días: valen para `validate-csv`, no para entrenar.

Los datos de cuentas, uso y planes son opcionales para la ingesta inicial.
Cuando faltan, el informe declara las capacidades de clientes que se omiten;
no se crean etiquetas ni valores inventados.
