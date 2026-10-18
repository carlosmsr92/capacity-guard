# Preguntas probables y respuestas honestas

**¿Por qué datos simulados?** No tenía métricas reales de un hosting. El simulador permite etiquetas exactas
(qué es incidente, cuándo empieza la saturación) y medir la antelación de forma honesta. La contrapartida: las cifras no
valen para producción. Por eso hay una comprobación con series reales públicas (NAB) y la ruta `score` para métricas propias.

**¿El modelo es mucho mejor que lo simple?** Frente al umbral fijo, sí en aviso temprano. Frente al z-score estacional la
ventaja es moderada. Lo cuento así porque es lo que miden el walk-forward y el test.

**¿Cómo evitaste fuga de información?** Partición temporal, variables solo con el pasado (hay un test que cambia el futuro
y comprueba que las variables pasadas no cambian), calibración de umbrales en validación y una prueba de cordura en el
diagnóstico: con etiquetas barajadas el acierto cae al azar.

**¿Por qué el autoencoder no gana?** Su ventana de 1 h tarda en "olvidar" y avisa tarde; además es un MLP de scikit-learn.
Lo dejé porque comparar y descartar con datos es parte del trabajo.

**¿Qué pasó con la calibración?** La isotónica no mejoró el Brier ni el AUC; la descarté y lo reporto.

**¿De dónde salen los costes en euros?** Son supuestos de ejemplo, editables en el dashboard. Con falsas alarmas caras el
umbral estático vuelve a ser la mejor opción; el modelo solo compensa si avisar a tiempo vale mucho más.

**¿Por qué DuckDB?** Consulta Parquet sin cargarlo en RAM y reproduce exactamente las variables de pandas (hay un test).
A ~26 M de filas funcionó en una máquina de 3 GB; pandas a ese tamaño no se probó.

**¿El LLM funciona?** La ruta está verificada con un cliente simulado (texto válido, cifra inventada, servicio caído). No se
ejecutó contra el servicio real por falta de clave. El guardarraíl impide que publique cifras que no estén en la evidencia.

**¿Qué haría con datos reales?** Reentrenar, recalibrar el presupuesto con el equipo de guardia, validar con incidentes
históricos y añadir señales de logs (p. ej., intentos a wp-login). Esperaría peor diagnóstico de causa que el simulado.

**¿Privacidad?** Cuando se activa la segmentación se usan datos de uso por cuenta seudonimizados, nunca nombres ni
contenido. La base legal debe ser interés legítimo documentado u opt-in, según el contexto, con minimización,
retención limitada y controles de acceso RGPD. No hay patrones oscuros: las ofertas son borradores y una persona
revisa cada caso. Una causa técnica se deriva a soporte, no se convierte en un upgrade.
