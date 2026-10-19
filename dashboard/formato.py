"""Formato español de números para lo que se muestra: miles con punto y decimales con coma (103.675 · 0,29)."""
from __future__ import annotations

import re


def _es(texto: str) -> str:
    """Intercambia los separadores de un número ya formateado en inglés: 1,234.5 → 1.234,5."""
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fni(x) -> str:
    """Entero con separador de miles (equivale a {x:,})."""
    return _es(f"{x:,}")


def fn0(x) -> str:
    return _es(f"{x:,.0f}")


def fn1(x) -> str:
    return _es(f"{x:,.1f}")


def fn2(x) -> str:
    return _es(f"{x:,.2f}")


def fsn1(x) -> str:
    """Con signo: +1,5 / -2,3."""
    return _es(f"{x:+,.1f}")


def fsn2(x) -> str:
    return _es(f"{x:+,.2f}")


def fp1(x) -> str:
    """Fracción como porcentaje con un decimal: 0,123 → 12,3%."""
    return _es(f"{x:,.1%}")


def variacion(actual, anterior):
    """Variación porcentual frente al periodo anterior, para el delta de st.metric (None si no hay base)."""
    if not anterior:
        return None
    return fsn1((actual - anterior) / anterior * 100) + "%"


def texto_es(texto) -> str:
    """Decimales en coma dentro de un texto ya redactado (17.5 → 17,5); no toca fechas, horas ni identificadores."""
    return re.sub(r"(?<![\w.,:])(-?\d+)\.(\d+)(?![\w.:-])", r"\1,\2", str(texto))


def tabla(df, decimales: int | dict = 2):
    """Styler para st.dataframe con separadores españoles; los valores siguen siendo numéricos (ordenan bien).

    decimales: los mismos para todas las columnas, o {columna: decimales} (el resto con 2).
    """
    estilo = df.style.format(precision=decimales if isinstance(decimales, int) else 2, decimal=",", thousands=".",
                             na_rep="–")
    if isinstance(decimales, dict):
        for col, d in decimales.items():
            estilo = estilo.format(precision=d, decimal=",", thousands=".", na_rep="–", subset=[col])
    return estilo


def grafico(fig, **kwargs):
    """st.plotly_chart con separadores españoles en ejes, etiquetas y textos emergentes."""
    import streamlit as st

    fig.update_layout(separators=",.")  # Plotly: primer carácter = decimal, segundo = miles
    return st.plotly_chart(fig, **kwargs)
