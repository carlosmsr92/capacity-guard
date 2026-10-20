"""Marco de marca CMSR92 común a los dashboards del portfolio: icono de pestaña, banda de cabecera con el logo y pie.

El mismo fichero está en ecommerce-analytics, cine y capacity-guard (junto a su carpeta assets/ con los PNG de marca,
exportados de los SVG de github.com/carlosmsr92/cmsr92/tree/main/brand-assets): si se cambia, se cambia en los tres.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
ICONO = str(ASSETS / "cmsr92-simbolo.png")  # page_icon de st.set_page_config
PORTFOLIO = "https://carlosmsr92.github.io/cmsr92/"

# Banda con fondo propio en el turquesa oscuro de marca: el texto blanco tiene 4,7:1 en su tono más claro (#087F93),
# así que se lee igual en el tema claro y en el oscuro. El logo es la variante blanca del logo horizontal.
_CSS = """<style>
.cmsr-banda {background: linear-gradient(90deg, #065A68 0%, #087F93 100%); color: #fff; border-radius: 8px;
  padding: 1.1rem 1.4rem; margin-bottom: 1rem; display: flex; flex-wrap: wrap; align-items: center;
  justify-content: space-between; gap: 0.75rem 1.5rem}
.cmsr-banda .cmsr-titulo {font-size: 2rem; font-weight: 700; line-height: 1.2; color: #fff}
.cmsr-banda .cmsr-sub {font-size: 0.95rem; margin-top: 0.25rem; color: #fff}
.cmsr-banda .cmsr-logo {height: 44px; width: auto}
.cmsr-pie {text-align: center; font-size: 0.85rem; opacity: 0.8; margin-top: 2.5rem; padding: 1.5rem 0 0.5rem;
  border-top: 1px solid rgba(128, 128, 128, 0.3)}
.cmsr-pie a {color: inherit}
@media (max-width: 640px) {
  .cmsr-banda .cmsr-titulo {font-size: 1.5rem}
  .cmsr-banda .cmsr-logo {height: 34px}
}
</style>"""


def cabecera(titulo: str, subtitulo: str) -> None:
    """Banda superior: título y subtítulo a la izquierda, logo CMSR92 a la derecha (pasa debajo en móvil)."""
    logo = base64.b64encode((ASSETS / "cmsr92-logo-blanco.png").read_bytes()).decode()
    st.markdown(f'{_CSS}<div class="cmsr-banda"><div><div class="cmsr-titulo">{titulo}</div>'
                f'<div class="cmsr-sub">{subtitulo}</div></div>'
                f'<img class="cmsr-logo" alt="CMSR92" src="data:image/png;base64,{logo}"></div>', unsafe_allow_html=True)


def pie(proyecto: str) -> None:
    """Firma al final de la página, con enlace al portfolio."""
    st.markdown(f'<div class="cmsr-pie">{proyecto} · Carlos Manuel Suárez Rivero · CMSR92 · '
                f'<a href="{PORTFOLIO}" target="_blank" rel="noopener">Portfolio</a></div>', unsafe_allow_html=True)
