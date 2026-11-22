"""CalcPro — calculadora científica, CAS, graficador y tutor.

Ejecutar:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="CalcPro · Calculadora científica", page_icon="∑", layout="wide",
                   initial_sidebar_state="expanded")

from ui import page_algebra, page_calculator, page_calculus, page_coordinates, page_formulas, page_graph, page_tools  # noqa: E402
from ui import theme, widgets  # noqa: E402

widgets.state_init()
settings = widgets.sidebar()
theme.apply(settings.theme)

PAGES = {
    "🧮 Calculadora": page_calculator.render,
    "∫ Cálculo": page_calculus.render,
    "📐 Coordenadas y vectores": page_coordinates.render,
    "📈 Graficador": page_graph.render,
    "🔢 Álgebra y matrices": page_algebra.render,
    "🛠 Herramientas": page_tools.render,
    "📚 Biblioteca de fórmulas": page_formulas.render_library,
    "❓ Ayuda": page_formulas.render_help,
}

page = st.sidebar.radio("Módulo", list(PAGES), key="page")
PAGES[page](settings)
