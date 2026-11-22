"""Páginas: biblioteca de fórmulas y panel de ayuda."""
from __future__ import annotations

import streamlit as st

from .formulas import LIBRARY


def render_library(settings) -> None:
    st.title("Biblioteca de fórmulas")
    q = st.text_input("Buscar", "", placeholder="Ej.: cadena, Jacobiano, Newton…").strip().lower()
    topics = st.multiselect("Temas", list(LIBRARY), default=list(LIBRARY) if not q else list(LIBRARY))
    shown = 0
    for topic in topics:
        items = [f for f in LIBRARY[topic] if not q or q in (f["name"] + f["use"] + f["variables"]).lower()]
        if not items:
            continue
        st.subheader(topic)
        for f in items:
            shown += 1
            with st.expander(f["name"]):
                st.latex(f["latex"])
                st.markdown(f"**Variables:** {f['variables']}")
                st.markdown(f"**Condiciones de aplicación:** {f['conditions']}")
                st.markdown(f"**Cuándo se usa:** {f['use']}")
                st.markdown(f"**Ejemplo:** {f['example']}")
    if not shown:
        st.info("Ninguna fórmula coincide con la búsqueda.")


def render_help(settings) -> None:
    st.title("Ayuda y sintaxis")
    st.markdown("""
**Operadores:** `+ - * / ^` · potencia `x^2` · factorial `5!` · porcentaje `20%` · valor absoluto `abs(x)` o `|x|`.
La multiplicación implícita funciona: `2x`, `3(x+1)`, `2pi`.

| Quiere | Escriba |
|---|---|
| Raíz cuadrada / índice n | `sqrt(x)`, `√x`, `root(x, 3)` |
| Logaritmos | `ln(x)` natural · `log(x)` base 10 · `log(x, 5)` base 5 · `log2(x)` |
| Exponencial | `exp(x)` o `e^x` |
| Trigonométricas | `sin cos tan cot sec csc` · inversas `asin acos atan` (o `arcsin`…) |
| Hiperbólicas | `sinh cosh tanh` · inversas `asinh acosh atanh` |
| Constantes | `pi` (o π), `e`, `i`, `oo` (∞) |
| Complejos | `(2+3i)*(1-i)`, `abs(3+4i)`, `conj(1+i)`, `arg(i)` |
| Grados ↔ radianes | `rad(180)`, `deg(pi)` o cambie el modo en la barra lateral |
| Notación científica | `1.5e-3` |
| Vectores | `(1, 2, 3)` o `<1,2,3>` |
| Matrices | filas separadas por `;` o salto de línea: `1,2; 3,4` |
| Griegas | `theta`, `phi`, `rho` (o θ, φ, ρ) |
| EDO | `y'' - 3y' + 2y = exp(x)` (variable independiente `x`) |

**Convenciones.** `log(x)` es base 10 (como en una calculadora) y `ln(x)` es el natural. En esféricas, θ es el ángulo azimutal y φ el polar medido desde +z.

**Teclado.** Escriba directamente en el cuadro de expresión; **Enter** evalúa y guarda en el historial. Los botones insertan texto en la posición final.
Streamlit no permite atajos de teclado globales personalizados; el flujo es Tab/Shift+Tab para moverse y Enter/Espacio para pulsar botones.

**Modos.** *Estudiante*: procedimiento paso a paso con cuatro niveles (Básico, Intermedio, Avanzado, Universitario completo — el predeterminado).
*Profesional*: resultado inmediato sin explicaciones extensas. Puede cambiar en cualquier momento desde la barra lateral.

**Exacto vs. aproximado.** Los resultados llevan una etiqueta EXACTO o APROXIMADO. Si una integral no tiene antiderivada elemental, se calcula numéricamente y se advierte claramente.
""")
