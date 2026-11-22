"""Página: Calculadora científica."""
from __future__ import annotations

import streamlit as st

from core.calculator import complex_info, evaluate
from core.parser import ExpressionError
from . import widgets as W

KEYPAD = {
    "Básico": [
        [("7", "7"), ("8", "8"), ("9", "9"), ("÷", "/"), ("(", "("), (")", ")")],
        [("4", "4"), ("5", "5"), ("6", "6"), ("×", "*"), ("xʸ", "^"), ("n!", "!")],
        [("1", "1"), ("2", "2"), ("3", "3"), ("−", "-"), ("√", "sqrt("), ("ⁿ√", "root(")],
        [("0", "0"), (".", "."), ("π", "pi"), ("+", "+"), ("e", "e"), ("i", "i")],
        [("%", "%"), ("|x|", "abs("), ("ANS", "ANS"), ("EXP", "e"), (",", ","), ("∞", "oo")],
    ],
    "Trig / Hiperb.": [
        [("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("cot", "cot("), ("sec", "sec("), ("csc", "csc(")],
        [("sin⁻¹", "asin("), ("cos⁻¹", "acos("), ("tan⁻¹", "atan("), ("sinh", "sinh("), ("cosh", "cosh("), ("tanh", "tanh(")],
        [("sinh⁻¹", "asinh("), ("cosh⁻¹", "acosh("), ("tanh⁻¹", "atanh("), ("rad(", "rad("), ("deg(", "deg("), ("θ", "theta")],
    ],
    "Log / Exp": [
        [("ln", "ln("), ("log₁₀", "log("), ("log₂", "log2("), ("logₐ", "log(x, "), ("eˣ", "exp("), ("10ˣ", "10^")],
        [("x²", "^2"), ("x³", "^3"), ("1/x", "1/("), ("Γ", "Gamma("), ("nCr", "binomial("), ("mod", "mod(")],
        [("⌊x⌋", "floor("), ("⌈x⌉", "ceil("), ("sgn", "sign("), ("gcd", "gcd("), ("lcm", "lcm("), ("max", "max(")],
    ],
    "Complejos": [
        [("i", "i"), ("Re", "re("), ("Im", "im("), ("conj", "conj("), ("arg", "arg("), ("|z|", "abs(")],
    ],
}


def _insert(token: str) -> None:
    st.session_state.expr = st.session_state.get("expr", "") + token


def _insert_memory() -> None:
    _insert(f"({st.session_state.memory})")


def _clear() -> None:
    st.session_state.expr = ""


def _back() -> None:
    st.session_state.expr = st.session_state.get("expr", "")[:-1]


def _prepare(expr: str) -> str:
    return expr.replace("ANS", f"({st.session_state.ans})")


def _commit(settings) -> None:
    expr = st.session_state.expr.strip()
    if not expr:
        return
    try:
        r = evaluate(_prepare(expr), settings.angle, settings.precision, settings.scientific)
    except ExpressionError:
        return
    st.session_state.ans = str(r.exact)
    st.session_state.last_exact = r.exact
    W.add_history("Calculadora", expr, r.latex, r.approx or str(r.exact))


def render(settings) -> None:
    st.title("Calculadora científica")
    st.caption("Escriba con el teclado o use los botones. Enter evalúa y guarda en el historial. "
               "Ejemplos: `sqrt(2)*pi`, `sin(pi/6)`, `log(1000)`, `ln(e^3)`, `(2+3i)*(1-i)`, `5!`, `20%*80`.")

    st.text_input("Expresión", key="expr", placeholder="Ej.: integrate no — aquí sólo expresiones: 2^10 + sqrt(144)",
                  on_change=_commit, args=(settings,))

    tabs = st.tabs(list(KEYPAD))
    for tab, (name, rows) in zip(tabs, KEYPAD.items()):
        with tab:
            for r, row in enumerate(rows):
                cols = st.columns(len(row))
                for c, (label, token) in zip(cols, row):
                    with c:
                        W.btn(label, key=f"k_{name}_{r}_{label}", on_click=_insert, args=(token,))

    a, b, c, d, e, f = st.columns(6)
    with a:
        W.btn("⌫", key="bk", on_click=_back, help="Borrar último carácter")
    with b:
        W.btn("C", key="cl", on_click=_clear, help="Borrar la expresión")
    with c:
        eq = W.btn("=", key="eq", type="primary")
    with d:
        mplus = W.btn("M+", key="mp", help="Sumar el último resultado a la memoria")
    with e:
        W.btn("MR", key="mr", help="Insertar la memoria en la expresión", on_click=_insert_memory)
    with f:
        mclr = W.btn("MC", key="mc", help="Borrar la memoria")

    if mplus and st.session_state.last_exact is not None:
        import sympy as sp
        st.session_state.memory = str(sp.simplify(sp.sympify(st.session_state.memory) + st.session_state.last_exact))
        st.toast(f"Memoria: {st.session_state.memory}")
    if mclr:
        st.session_state.memory = "0"
        st.toast("Memoria borrada")
    st.caption(f"Memoria: `{st.session_state.memory}` · ANS: `{st.session_state.ans}`")

    expr = st.session_state.expr.strip()
    if eq and expr:
        _commit(settings)
    if not expr:
        W.history_panel()
        return

    try:
        res = evaluate(_prepare(expr), settings.angle, settings.precision, settings.scientific)
    except ExpressionError as err:
        st.error(f"⚠️ {err}")
        W.history_panel()
        return
    except Exception as err:  # noqa: BLE001
        st.error(f"⚠️ No se pudo evaluar ({type(err).__name__}).")
        W.history_panel()
        return

    if res.kind == "simbólico":
        st.info("La expresión contiene variables: se muestra su forma simplificada (resultado simbólico).")
        W.show_result(res.latex, str(res.exact), None, True)
    else:
        W.show_result(res.latex, res.approx if not res.is_exact else str(res.exact),
                      res.approx if not (res.exact.is_Integer or res.exact.is_Float) else None, res.is_exact)
        if res.exact.has(__import__("sympy").I):
            with st.expander("Detalle del número complejo"):
                for k, v in complex_info(res.exact).items():
                    st.markdown(f"**{k}**")
                    st.latex(v)
    st.markdown("---")
    W.history_panel()
