"""Calculadora científica: evaluación exacta y numérica."""
from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

from .parser import ExpressionError, parse_expression


@dataclass
class CalcResult:
    exact: sp.Basic
    approx: str
    latex: str
    is_exact: bool
    kind: str  # "numérico" | "simbólico"


def format_number(value, precision: int = 10, scientific: bool = False) -> str:
    """Aproximación decimal con `precision` cifras significativas."""
    try:
        v = sp.N(value, max(int(precision), 1) + 2)
    except Exception:  # noqa: BLE001
        return str(value)
    if scientific and v.is_real:
        return f"{float(v):.{int(precision)}e}"
    txt = str(sp.N(v, int(precision)))
    return re.sub(r"\*?I\b", "i", txt)


def evaluate(text: str, angle_mode: str = "rad", precision: int = 10, scientific: bool = False) -> CalcResult:
    expr = parse_expression(text, angle_mode)
    exact = expr if expr.is_Atom else sp.simplify(expr)
    if exact.has(sp.zoo, sp.nan):
        raise ExpressionError("La operación no está definida (división entre cero o indeterminación).")
    if exact.free_symbols:
        return CalcResult(exact, "", sp.latex(exact), True, "simbólico")

    is_exact = not exact.has(sp.Float)
    approx = format_number(exact, precision, scientific)
    latex = sp.latex(exact)
    if not (exact.is_Integer or exact.is_Float or exact == sp.oo or exact == -sp.oo):
        latex += r" \approx " + approx.replace("e", r"\times 10^{") + ("}" if "e" in approx else "")
    return CalcResult(exact, approx, latex, is_exact, "numérico")


def complex_info(z) -> dict:
    """Forma binómica, polar y exponencial de un número complejo."""
    z = sp.sympify(z)
    re_, im_ = sp.simplify(sp.re(z)), sp.simplify(sp.im(z))
    r = sp.simplify(sp.Abs(z))
    th = sp.simplify(sp.arg(z))
    return {
        "binómica": sp.latex(re_ + sp.I * im_),
        "parte real": sp.latex(re_),
        "parte imaginaria": sp.latex(im_),
        "módulo": sp.latex(r),
        "argumento (rad)": sp.latex(th),
        "argumento (grados)": sp.latex(sp.simplify(th * 180 / sp.pi)),
        "polar": sp.latex(r) + r"\,\left(\cos " + sp.latex(th) + r" + i\sin " + sp.latex(th) + r"\right)",
        "exponencial": sp.latex(r) + r"\,e^{i\," + sp.latex(th) + "}",
        "conjugado": sp.latex(sp.conjugate(z)),
    }
