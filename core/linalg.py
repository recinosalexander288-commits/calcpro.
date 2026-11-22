"""Álgebra lineal exacta (matrices, determinantes, valores propios...)."""
from __future__ import annotations

import sympy as sp

from .parser import ExpressionError

OPERATIONS = [
    "Determinante", "Inversa", "Transpuesta", "Rango", "Traza",
    "Valores propios", "Vectores propios", "Polinomio característico",
    "Forma escalonada reducida (RREF)", "Espacio nulo", "Diagonalizar (A = P D P⁻¹)",
    "Descomposición LU", "Potencia Aⁿ", "A + B", "A − B", "A × B", "Resolver A·x = b",
]
NEEDS_B = {"A + B", "A − B", "A × B", "Resolver A·x = b"}


def _square(A):
    if A.rows != A.cols:
        raise ExpressionError("Esta operación requiere una matriz cuadrada.")


def operate(op: str, A: sp.Matrix, B: sp.Matrix | None = None, n: int = 2) -> list[tuple[str, str]]:
    """Devuelve una lista de (título, LaTeX)."""
    L = sp.latex
    if op in NEEDS_B and B is None:
        raise ExpressionError("Esta operación necesita la matriz B.")

    if op == "Determinante":
        _square(A)
        return [("det(A)", L(A.det()))]
    if op == "Inversa":
        _square(A)
        d = sp.simplify(A.det())
        if d == 0:
            raise ExpressionError("La matriz es singular (det = 0): no tiene inversa.")
        return [("det(A)", L(d)), ("A⁻¹", L(A.inv()))]
    if op == "Transpuesta":
        return [("Aᵀ", L(A.T))]
    if op == "Rango":
        return [("rango(A)", L(A.rank()))]
    if op == "Traza":
        _square(A)
        return [("tr(A)", L(A.trace()))]
    if op == "Valores propios":
        _square(A)
        ev = A.eigenvals()
        return [(f"λ (multiplicidad {m})", L(sp.simplify(v))) for v, m in ev.items()]
    if op == "Vectores propios":
        _square(A)
        out = []
        for val, mult, vecs in A.eigenvects():
            for v in vecs:
                out.append((f"λ = {L(sp.simplify(val))} (mult. {mult})", L(sp.simplify(v))))
        return out
    if op == "Polinomio característico":
        _square(A)
        lam = sp.Symbol("lambda")
        return [("p(λ)", L(sp.factor(A.charpoly(lam).as_expr())))]
    if op == "Forma escalonada reducida (RREF)":
        R, piv = A.rref()
        return [("RREF", L(R)), ("columnas pivote", L([p + 1 for p in piv]))]
    if op == "Espacio nulo":
        ns = A.nullspace()
        if not ns:
            return [("Espacio nulo", r"\{\mathbf{0}\}")]
        return [(f"Vector base {i + 1}", L(v)) for i, v in enumerate(ns)]
    if op.startswith("Diagonalizar"):
        _square(A)
        if not A.is_diagonalizable():
            raise ExpressionError("La matriz no es diagonalizable sobre este cuerpo.")
        P, D = A.diagonalize()
        return [("P", L(P)), ("D", L(D)), ("P⁻¹", L(P.inv()))]
    if op == "Descomposición LU":
        Lm, U, perm = A.LUdecomposition()
        return [("L", L(Lm)), ("U", L(U)), ("permutaciones", L(perm))]
    if op.startswith("Potencia"):
        _square(A)
        return [(f"A^{n}", L(A ** int(n)))]
    if op == "A + B":
        if A.shape != B.shape:
            raise ExpressionError("A y B deben tener las mismas dimensiones.")
        return [("A + B", L(A + B))]
    if op == "A − B":
        if A.shape != B.shape:
            raise ExpressionError("A y B deben tener las mismas dimensiones.")
        return [("A − B", L(A - B))]
    if op == "A × B":
        if A.cols != B.rows:
            raise ExpressionError(f"No se puede multiplicar: A tiene {A.cols} columnas y B tiene {B.rows} filas.")
        return [("A · B", L(A * B))]
    if op == "Resolver A·x = b":
        if B.cols != 1 or B.rows != A.rows:
            raise ExpressionError("B debe ser un vector columna con tantas filas como A.")
        syms = sp.symbols(f"x1:{A.cols + 1}")
        sol = sp.linsolve((A, B), *syms)
        if sol == sp.S.EmptySet:
            raise ExpressionError("El sistema es incompatible (no tiene solución).")
        return [("Solución", L(sol))]
    raise ExpressionError(f"Operación desconocida: {op}")
