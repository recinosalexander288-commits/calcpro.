"""Álgebra: simplificación, ecuaciones, sistemas y polinomios."""
from __future__ import annotations

import numpy as np
import sympy as sp

from .parser import ExpressionError

SIMPLIFY_OPS = {
    "Simplificar": sp.simplify,
    "Expandir": sp.expand,
    "Factorizar": sp.factor,
    "Fracciones parciales": sp.apart,
    "Combinar fracciones": sp.together,
    "Simplificar trigonometría": sp.trigsimp,
    "Expandir trigonometría": sp.expand_trig,
    "Simplificar potencias": sp.powsimp,
    "Racionalizar denominador": sp.radsimp,
    "Combinar logaritmos": lambda e: sp.logcombine(e, force=True),
    "Expandir logaritmos": lambda e: sp.expand_log(e, force=True),
}


def apply_operation(name: str, expr: sp.Basic) -> sp.Basic:
    try:
        return SIMPLIFY_OPS[name](expr)
    except KeyError as exc:
        raise ExpressionError(f"Operación desconocida: {name}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"No se pudo aplicar «{name}» a esta expresión ({type(exc).__name__}).") from exc


def numeric_roots(expr, var, lo: float = -10.0, hi: float = 10.0, n: int = 4000) -> list:
    """Raíces reales aproximadas por cambio de signo + Brent (descarta polos)."""
    from scipy.optimize import brentq

    f = sp.lambdify(var, expr, "numpy")
    xs = np.linspace(lo, hi, n)
    try:
        with np.errstate(all="ignore"):
            ys = np.broadcast_to(np.asarray(f(xs), dtype=float), xs.shape)
    except Exception:  # noqa: BLE001
        return []
    roots: list[float] = []
    for i in range(n - 1):
        a, b = ys[i], ys[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0:
            roots.append(float(xs[i]))
        elif a * b < 0:
            try:
                r = brentq(lambda t: float(f(t)), xs[i], xs[i + 1])
            except Exception:  # noqa: BLE001
                continue
            if abs(float(f(r))) < 1e-6:
                roots.append(float(r))
    out: list[float] = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-8:
            out.append(r)
    return [sp.Float(r, 15) for r in out]


def solve_equation(eq: sp.Eq, var: sp.Symbol, domain: str = "real") -> dict:
    expr = sp.simplify(eq.lhs - eq.rhs)
    dom = sp.S.Reals if domain == "real" else sp.S.Complexes
    try:
        sol = sp.solveset(expr, var, domain=dom)
    except Exception:  # noqa: BLE001
        sol = None
    if sol is None or isinstance(sol, sp.ConditionSet):
        roots = numeric_roots(expr, var) if domain == "real" else []
        return {"set": None, "solutions": roots, "exact": False,
                "note": "No se encontró una solución simbólica cerrada; se muestran raíces reales "
                        "aproximadas en [-10, 10] (resultado numérico)."}
    if sol.is_FiniteSet:
        sols = list(sol)
        try:
            sols.sort(key=lambda s: (float(sp.re(sp.N(s))), float(sp.im(sp.N(s)))))
        except Exception:  # noqa: BLE001
            pass
        return {"set": sol, "solutions": sols, "exact": True, "note": ""}
    return {"set": sol, "solutions": [], "exact": True,
            "note": "El conjunto solución es infinito; se muestra en notación de conjuntos."}


def solve_system(eqs: list, variables: list) -> dict:
    exprs = [sp.simplify(e.lhs - e.rhs) for e in eqs]
    try:
        sols = sp.solve(exprs, variables, dict=True)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"No se pudo resolver el sistema ({type(exc).__name__}).") from exc
    return {"solutions": sols, "linear": all(sp.Poly(e, *variables).total_degree() <= 1
                                              for e in exprs if e.is_polynomial(*variables))}


def polynomial_info(expr, var) -> dict:
    if not expr.is_polynomial(var):
        raise ExpressionError("La expresión no es un polinomio en la variable indicada.")
    p = sp.Poly(expr, var)
    roots = sp.roots(p, multiple=False)
    return {
        "grado": p.degree(),
        "coeficientes": p.all_coeffs(),
        "factorizado": sp.factor(expr),
        "factores": sp.factor_list(expr),
        "raices": roots,
        "raices_numericas": [sp.N(r, 12) for r in p.nroots()],
    }
