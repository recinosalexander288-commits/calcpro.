"""Cálculo diferencial, integral y vectorial (simbólico con respaldo numérico)."""
from __future__ import annotations

import warnings

import numpy as np
import sympy as sp

from .parser import ExpressionError


# ───────────────────────── utilidades ─────────────────────────
def unresolved(e) -> bool:
    return e is None or e.has(sp.Integral) or e.has(sp.zoo) or e.has(sp.nan)


def _to_float(v) -> float:
    v = sp.sympify(v)
    if v == sp.oo:
        return np.inf
    if v == -sp.oo:
        return -np.inf
    return float(sp.N(v))


# ───────────────────────── cálculo diferencial ─────────────────────────
def derivative(f, x, order: int = 1):
    return sp.diff(f, x, order)


def partial_derivative(f, *vars_):
    return sp.diff(f, *vars_)


def implicit_derivative(eq: sp.Eq, y: sp.Symbol, x: sp.Symbol):
    """dy/dx de F(x, y) = 0 mediante -Fx/Fy."""
    F = eq.lhs - eq.rhs
    if not F.has(y):
        raise ExpressionError("La ecuación no contiene la variable dependiente.")
    return sp.simplify(sp.idiff(F, y, x))


def gradient(f, vars_):
    return sp.Matrix([sp.diff(f, v) for v in vars_])


def divergence(F, vars_):
    if len(F) != len(vars_):
        raise ExpressionError("El campo y la lista de variables deben tener la misma dimensión.")
    return sp.simplify(sum(sp.diff(F[i], v) for i, v in enumerate(vars_)))


def curl(F, vars_):
    if len(F) == 3 and len(vars_) == 3:
        x, y, z = vars_
        return sp.simplify(sp.Matrix([
            sp.diff(F[2], y) - sp.diff(F[1], z),
            sp.diff(F[0], z) - sp.diff(F[2], x),
            sp.diff(F[1], x) - sp.diff(F[0], y),
        ]))
    if len(F) == 2 and len(vars_) == 2:
        x, y = vars_
        return sp.simplify(sp.diff(F[1], x) - sp.diff(F[0], y))  # componente k
    raise ExpressionError("El rotacional requiere un campo de 3 componentes (o 2 para el rotacional escalar).")


def jacobian(F, vars_):
    return sp.Matrix(list(F)).jacobian(list(vars_))


def hessian(f, vars_):
    return sp.hessian(f, list(vars_))


def directional_derivative(f, vars_, direction, point: dict | None = None):
    v = sp.Matrix(direction)
    norm = sp.sqrt(sum(c ** 2 for c in v))
    if sp.simplify(norm) == 0:
        raise ExpressionError("La dirección no puede ser el vector cero.")
    u = v / norm
    g = gradient(f, vars_)
    if point:
        g = g.subs(point)
    return sp.simplify(g.dot(u)), u


def critical_points(f, vars_) -> dict:
    grad = [sp.diff(f, v) for v in vars_]
    try:
        sols = sp.solve(grad, list(vars_), dict=True)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"No se pudo resolver ∇f = 0 ({type(exc).__name__}).") from exc
    H = sp.hessian(f, list(vars_))
    points, skipped = [], 0
    for s in sols:
        if len(s) < len(vars_):
            skipped += 1
            continue
        if any(abs(complex(sp.N(val)).imag) > 1e-12 for val in s.values()):
            continue
        Hp = H.subs(s)
        points.append({"point": s, "value": sp.simplify(f.subs(s)), "hessian": Hp,
                       "kind": classify(Hp, len(vars_))})
    return {"gradient": grad, "hessian": H, "points": points, "skipped": skipped}


def classify(Hp: sp.Matrix, n: int) -> str:
    try:
        if n == 1:
            d = sp.simplify(Hp[0, 0])
            return "mínimo local" if d > 0 else "máximo local" if d < 0 else "indeterminado (f'' = 0)"
        if n == 2:
            D = sp.simplify(Hp.det())
            fxx = sp.simplify(Hp[0, 0])
            if D > 0:
                return "mínimo local" if fxx > 0 else "máximo local"
            if D < 0:
                return "punto silla"
            return "criterio no concluyente (D = 0)"
        eig = np.linalg.eigvalsh(np.array(Hp.evalf(), dtype=float))
        if np.all(eig > 1e-9):
            return "mínimo local"
        if np.all(eig < -1e-9):
            return "máximo local"
        if np.any(eig > 1e-9) and np.any(eig < -1e-9):
            return "punto silla"
        return "criterio no concluyente"
    except Exception:  # noqa: BLE001
        return "no se pudo clasificar"


def lagrange(f, constraints, vars_) -> dict:
    lams = list(sp.symbols(f"lambda1:{len(constraints) + 1}"))
    L = f - sum(l * g for l, g in zip(lams, constraints))
    eqs = [sp.diff(L, v) for v in vars_] + list(constraints)  # ∇L = 0 y g_i = 0
    try:
        sols = sp.solve(eqs, list(vars_) + lams, dict=True)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"No se pudo resolver el sistema de Lagrange ({type(exc).__name__}).") from exc
    out = []
    for s in sols:
        if all(v in s for v in vars_):
            out.append({"point": {v: s[v] for v in vars_}, "multipliers": {l: s.get(l) for l in lams},
                        "value": sp.simplify(f.subs({v: s[v] for v in vars_}))})
    return {"lagrangian": L, "equations": eqs, "solutions": out, "lams": lams}


def limit_value(f, x, point, direction: str = "+-"):
    return sp.limit(f, x, point, dir=direction)


# ───────────────────────── integrales ─────────────────────────
def indefinite(f, x):
    res = sp.integrate(f, x)
    return None if unresolved(res) else res


def numeric_integral_1d(f, x, a, b) -> tuple[float, float]:
    from scipy import integrate as si
    fn = sp.lambdify(x, f, modules=["scipy", "numpy"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", si.IntegrationWarning)
            val, err = si.quad(lambda t: float(np.real(fn(t))), _to_float(a), _to_float(b), limit=300)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(
            "La integral numérica no converge o el integrando no está definido en el intervalo "
            f"(posible divergencia). Detalle: {type(exc).__name__}."
        ) from exc
    return val, err


def numeric_integral_nd(f, limits) -> tuple[float, float]:
    """limits: [(var, lo, hi), ...] con la variable MÁS INTERNA primero."""
    from scipy import integrate as si
    vars_ = [v for v, _, _ in limits]
    fn = sp.lambdify(vars_, f, modules=["scipy", "numpy"])
    ranges = []
    for i, (v, lo, hi) in enumerate(limits):
        outer = vars_[i + 1:]
        lo, hi = sp.sympify(lo), sp.sympify(hi)
        if (lo.free_symbols | hi.free_symbols) & set(outer):
            flo = sp.lambdify(outer, lo, modules=["scipy", "numpy"])
            fhi = sp.lambdify(outer, hi, modules=["scipy", "numpy"])
            ranges.append(lambda *a, flo=flo, fhi=fhi: [float(flo(*a)), float(fhi(*a))])
        else:
            ranges.append([_to_float(lo), _to_float(hi)])
    try:
        val, err = si.nquad(lambda *a: float(np.real(fn(*a))), ranges)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"La integración numérica falló ({type(exc).__name__}).") from exc
    return val, err


def definite(f, x, a, b) -> dict:
    """Integral definida/impropia: exacta si es posible, si no numérica."""
    out = {"exact": None, "numeric": None, "divergent": False, "note": ""}
    try:
        res = sp.integrate(f, (x, a, b))
    except Exception:  # noqa: BLE001
        res = None
    if res is not None and res in (sp.oo, -sp.oo):
        out["divergent"] = True
        out["exact"] = res
        out["note"] = "La integral impropia DIVERGE."
        return out
    if res is not None and not unresolved(res) and not res.has(sp.Piecewise):
        out["exact"] = sp.simplify(res)
        return out
    try:
        out["numeric"] = numeric_integral_1d(f, x, a, b)
        out["note"] = ("No existe antiderivada elemental (o SymPy no la encontró): "
                       "el valor es una APROXIMACIÓN NUMÉRICA (cuadratura adaptativa de QUADPACK).")
    except ExpressionError as e:
        out["note"] = str(e)
    return out


def iterated(f, limits) -> dict:
    """Integral iterada; limits con la variable MÁS INTERNA primero."""
    out = {"exact": None, "numeric": None, "note": ""}
    try:
        res = sp.integrate(f, *limits)
    except Exception:  # noqa: BLE001
        res = None
    if res is not None and not unresolved(res):
        out["exact"] = sp.simplify(res)
        return out
    try:
        out["numeric"] = numeric_integral_nd(f, limits)
        out["note"] = "Resultado APROXIMADO por integración numérica (scipy.integrate.nquad)."
    except ExpressionError as e:
        out["note"] = str(e)
    return out


# ───────────────────────── sistemas de coordenadas ─────────────────────────
def coordinate_map(system: str):
    """Devuelve (nuevas_variables, sustitución, jacobiano, descripción)."""
    x, y, z = sp.symbols("x y z")
    r, th, ph, rho = sp.symbols("r theta phi rho")
    if system == "polares":
        return (r, th), {x: r * sp.cos(th), y: r * sp.sin(th)}, r, "dA = r\\,dr\\,d\\theta"
    if system == "cilíndricas":
        return (r, th, z), {x: r * sp.cos(th), y: r * sp.sin(th)}, r, "dV = r\\,dz\\,dr\\,d\\theta"
    if system == "esféricas":
        return ((rho, th, ph),
                {x: rho * sp.sin(ph) * sp.cos(th), y: rho * sp.sin(ph) * sp.sin(th), z: rho * sp.cos(ph)},
                rho ** 2 * sp.sin(ph), "dV = \\rho^{2}\\sin\\varphi\\,d\\rho\\,d\\theta\\,d\\varphi")
    raise ExpressionError(f"Sistema de coordenadas desconocido: {system}")


def to_coordinates(f, system: str):
    new_vars, subs, jac, _ = coordinate_map(system)
    g = sp.simplify(sp.trigsimp(f.subs(subs, simultaneous=True)))
    return g, jac, new_vars


# ───────────────────────── geometría con integrales ─────────────────────────
def arc_length_param(r_vec, t, a, b) -> dict:
    d = sp.Matrix(r_vec).diff(t)
    speed = sp.simplify(sp.sqrt(sum(c ** 2 for c in d)))
    return {"speed": speed, "result": definite(speed, t, a, b)}


def arc_length_graph(f, x, a, b) -> dict:
    speed = sp.simplify(sp.sqrt(1 + sp.diff(f, x) ** 2))
    return {"speed": speed, "result": definite(speed, x, a, b)}


def line_integral_scalar(f, r_vec, t, a, b) -> dict:
    sub = dict(zip(sp.symbols("x y z")[:len(r_vec)], r_vec))
    speed = sp.simplify(sp.sqrt(sum(c ** 2 for c in sp.Matrix(r_vec).diff(t))))
    integrand = sp.simplify(f.subs(sub, simultaneous=True) * speed)
    return {"integrand": integrand, "result": definite(integrand, t, a, b)}


def line_integral_vector(F, r_vec, t, a, b) -> dict:
    sub = dict(zip(sp.symbols("x y z")[:len(r_vec)], r_vec))
    dr = sp.Matrix(r_vec).diff(t)
    integrand = sp.simplify(sp.Matrix(F).subs(sub, simultaneous=True).dot(dr))
    return {"integrand": integrand, "result": definite(integrand, t, a, b)}


def surface_integral_scalar(f, r_vec, u, v, u_lim, v_lim) -> dict:
    sub = dict(zip(sp.symbols("x y z"), r_vec))
    ru, rv = sp.Matrix(r_vec).diff(u), sp.Matrix(r_vec).diff(v)
    n = ru.cross(rv)
    dS = sp.simplify(sp.sqrt(sum(c ** 2 for c in n)))
    integrand = sp.simplify(f.subs(sub, simultaneous=True) * dS)
    return {"normal": sp.simplify(n), "dS": dS, "integrand": integrand,
            "result": iterated(integrand, [(u, *u_lim), (v, *v_lim)])}


def flux_integral(F, r_vec, u, v, u_lim, v_lim) -> dict:
    sub = dict(zip(sp.symbols("x y z"), r_vec))
    n = sp.Matrix(r_vec).diff(u).cross(sp.Matrix(r_vec).diff(v))
    integrand = sp.simplify(sp.Matrix(F).subs(sub, simultaneous=True).dot(n))
    return {"normal": sp.simplify(n), "integrand": integrand,
            "result": iterated(integrand, [(u, *u_lim), (v, *v_lim)])}


def surface_area_graph(f, x, y, x_lim, y_lim) -> dict:
    dS = sp.simplify(sp.sqrt(1 + sp.diff(f, x) ** 2 + sp.diff(f, y) ** 2))
    return {"dS": dS, "result": iterated(dS, [(x, *x_lim), (y, *y_lim)])}


def volume_under_graph(f, x, y, x_lim, y_lim) -> dict:
    return {"result": iterated(f, [(x, *x_lim), (y, *y_lim)])}
