"""Series, ecuaciones diferenciales, métodos numéricos, interpolación y estadística."""
from __future__ import annotations

import re

import numpy as np
import sympy as sp

from .parser import ExpressionError, parse_equation, parse_expression


# ───────────────────────── series y sucesiones ─────────────────────────
def taylor(f, x, x0, n: int):
    return sp.series(f, x, x0, n + 1).removeO()


def summation(term, k, a, b):
    res = sp.summation(term, (k, a, b))
    if res.has(sp.Sum):
        raise ExpressionError("SymPy no encontró una forma cerrada para esta sumatoria.")
    return sp.simplify(res)


def sequence_limit(term, n):
    return sp.limit(term, n, sp.oo)


def convergence(term, k, a=1) -> dict:
    s = sp.Sum(term, (k, a, sp.oo))
    out = {"convergente": None, "ratio": None, "root": None}
    try:
        out["ratio"] = sp.limit(sp.Abs(term.subs(k, k + 1) / term), k, sp.oo)
    except Exception:  # noqa: BLE001
        pass
    try:
        out["convergente"] = s.is_convergent()
    except Exception:  # noqa: BLE001
        out["convergente"] = None
    return out


# ───────────────────────── ecuaciones diferenciales ─────────────────────────
def parse_ode(text: str):
    """Acepta  y' + 2y = x,  y'' - y = 0,  etc. Variable independiente: x."""
    s = text.replace("’", "'").replace("′", "'")
    s = re.sub(r"y'''", "D3y", s)
    s = re.sub(r"y''", "D2y", s)
    s = re.sub(r"y'", "D1y", s)
    eq = parse_equation(s, extra_symbols=("D1y", "D2y", "D3y"))
    x, y = sp.symbols("x y")
    yf = sp.Function("y")
    D1, D2, D3 = sp.symbols("D1y D2y D3y")
    return eq.subs({D3: yf(x).diff(x, 3), D2: yf(x).diff(x, 2), D1: yf(x).diff(x), y: yf(x)},
                   simultaneous=True), yf, x


def solve_ode(text: str, x0=None, y0=None, dy0=None):
    eq, yf, x = parse_ode(text)
    ics = {}
    if x0 is not None and y0 is not None:
        ics[yf(x0)] = y0
    if x0 is not None and dy0 is not None:
        ics[yf(x).diff(x).subs(x, x0)] = dy0
    try:
        sol = sp.dsolve(eq, yf(x), ics=ics or None)
    except Exception as exc:  # noqa: BLE001
        raise ExpressionError(f"SymPy no pudo resolver esta EDO de forma simbólica ({type(exc).__name__}). "
                              "Pruebe el método numérico (Runge-Kutta).") from exc
    return eq, sol


def ode_numeric(f_text: str, x0: float, y0: float, x_end: float, points: int = 200):
    """Resuelve y' = f(x, y) con Runge-Kutta 4(5) (scipy.integrate.solve_ivp)."""
    from scipy.integrate import solve_ivp
    f = parse_expression(f_text)
    x, y = sp.symbols("x y")
    if not f.free_symbols <= {x, y}:
        raise ExpressionError("f(x, y) sólo puede contener las variables x e y.")
    fn = sp.lambdify((x, y), f, "numpy")
    xs = np.linspace(x0, x_end, points)
    sol = solve_ivp(lambda t, u: [float(fn(t, u[0]))], (x0, x_end), [y0], t_eval=xs, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise ExpressionError(f"El integrador numérico falló: {sol.message}")
    return sol.t, sol.y[0]


# ───────────────────────── raíces ─────────────────────────
def newton(f, x, x0: float, tol: float = 1e-12, max_iter: int = 50) -> list[dict]:
    fp = sp.diff(f, x)
    fn, dfn = sp.lambdify(x, f, "numpy"), sp.lambdify(x, fp, "numpy")
    rows, xi = [], float(x0)
    for i in range(1, max_iter + 1):
        d = float(dfn(xi))
        if d == 0:
            raise ExpressionError(f"La derivada se anula en x = {xi}: el método de Newton no puede continuar.")
        xn = xi - float(fn(xi)) / d
        rows.append({"iteración": i, "x_n": xi, "f(x_n)": float(fn(xi)), "x_{n+1}": xn, "error": abs(xn - xi)})
        if abs(xn - xi) < tol:
            break
        xi = xn
    return rows


def bisection(f, x, a: float, b: float, tol: float = 1e-10, max_iter: int = 200) -> list[dict]:
    fn = sp.lambdify(x, f, "numpy")
    fa, fb = float(fn(a)), float(fn(b))
    if fa * fb > 0:
        raise ExpressionError("f(a) y f(b) deben tener signos opuestos para garantizar una raíz en [a, b].")
    rows = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = float(fn(c))
        rows.append({"iteración": i, "a": a, "b": b, "c": c, "f(c)": fc, "error": (b - a) / 2})
        if abs(fc) == 0 or (b - a) / 2 < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return rows


def secant(f, x, x0: float, x1: float, tol: float = 1e-12, max_iter: int = 60) -> list[dict]:
    fn = sp.lambdify(x, f, "numpy")
    rows = []
    for i in range(1, max_iter + 1):
        f0, f1 = float(fn(x0)), float(fn(x1))
        if f1 == f0:
            raise ExpressionError("Denominador nulo en el método de la secante.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        rows.append({"iteración": i, "x_{n-1}": x0, "x_n": x1, "x_{n+1}": x2, "error": abs(x2 - x1)})
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return rows


# ───────────────────────── interpolación ─────────────────────────
def lagrange_polynomial(xs, ys, x):
    pts = list(zip([sp.nsimplify(v) for v in xs], [sp.nsimplify(v) for v in ys]))
    if len({p[0] for p in pts}) != len(pts):
        raise ExpressionError("Los valores de x deben ser distintos.")
    return sp.expand(sp.interpolate(pts, x))


def newton_divided_differences(xs, ys):
    n = len(xs)
    table = [list(map(float, ys))]
    for k in range(1, n):
        prev = table[-1]
        table.append([(prev[i + 1] - prev[i]) / (xs[i + k] - xs[i]) for i in range(n - k)])
    return table


def cubic_spline(xs, ys):
    from scipy.interpolate import CubicSpline
    order = np.argsort(xs)
    return CubicSpline(np.array(xs, float)[order], np.array(ys, float)[order])


# ───────────────────────── estadística y probabilidad ─────────────────────────
def parse_numbers(text: str) -> list[float]:
    parts = [p for p in re.split(r"[,\s;]+", (text or "").strip()) if p]
    if not parts:
        raise ExpressionError("Ingrese al menos un dato numérico.")
    out = []
    for p in parts:
        v = parse_expression(p)
        if v.free_symbols:
            raise ExpressionError(f"«{p}» no es un número.")
        out.append(float(sp.N(v)))
    return out


def describe(data) -> dict:
    a = np.asarray(data, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {
        "n": int(a.size), "media": a.mean(), "mediana": med, "mínimo": a.min(), "máximo": a.max(),
        "rango": a.max() - a.min(), "Q1": q1, "Q3": q3, "IQR": q3 - q1,
        "varianza muestral": a.var(ddof=1) if a.size > 1 else float("nan"),
        "desv. estándar muestral": a.std(ddof=1) if a.size > 1 else float("nan"),
        "varianza poblacional": a.var(), "desv. estándar poblacional": a.std(),
    }


def linear_regression(xs, ys) -> dict:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size != y.size or x.size < 2:
        raise ExpressionError("x e y deben tener la misma cantidad (≥ 2) de datos.")
    m, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1] if x.size > 2 else float("nan")
    return {"pendiente": m, "intercepto": b, "r": r, "R²": r ** 2}


def distribution(kind: str, params: dict, x=None, a=None, b=None) -> dict:
    from scipy import stats
    if kind == "Binomial":
        d = stats.binom(int(params["n"]), params["p"])
    elif kind == "Poisson":
        d = stats.poisson(params["lambda"])
    elif kind == "Normal":
        if params["sigma"] <= 0:
            raise ExpressionError("σ debe ser positiva.")
        d = stats.norm(params["mu"], params["sigma"])
    elif kind == "Exponencial":
        d = stats.expon(scale=1 / params["lambda"])
    elif kind == "Uniforme":
        d = stats.uniform(params["a"], params["b"] - params["a"])
    else:
        raise ExpressionError("Distribución no soportada.")
    out = {"media": float(d.mean()), "varianza": float(d.var())}
    discrete = kind in ("Binomial", "Poisson")
    if x is not None:
        out["P(X = x)" if discrete else "f(x)"] = float(d.pmf(x) if discrete else d.pdf(x))
        out["P(X ≤ x)"] = float(d.cdf(x))
    if a is not None and b is not None:
        out["P(a ≤ X ≤ b)"] = float(d.cdf(b) - (d.cdf(a - 1) if discrete else d.cdf(a)))
    return out
