"""Sistemas de coordenadas, vectores, rectas y planos.

Convenciones (matemáticas, como en la mayoría de textos de Cálculo):
  • Cilíndricas (r, θ, z):  x = r cosθ,  y = r senθ,  z = z,  r ≥ 0, 0 ≤ θ < 2π.
  • Esféricas  (ρ, θ, φ):   x = ρ senφ cosθ,  y = ρ senφ senθ,  z = ρ cosφ,
    ρ ≥ 0, 0 ≤ θ < 2π (ángulo azimutal), 0 ≤ φ ≤ π (ángulo polar medido desde +z).
  En muchos textos de física θ y φ están intercambiados: verifique la convención de su curso.
"""
from __future__ import annotations

import sympy as sp

from .parser import ExpressionError

PI = sp.pi


def _ang_in(a, unit: str):
    return a * PI / 180 if unit == "deg" else a


def _ang_out(a, unit: str):
    return sp.simplify(a * 180 / PI) if unit == "deg" else sp.simplify(a)


def _theta(x, y):
    if sp.simplify(x) == 0 and sp.simplify(y) == 0:
        return sp.Integer(0)  # convención: θ = 0 sobre el eje z
    th = sp.simplify(sp.atan2(y, x))
    try:
        if th.is_negative:
            th = sp.simplify(th + 2 * PI)
    except Exception:  # noqa: BLE001
        pass
    return th


# ── polares 2D ──
def cart_to_polar(x, y, unit="rad"):
    return sp.simplify(sp.sqrt(x ** 2 + y ** 2)), _ang_out(_theta(x, y), unit)


def polar_to_cart(r, th, unit="rad"):
    th = _ang_in(th, unit)
    return sp.simplify(r * sp.cos(th)), sp.simplify(r * sp.sin(th))


# ── cilíndricas ──
def cart_to_cyl(x, y, z, unit="rad"):
    return sp.simplify(sp.sqrt(x ** 2 + y ** 2)), _ang_out(_theta(x, y), unit), z


def cyl_to_cart(r, th, z, unit="rad"):
    th = _ang_in(th, unit)
    return sp.simplify(r * sp.cos(th)), sp.simplify(r * sp.sin(th)), z


# ── esféricas ──
def cart_to_sph(x, y, z, unit="rad"):
    rho = sp.simplify(sp.sqrt(x ** 2 + y ** 2 + z ** 2))
    if rho == 0:
        raise ExpressionError("En el origen ρ = 0 y los ángulos θ, φ no están definidos.")
    phi = sp.simplify(sp.acos(sp.simplify(z / rho)))
    return rho, _ang_out(_theta(x, y), unit), _ang_out(phi, unit)


def sph_to_cart(rho, th, phi, unit="rad"):
    th, phi = _ang_in(th, unit), _ang_in(phi, unit)
    return (sp.simplify(rho * sp.sin(phi) * sp.cos(th)),
            sp.simplify(rho * sp.sin(phi) * sp.sin(th)),
            sp.simplify(rho * sp.cos(phi)))


def cyl_to_sph(r, th, z, unit="rad"):
    x, y, zz = cyl_to_cart(r, th, z, unit)
    return cart_to_sph(x, y, zz, unit)


def sph_to_cyl(rho, th, phi, unit="rad"):
    x, y, z = sph_to_cart(rho, th, phi, unit)
    return cart_to_cyl(x, y, z, unit)


def to_cartesian(system: str, a, b, c=None, unit="rad"):
    if system == "polares":
        return polar_to_cart(a, b, unit)
    if system == "cilíndricas":
        return cyl_to_cart(a, b, c, unit)
    if system == "esféricas":
        return sph_to_cart(a, b, c, unit)
    return (a, b) if c is None else (a, b, c)


def from_cartesian(system: str, x, y, z=None, unit="rad"):
    if system == "polares":
        return cart_to_polar(x, y, unit)
    if system == "cilíndricas":
        return cart_to_cyl(x, y, z, unit)
    if system == "esféricas":
        return cart_to_sph(x, y, z, unit)
    return (x, y) if z is None else (x, y, z)


def distance(p, q):
    d = sp.Matrix(p) - sp.Matrix(q)
    return sp.simplify(sp.sqrt(sum(c ** 2 for c in d)))


# ── vectores ──
def magnitude(a):
    return sp.simplify(sp.sqrt(sum(c ** 2 for c in sp.Matrix(a))))


def dot(a, b):
    a, b = sp.Matrix(a), sp.Matrix(b)
    if len(a) != len(b):
        raise ExpressionError("Los vectores deben tener la misma dimensión.")
    return sp.simplify(a.dot(b))


def cross(a, b):
    a, b = sp.Matrix(a), sp.Matrix(b)
    if len(a) == 2:
        a, b = a.col_join(sp.Matrix([0])), b.col_join(sp.Matrix([0]))
    if len(a) != 3 or len(b) != 3:
        raise ExpressionError("El producto vectorial se define para vectores de 3 componentes (o 2, con z = 0).")
    return sp.simplify(a.cross(b))


def angle_between(a, b, unit="rad"):
    na, nb = magnitude(a), magnitude(b)
    if na == 0 or nb == 0:
        raise ExpressionError("El ángulo no está definido con el vector cero.")
    c = sp.simplify(dot(a, b) / (na * nb))
    return _ang_out(sp.acos(c), unit), c


def projection(a, b):
    """Proyección vectorial de a sobre b."""
    nb2 = sp.simplify(sum(c ** 2 for c in sp.Matrix(b)))
    if nb2 == 0:
        raise ExpressionError("No se puede proyectar sobre el vector cero.")
    return sp.simplify(dot(a, b) / nb2 * sp.Matrix(b))


def triple_scalar(a, b, c):
    return sp.simplify(sp.Matrix(a).dot(sp.Matrix(b).cross(sp.Matrix(c))))


# ── rectas y planos ──
def line_through(p, q):
    p, q = sp.Matrix(p), sp.Matrix(q)
    d = sp.simplify(q - p)
    if all(c == 0 for c in d):
        raise ExpressionError("Los dos puntos coinciden: no definen una recta.")
    t = sp.Symbol("t")
    return {"direction": d, "parametric": p + t * d, "t": t}


def plane_normal_point(n, p0):
    x, y, z = sp.symbols("x y z")
    n, p0 = sp.Matrix(n), sp.Matrix(p0)
    if all(c == 0 for c in n):
        raise ExpressionError("El vector normal no puede ser cero.")
    eq = sp.expand(n[0] * (x - p0[0]) + n[1] * (y - p0[1]) + n[2] * (z - p0[2]))
    return sp.Eq(eq, 0), n


def plane_three_points(p, q, r):
    p, q, r = sp.Matrix(p), sp.Matrix(q), sp.Matrix(r)
    n = (q - p).cross(r - p)
    if all(sp.simplify(c) == 0 for c in n):
        raise ExpressionError("Los tres puntos son colineales: no definen un único plano.")
    eq, n = plane_normal_point(n, p)
    return eq, sp.simplify(n)


def distance_point_plane(point, normal, p0):
    n, p, p0 = sp.Matrix(normal), sp.Matrix(point), sp.Matrix(p0)
    return sp.simplify(sp.Abs(n.dot(p - p0)) / magnitude(n))


def distance_point_line(point, a, d):
    ap = sp.Matrix(point) - sp.Matrix(a)
    return sp.simplify(magnitude(cross(ap, d)) / magnitude(d))


def line_plane_intersection(a, d, normal, p0):
    t = sp.Symbol("t")
    a, d, n, p0 = sp.Matrix(a), sp.Matrix(d), sp.Matrix(normal), sp.Matrix(p0)
    denom = sp.simplify(n.dot(d))
    if denom == 0:
        if sp.simplify(n.dot(p0 - a)) == 0:
            return "la recta está contenida en el plano", None
        return "la recta es paralela al plano (no hay intersección)", None
    t0 = sp.simplify(n.dot(p0 - a) / denom)
    return "intersección en un punto", sp.simplify(a + t0 * d)


def angle_between_planes(n1, n2, unit="rad"):
    return angle_between(n1, n2, unit)[0]


# ── transformación simbólica de expresiones ──
def expr_to_cylindrical(f):
    x, y, z = sp.symbols("x y z")
    r, th = sp.symbols("r theta")
    return sp.simplify(sp.trigsimp(f.subs({x: r * sp.cos(th), y: r * sp.sin(th)}, simultaneous=True)))


def expr_to_spherical(f):
    x, y, z = sp.symbols("x y z")
    rho, th, ph = sp.symbols("rho theta phi")
    sub = {x: rho * sp.sin(ph) * sp.cos(th), y: rho * sp.sin(ph) * sp.sin(th), z: rho * sp.cos(ph)}
    return sp.simplify(sp.trigsimp(f.subs(sub, simultaneous=True)))
