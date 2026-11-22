"""Pruebas con operaciones matemáticas reales.  Ejecutar:  python -m pytest -q"""
import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import algebra, calculus, coordinates as C, linalg  # noqa: E402
from core.calculator import evaluate  # noqa: E402
from core.parser import ExpressionError, parse_equation, parse_expression, parse_matrix, parse_vector  # noqa: E402
from tutor import derivatives, equations, integrals, limits, multiple  # noqa: E402

x, y, z = sp.symbols("x y z")


# ── parser y seguridad ──
@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "x.__class__", "open('f')", "1/0", "2+", "(1+2", "a;b", "x[0]"])
def test_rechaza_entradas_peligrosas_o_invalidas(bad):
    with pytest.raises(ExpressionError):
        parse_expression(bad)


def test_sintaxis_y_multiplicacion_implicita():
    assert sp.simplify(parse_expression("2x^2 + 3(x+1)") - (2 * x ** 2 + 3 * x + 3)) == 0
    assert parse_expression("5!") == 120
    assert parse_expression("20%*80") == 16
    assert parse_expression("|-3|") == 3
    assert sp.simplify(parse_expression("sin^2(x)+cos^2(x)") - 1) == 0


# ── calculadora ──
def test_resultados_exactos():
    assert evaluate("sqrt(8)").exact == 2 * sp.sqrt(2)
    assert evaluate("sin(pi/6)").exact == sp.Rational(1, 2)
    assert evaluate("ln(e^3)").exact == 3
    assert evaluate("log(1000)").exact == 3
    assert evaluate("(2+3i)*(1-i)").exact == 5 + sp.I
    assert evaluate("1/3 + 1/6").exact == sp.Rational(1, 2)


def test_grados():
    assert evaluate("sin(30)", angle_mode="deg").exact == sp.Rational(1, 2)
    assert evaluate("asin(1/2)", angle_mode="deg").exact == 30


def test_aproximacion_configurable():
    assert evaluate("pi", precision=5).approx.startswith("3.1416")


# ── derivadas ──
def test_derivada_producto_y_cadena():
    sol = derivatives.derivative_solution(x ** 2 * sp.sin(x), x)
    assert sp.simplify(sol.final - (2 * x * sp.sin(x) + x ** 2 * sp.cos(x))) == 0
    assert sol.verification.startswith("✔")
    sol = derivatives.derivative_solution(sp.sin(x ** 2), x)
    assert sp.simplify(sol.final - 2 * x * sp.cos(x ** 2)) == 0


def test_derivada_orden_superior_y_cociente():
    sol = derivatives.derivative_solution((x ** 2 + 1) / (x + 2), x, 2)
    assert sp.simplify(sol.final - sp.diff((x ** 2 + 1) / (x + 2), x, 2)) == 0


# ── integrales ──
def test_integrales_basicas():
    sol = integrals.integral_solution(x * sp.exp(x), x)
    assert "C" in sol.result and sol.verification.startswith("✔")
    sol = integrals.integral_solution(2 * x * sp.cos(x ** 2), x)
    assert "sustitución" in sol.method
    sol = integrals.integral_solution(x ** 2, x, 0, 3)
    assert sol.final == 9


def test_integral_impropia_gaussiana():
    sol = integrals.integral_solution(sp.exp(-x ** 2), x, -sp.oo, sp.oo)
    assert sp.simplify(sol.final - sp.sqrt(sp.pi)) == 0


def test_integral_sin_antiderivada_elemental_es_aproximada():
    d = calculus.definite(sp.exp(-x ** 2) * sp.sin(x ** 3) / x, x, 1, 2)
    assert d["exact"] is not None or d["numeric"] is not None


# ── límites ──
def test_limites():
    assert limits.limit_solution(sp.sin(3 * x) / x, x, 0).final == 3
    assert limits.limit_solution((x ** 2 - 4) / (x - 2), x, 2).final == 4
    assert limits.limit_solution((1 + 1 / x) ** x, x, sp.oo).final == sp.E
    assert limits.limit_solution((sp.sqrt(x + 4) - 2) / x, x, 0).final == sp.Rational(1, 4)


# ── múltiples ──
def test_volumen_esfera_en_esfericas():
    rho, th, ph = sp.symbols("rho theta phi")
    sol = multiple.multiple_integral_solution(sp.Integer(1), [(rho, 0, 2), (ph, 0, sp.pi), (th, 0, 2 * sp.pi)], "esféricas")
    assert sp.simplify(sol.final - sp.Rational(32, 3) * sp.pi) == 0


def test_area_circulo_en_polares():
    r, th = sp.symbols("r theta")
    sol = multiple.multiple_integral_solution(sp.Integer(1), [(r, 0, 3), (th, 0, 2 * sp.pi)], "polares")
    assert sp.simplify(sol.final - 9 * sp.pi) == 0


def test_integral_doble_cartesiana():
    sol = multiple.multiple_integral_solution(x * y, [(x, 0, y), (y, 0, 1)])
    assert sol.final == sp.Rational(1, 8)


# ── vectorial ──
def test_operadores_vectoriales():
    assert calculus.gradient(x ** 2 * y, [x, y]) == sp.Matrix([2 * x * y, x ** 2])
    assert calculus.divergence(sp.Matrix([x, y, z]), [x, y, z]) == 3
    assert calculus.curl(sp.Matrix([-y, x, 0]), [x, y, z]) == sp.Matrix([0, 0, 2])


def test_puntos_criticos():
    cp = calculus.critical_points(x ** 3 - 3 * x, [x])
    kinds = {p["point"][x]: p["kind"] for p in cp["points"]}
    assert kinds[1] == "mínimo local" and kinds[-1] == "máximo local"


def test_lagrange():
    r = calculus.lagrange(x * y, [x + y - 10], [x, y])
    assert r["solutions"][0]["value"] == 25


# ── coordenadas ──
def test_conversiones():
    assert C.cart_to_cyl(1, 1, 5) == (sp.sqrt(2), sp.pi / 4, 5)
    rho, th, ph = C.cart_to_sph(0, 0, 2)
    assert rho == 2 and ph == 0
    xx, yy, zz = C.sph_to_cart(2, sp.pi / 2, sp.pi / 2)
    assert (xx, yy, zz) == (0, 2, 0)
    assert C.cart_to_cyl(0, 2, 0, unit="deg")[1] == 90


def test_vectores_y_planos():
    assert C.cross([1, 0, 0], [0, 1, 0]) == sp.Matrix([0, 0, 1])
    assert C.dot([1, 2, 3], [4, 5, 6]) == 32
    eq, n = C.plane_three_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert sp.simplify(eq.lhs) in (z, -z)


# ── álgebra lineal / ecuaciones ──
def test_matrices():
    A = parse_matrix("1,2;3,4")
    assert A.det() == -2
    out = dict(linalg.operate("Inversa", A))
    assert "A⁻¹" in out


def test_ecuaciones_y_sistemas():
    sol = equations.equation_solution(parse_equation("x^2 - 5x + 6 = 0"), x)
    assert set(sol.final) == {2, 3}
    s = equations.linear_system_solution([parse_equation("2x+y=5"), parse_equation("x-y=1")], [x, y])
    assert "x=2" in s.result and "y=1" in s.result


def test_vector_parser():
    assert list(parse_vector("(1, 2, sin(pi/2))")) == [1, 2, 1]
