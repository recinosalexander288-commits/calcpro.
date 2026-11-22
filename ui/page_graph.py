"""Página: graficador matemático interactivo."""
from __future__ import annotations

import sympy as sp
import streamlit as st

from core.parser import ExpressionError, parse_equation, parse_expression, split_top_level
from plotting import graphs
from . import widgets as W

TYPES = ["Funciones y = f(x)", "Paramétrica 2D", "Polar r(θ)", "Curva implícita F(x,y)=0", "Superficie z = f(x, y)",
         "Curvas de nivel", "Superficie paramétrica", "Superficie cilíndrica r = f(θ, z)", "Superficie esférica ρ = f(θ, φ)",
         "Curva paramétrica 3D", "Campo vectorial 2D", "Campo vectorial 3D", "Región de integración (entre curvas)",
         "Región de integración (polar)"]


def _pair(text: str, default=(-5.0, 5.0)):
    parts = split_top_level(text)
    if len(parts) != 2:
        raise ExpressionError("Escriba el intervalo como «mín, máx».")
    lo, hi = (float(sp.N(parse_expression(p))) for p in parts)
    if lo >= hi:
        raise ExpressionError("En un intervalo el mínimo debe ser menor que el máximo.")
    return lo, hi


def render(settings) -> None:
    st.title("Graficador")
    st.caption("Zoom con la rueda, desplazamiento arrastrando, rotación 3D arrastrando. Doble clic restablece la vista.")
    dark = settings.theme == "Oscuro"
    kind = st.selectbox("Tipo de gráfica", TYPES)
    x, y, z, t, u, v, th, ph = sp.symbols("x y z t u v theta phi")

    def P(s):
        return parse_expression(s, "rad", extra_symbols=("theta", "phi"))

    fig = None
    try:
        if kind == TYPES[0]:
            txt = st.text_area("Una función por línea (compare varias a la vez)", "sin(x)\nx^2/5\n1/x", height=100)
            xr = st.text_input("Intervalo de x", "-6, 6")
            c1, c2, c3 = st.columns(3)
            dd, ii = c1.checkbox("Mostrar f′(x) de la 1.ª función"), c2.checkbox("Mostrar ∫f dx (1.ª función)")
            n_f = len([l for l in txt.splitlines() if l.strip()])
            colors = [st.color_picker(f"Color {i + 1}", graphs.PALETTE[i % 8], key=f"gc{i}") for i in range(min(n_f, 6))]
            pts = st.text_input("Puntos a marcar «x,y; x,y» (opcional)", "")
            lo, hi = _pair(xr)
            items = [(P(l), colors[i] if i < len(colors) else None, l.strip()) for i, l in enumerate(txt.splitlines()) if l.strip()]
            points = []
            for chunk in [c for c in pts.split(";") if c.strip()]:
                a, b = split_top_level(chunk)
                points.append((float(sp.N(P(a))), float(sp.N(P(b))), f"({a.strip()}, {b.strip()})"))
            fig = graphs.plot_functions_2d(items, x, lo, hi, points, dd, ii, dark)
        elif kind == TYPES[1]:
            xt, yt = st.text_input("x(t)", "cos(t) + cos(5t)/2"), st.text_input("y(t)", "sin(t) + sin(5t)/2")
            lo, hi = _pair(st.text_input("t ∈", "0, 2*pi"))
            fig = graphs.plot_parametric_2d(P(xt), P(yt), t, lo, hi, dark=dark)
        elif kind == TYPES[2]:
            r = st.text_input("r(θ)", "1 + cos(theta)")
            lo, hi = _pair(st.text_input("θ ∈", "0, 2*pi"))
            fig = graphs.plot_polar(P(r), th, lo, hi, dark=dark)
        elif kind == TYPES[3]:
            eq = parse_equation(st.text_input("F(x, y) = 0", "x^2 + y^2 - x*y = 4"))
            fig = graphs.plot_implicit(eq.lhs - eq.rhs, x, y, _pair(st.text_input("x ∈", "-4, 4")), _pair(st.text_input("y ∈", "-4, 4")), dark=dark)
        elif kind in (TYPES[4], TYPES[5]):
            f = P(st.text_input("f(x, y)", "sin(x)*cos(y)"))
            xr, yr = _pair(st.text_input("x ∈", "-3, 3")), _pair(st.text_input("y ∈", "-3, 3"))
            fig = graphs.plot_surface(f, x, y, xr, yr, dark) if kind == TYPES[4] else graphs.plot_level_curves(f, x, y, xr, yr, dark=dark)
        elif kind == TYPES[6]:
            X = P(st.text_input("x(u, v)", "(2 + cos(v))*cos(u)"))
            Y = P(st.text_input("y(u, v)", "(2 + cos(v))*sin(u)"))
            Z = P(st.text_input("z(u, v)", "sin(v)"))
            fig = graphs.plot_param_surface(X, Y, Z, u, v, _pair(st.text_input("u ∈", "0, 2*pi")), _pair(st.text_input("v ∈", "0, 2*pi")), dark)
        elif kind == TYPES[7]:
            r = P(st.text_input("r = f(θ, z)", "1 + 0.3*sin(3*theta)"))
            fig = graphs.plot_cylindrical_surface(r, th, z, _pair(st.text_input("θ ∈", "0, 2*pi")), _pair(st.text_input("z ∈", "0, 3")), dark)
        elif kind == TYPES[8]:
            r = P(st.text_input("ρ = f(θ, φ)", "2 + sin(4*phi)*cos(3*theta)/2"))
            fig = graphs.plot_spherical_surface(r, th, ph, _pair(st.text_input("θ ∈", "0, 2*pi")), _pair(st.text_input("φ ∈", "0, pi")), dark)
        elif kind == TYPES[9]:
            c = [P(st.text_input(n, d)) for n, d in (("x(t)", "cos(t)"), ("y(t)", "sin(t)"), ("z(t)", "t/3"))]
            fig = graphs.plot_curve_3d(*c, t, *_pair(st.text_input("t ∈", "0, 6*pi")), dark=dark)
        elif kind == TYPES[10]:
            Pf, Qf = P(st.text_input("P(x, y)", "-y")), P(st.text_input("Q(x, y)", "x"))
            fig = graphs.plot_vector_field_2d(Pf, Qf, x, y, _pair(st.text_input("x ∈", "-3, 3")), _pair(st.text_input("y ∈", "-3, 3")), dark=dark)
        elif kind == TYPES[11]:
            comps = [P(st.text_input(n, d)) for n, d in (("P(x,y,z)", "-y"), ("Q(x,y,z)", "x"), ("R(x,y,z)", "z"))]
            r_ = _pair(st.text_input("Intervalo (igual en x, y, z)", "-2, 2"))
            fig = graphs.plot_vector_field_3d(*comps, x, y, z, r_, r_, r_, dark=dark)
        elif kind == TYPES[12]:
            lo_f, hi_f = P(st.text_input("y inferior g₁(x)", "x^2")), P(st.text_input("y superior g₂(x)", "x"))
            a, b = _pair(st.text_input("x ∈", "0, 1"))
            fig = graphs.plot_region_between(lo_f, hi_f, x, a, b, dark)
        else:
            r1, r2 = P(st.text_input("r inferior", "0")), P(st.text_input("r superior", "1 + cos(theta)"))
            fig = graphs.plot_polar_region(r1, r2, th, *_pair(st.text_input("θ ∈", "0, 2*pi")), dark=dark)
    except ExpressionError as e:
        st.error(f"⚠️ {e}")
    except (ValueError, TypeError) as e:
        st.error(f"⚠️ Revise los datos de entrada ({type(e).__name__}).")
    except Exception as e:  # noqa: BLE001
        st.error(f"⚠️ No se pudo graficar ({type(e).__name__}). Revise la expresión y los intervalos.")
    if fig is not None:
        W.plot(fig, key="graph_main")
