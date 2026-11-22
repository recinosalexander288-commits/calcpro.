"""Página: sistemas de coordenadas, vectores, rectas y planos."""
from __future__ import annotations

import sympy as sp
import streamlit as st

from core import coordinates as C
from core.parser import parse_expression, parse_vector
from plotting import graphs
from tutor.model import L
from . import widgets as W


def _unit(settings_angle: str) -> str:
    return settings_angle


def render(settings) -> None:
    st.title("Coordenadas y vectores")
    tab1, tab2, tab3 = st.tabs(["Conversión de coordenadas", "Vectores", "Rectas y planos"])
    dark = settings.theme == "Oscuro"
    unit = settings.angle

    with tab1:
        st.caption("Convención: θ = ángulo azimutal [0, 2π), φ = ángulo polar medido desde +z [0, π]. "
                   f"Ángulos en {'grados' if unit == 'deg' else 'radianes'} (cámbielo en la barra lateral).")
        src = st.selectbox("Sistema de origen", ["cartesianas", "cilíndricas", "esféricas", "polares"], key="co_src")
        dst = st.selectbox("Convertir a", ["cartesianas", "cilíndricas", "esféricas", "polares"], index=1, key="co_dst")
        labels = {"cartesianas": "x, y, z", "cilíndricas": "r, θ, z", "esféricas": "ρ, θ, φ", "polares": "r, θ"}
        txt = st.text_input(f"Coordenadas ({labels[src]})", "1, 1, 1" if src != "polares" else "2, pi/4", key="co_in")
        if W.btn("Convertir", key="co_go", type="primary"):
            def go():
                comps = list(parse_vector(txt, 2 if src == "polares" else 3, "deg" if unit == "deg" else "rad"))
                if src == "polares":
                    cart = C.polar_to_cart(*comps, unit=unit)
                else:
                    cart = C.to_cartesian(src, *comps, unit=unit)
                if len(cart) == 2 and dst != "polares":
                    cart = (cart[0], cart[1], sp.Integer(0))  # punto en el plano z = 0
                out = C.from_cartesian(dst, *cart, unit=unit) if dst != "polares" else C.cart_to_polar(cart[0], cart[1], unit)
                st.markdown(f"**Cartesianas:** ${L(sp.Matrix(cart).T)}$")
                names = {"cartesianas": ["x", "y", "z"], "cilíndricas": ["r", r"\theta", "z"],
                         "esféricas": [r"\rho", r"\theta", r"\varphi"], "polares": ["r", r"\theta"]}[dst]
                res = ",\\;".join(f"{n}={L(v)}" for n, v in zip(names, out))
                approx = ",  ".join(f"{sp.N(v, settings.precision)}" for v in out)
                W.show_result(res, str(tuple(out)), approx, True)
                W.add_history("Coordenadas", f"{src}→{dst}: {txt}", res, str(tuple(out)))
                if len(cart) == 3:
                    W.plot(graphs.plot_points_3d([(float(sp.N(cart[0])), float(sp.N(cart[1])), float(sp.N(cart[2])), "P")], dark), key="co_plot")
            W.run_guarded(go)
        with st.expander("Distancia entre dos puntos"):
            p1 = st.text_input("P₁ (cartesianas)", "0, 0, 0", key="co_p1")
            p2 = st.text_input("P₂ (cartesianas)", "1, 2, 2", key="co_p2")
            if W.btn("Calcular distancia", key="co_d"):
                W.run_guarded(lambda: st.latex(rf"d={L(C.distance(parse_vector(p1), parse_vector(p2)))}"))

    with tab2:
        a_txt = st.text_input("Vector a", "(1, 2, 3)", key="v_a")
        b_txt = st.text_input("Vector b", "(4, 5, 6)", key="v_b")
        if W.btn("Calcular operaciones", key="v_go", type="primary"):
            def go():
                a, b = parse_vector(a_txt), parse_vector(b_txt)
                st.latex(rf"\|\mathbf a\|={L(C.magnitude(a))},\quad \|\mathbf b\|={L(C.magnitude(b))}")
                st.latex(rf"\mathbf a\cdot\mathbf b={L(C.dot(a, b))}")
                if len(a) in (2, 3) and len(b) == len(a):
                    st.latex(rf"\mathbf a\times\mathbf b={L(C.cross(a, b))}")
                ang, cosv = C.angle_between(a, b, unit)
                st.latex(rf"\cos\theta={L(cosv)},\quad \theta={L(ang)}\ \text{{({'grados' if unit == 'deg' else 'rad'})}}")
                st.latex(rf"\operatorname{{proy}}_{{\mathbf b}}\mathbf a={L(C.projection(a, b))}")
                W.add_history("Vectores", f"{a_txt}, {b_txt}", rf"\mathbf a\cdot\mathbf b={L(C.dot(a, b))}")
            W.run_guarded(go)
        c_txt = st.text_input("Vector c (producto mixto, opcional)", "(7, 8, 10)", key="v_c")
        if W.btn("Producto mixto a·(b×c)", key="v_mix"):
            W.run_guarded(lambda: st.latex(rf"\mathbf a\cdot(\mathbf b\times\mathbf c)={L(C.triple_scalar(parse_vector(a_txt, 3), parse_vector(b_txt, 3), parse_vector(c_txt, 3)))}"))

    with tab3:
        kind = st.selectbox("Operación", ["Recta por dos puntos", "Plano: normal y punto", "Plano por tres puntos",
                                          "Distancia punto–plano", "Distancia punto–recta", "Intersección recta–plano"], key="rp_k")
        t = {k: st.text_input(k, v, key=f"rp_{k}") for k, v in
             {"P": "1, 2, 3", "Q": "4, 6, 8", "R": "0, 1, 5", "Normal n": "1, 2, 3", "Punto P₀ del plano": "0, 0, 0",
              "Dirección d": "1, 1, 1"}.items()}
        if W.btn("Calcular", key="rp_go", type="primary"):
            def go():
                v = lambda k: parse_vector(t[k], 3)
                if kind == "Recta por dos puntos":
                    r = C.line_through(v("P"), v("Q"))
                    st.latex(rf"\mathbf d={L(r['direction'])}")
                    st.latex(rf"\mathbf r(t)={L(r['parametric'])}")
                elif kind == "Plano: normal y punto":
                    eq, n = C.plane_normal_point(v("Normal n"), v("Punto P₀ del plano"))
                    st.latex(L(eq))
                elif kind == "Plano por tres puntos":
                    eq, n = C.plane_three_points(v("P"), v("Q"), v("R"))
                    st.latex(rf"\mathbf n=(Q-P)\times(R-P)={L(n)}")
                    st.latex(L(eq))
                elif kind == "Distancia punto–plano":
                    st.latex(rf"d={L(C.distance_point_plane(v('P'), v('Normal n'), v('Punto P₀ del plano')))}")
                elif kind == "Distancia punto–recta":
                    st.latex(rf"d={L(C.distance_point_line(v('P'), v('Q'), v('Dirección d')))}")
                else:
                    msg, pt = C.line_plane_intersection(v("P"), v("Dirección d"), v("Normal n"), v("Punto P₀ del plano"))
                    st.write(msg.capitalize())
                    if pt is not None:
                        st.latex(rf"\text{{punto}}={L(pt.T)}")
            W.run_guarded(go)
