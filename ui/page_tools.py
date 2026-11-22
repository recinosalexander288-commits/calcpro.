"""Página: series, EDO, métodos numéricos, interpolación, estadística y probabilidad."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

from core import numerics as N
from core.parser import ExpressionError, parse_expression, parse_symbol, split_top_level
from plotting import graphs
from tutor.model import L
from . import widgets as W


def _f(s: str, **k):
    return parse_expression(s, "rad", **k)


def render(settings) -> None:
    st.title("Herramientas de ingeniería")
    dark = settings.theme == "Oscuro"
    tabs = st.tabs(["Series y sucesiones", "Ecuaciones diferenciales", "Raíces", "Interpolación", "Estadística", "Probabilidad"])

    with tabs[0]:
        op = st.selectbox("Operación", ["Serie de Taylor", "Sumatoria", "Límite de una sucesión", "Convergencia de una serie"], key="t_op")
        term = st.text_input("Expresión", "exp(x)" if op == "Serie de Taylor" else "1/k^2", key="t_term")
        c1, c2, c3 = st.columns(3)
        if op == "Serie de Taylor":
            x0, n = c1.text_input("Centro x₀", "0", key="t_x0"), c2.number_input("Grado n", 1, 20, 5, key="t_n")
        elif op == "Sumatoria":
            a, b = c1.text_input("Desde", "1", key="t_a"), c2.text_input("Hasta (oo = ∞)", "oo", key="t_b")
        elif op == "Convergencia de una serie":
            a = c1.text_input("Desde", "1", key="t_a2")
        if W.btn("Calcular", key="t_go", type="primary"):
            def go_():
                e = _f(term, extra_symbols=("k", "n"))
                if op == "Serie de Taylor":
                    x = parse_symbol("x")
                    s = N.taylor(e, x, _f(x0), int(n))
                    W.show_result(rf"{L(e)}\approx {L(s)}", str(s), None, True)
                    xs = np.linspace(-3, 3, 400)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=xs, y=graphs._fn(e, x)(xs), name="f(x)"))
                    fig.add_trace(go.Scatter(x=xs, y=graphs._fn(s, x)(xs), name=f"Taylor grado {int(n)}", line=dict(dash="dash")))
                    fig.update_yaxes(range=[-10, 10])
                    W.plot(graphs._layout(fig, "Aproximación de Taylor", dark), key="t_taylor")
                elif op == "Sumatoria":
                    k = parse_symbol("k")
                    r = N.summation(e, k, _f(a), _f(b))
                    W.show_result(rf"\sum_{{k={a}}}^{{{b}}}{L(e)}={L(r)}", str(r), None, True)
                elif op == "Límite de una sucesión":
                    r = N.sequence_limit(e, parse_symbol("n"))
                    W.show_result(rf"\lim_{{n\to\infty}}{L(e)}={L(r)}", str(r), None, True)
                else:
                    info = N.convergence(e, parse_symbol("k"), _f(a))
                    st.latex(rf"\lim\left|\frac{{a_{{k+1}}}}{{a_k}}\right|={L(info['ratio'])}" if info["ratio"] is not None else r"\text{criterio del cociente no concluyente}")
                    conv = info["convergente"]
                    (st.success if conv else st.warning)("La serie CONVERGE." if conv else ("La serie DIVERGE o SymPy no pudo decidirlo." if conv is False else "SymPy no pudo decidir la convergencia."))
            W.run_guarded(go_)

    with tabs[1]:
        mode = st.radio("Método", ["Simbólico (dsolve)", "Numérico (Runge-Kutta)"], horizontal=True, key="t_ode")
        if mode.startswith("Simbólico"):
            ode = st.text_input("EDO en x (use y', y'', y''')", "y'' - 3y' + 2y = exp(x)", key="t_ode_txt")
            c1, c2, c3 = st.columns(3)
            use_ic = st.checkbox("Condiciones iniciales", key="t_ic")
            ic = (c1.text_input("x₀", "0", key="t_x0i"), c2.text_input("y(x₀)", "1", key="t_y0"), c3.text_input("y′(x₀)", "0", key="t_dy0")) if use_ic else None
            if W.btn("Resolver EDO", key="t_ode_go", type="primary"):
                def go_():
                    args = (_f(ic[0]), _f(ic[1]), _f(ic[2]) if "''" in ode else None) if ic else ()
                    eq, sol = N.solve_ode(ode, *args)
                    st.latex(L(eq))
                    W.show_result(L(sol), str(sol), None, True)
                    W.add_history("EDO", ode, L(sol), str(sol))
                W.run_guarded(go_)
        else:
            f_txt = st.text_input("y′ = f(x, y)", "x - y", key="t_ode_f")
            c1, c2, c3 = st.columns(3)
            x0, y0, xe = c1.number_input("x₀", value=0.0, key="t_nx0"), c2.number_input("y(x₀)", value=1.0, key="t_ny0"), c3.number_input("x final", value=5.0, key="t_nxe")
            if W.btn("Integrar numéricamente", key="t_ode_ngo", type="primary"):
                def go_():
                    xs, ys = N.ode_numeric(f_txt, x0, y0, xe)
                    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="y(x)"))
                    W.plot(graphs._layout(fig, "Solución numérica (aproximada)", dark), key="t_ode_plot")
                    st.warning("Resultado NUMÉRICO (RK45, rtol=1e-9), no una fórmula exacta.")
                    st.dataframe(pd.DataFrame({"x": xs[::20], "y": ys[::20]}))
                W.run_guarded(go_)

    with tabs[2]:
        meth = st.selectbox("Método", ["Newton–Raphson", "Bisección", "Secante"], key="t_rm")
        f_txt = st.text_input("f(x) = 0", "x^3 - 2x - 5", key="t_rf")
        c1, c2 = st.columns(2)
        p1 = c1.number_input("x₀ / a", value=2.0, key="t_r1")
        p2 = c2.number_input("x₁ / b (bisección y secante)", value=3.0, key="t_r2")
        if W.btn("Aproximar raíz", key="t_rgo", type="primary"):
            def go_():
                x, f = parse_symbol("x"), _f(f_txt)
                rows = {"Newton–Raphson": lambda: N.newton(f, x, p1), "Bisección": lambda: N.bisection(f, x, p1, p2),
                        "Secante": lambda: N.secant(f, x, p1, p2)}[meth]()
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                last = rows[-1]
                root = last.get("x_{n+1}", last.get("c"))
                st.success(f"Raíz aproximada: x ≈ {root:.12g}  (resultado numérico)")
            W.run_guarded(go_)

    with tabs[3]:
        data = st.text_area("Puntos (x, y) — uno por línea", "0, 1\n1, 3\n2, 2\n3, 5", key="t_ip")
        m = st.selectbox("Método", ["Polinomio de Lagrange", "Diferencias divididas de Newton", "Spline cúbico"], key="t_im")
        if W.btn("Interpolar", key="t_igo", type="primary"):
            def go_():
                pts = []
                for line in data.splitlines():
                    if line.strip():
                        a, b = split_top_level(line)
                        pts.append((float(sp.N(_f(a))), float(sp.N(_f(b)))))
                if len(pts) < 2:
                    raise ExpressionError("Se necesitan al menos 2 puntos.")
                xs, ys = zip(*pts)
                x = parse_symbol("x")
                grid = np.linspace(min(xs), max(xs), 300)
                fig = go.Figure()
                if m == "Polinomio de Lagrange":
                    p = N.lagrange_polynomial(xs, ys, x)
                    st.latex(rf"P(x)={L(p)}")
                    fig.add_trace(go.Scatter(x=grid, y=graphs._fn(p, x)(grid), name="P(x)"))
                elif m.startswith("Diferencias"):
                    table = N.newton_divided_differences(list(xs), list(ys))
                    st.dataframe(pd.DataFrame({f"orden {k}": pd.Series(col) for k, col in enumerate(table)}))
                    p = N.lagrange_polynomial(xs, ys, x)
                    st.latex(rf"P(x)={L(p)}")
                    fig.add_trace(go.Scatter(x=grid, y=graphs._fn(p, x)(grid), name="P(x)"))
                else:
                    cs = N.cubic_spline(xs, ys)
                    fig.add_trace(go.Scatter(x=grid, y=cs(grid), name="spline cúbico"))
                fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="markers", name="datos", marker=dict(size=10)))
                W.plot(graphs._layout(fig, "Interpolación", dark), key="t_ip_plot")
            W.run_guarded(go_)

    with tabs[4]:
        kind = st.radio("Análisis", ["Estadística descriptiva", "Regresión lineal"], horizontal=True, key="t_sk")
        if kind == "Estadística descriptiva":
            d = st.text_area("Datos (separados por coma, espacio o salto de línea)", "4, 8, 15, 16, 23, 42", key="t_sd")
            if W.btn("Calcular", key="t_sgo", type="primary"):
                def go_():
                    res = N.describe(N.parse_numbers(d))
                    st.dataframe(pd.DataFrame({"valor": res}))
                W.run_guarded(go_)
        else:
            xs_t = st.text_input("x", "1, 2, 3, 4, 5", key="t_rx")
            ys_t = st.text_input("y", "2.1, 3.9, 6.2, 7.8, 10.1", key="t_ry")
            if W.btn("Ajustar recta", key="t_rego", type="primary"):
                def go_():
                    xs, ys = N.parse_numbers(xs_t), N.parse_numbers(ys_t)
                    r = N.linear_regression(xs, ys)
                    st.latex(rf"y={r['pendiente']:.6g}\,x+{r['intercepto']:.6g},\quad r={r['r']:.6g},\quad R^2={r['R²']:.6g}")
                    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", name="datos"))
                    gx = np.linspace(min(xs), max(xs), 50)
                    fig.add_trace(go.Scatter(x=gx, y=r["pendiente"] * gx + r["intercepto"], name="ajuste"))
                    W.plot(graphs._layout(fig, "Regresión lineal", dark), key="t_reg")
                W.run_guarded(go_)

    with tabs[5]:
        dist = st.selectbox("Distribución", ["Binomial", "Poisson", "Normal", "Exponencial", "Uniforme"], key="t_dist")
        params = {}
        c1, c2 = st.columns(2)
        if dist == "Binomial":
            params = {"n": c1.number_input("n", 1, 10000, 10, key="t_pn"), "p": c2.number_input("p", 0.0, 1.0, 0.5, key="t_pp")}
        elif dist == "Poisson":
            params = {"lambda": c1.number_input("λ", 0.0001, 1000.0, 3.0, key="t_pl")}
        elif dist == "Normal":
            params = {"mu": c1.number_input("μ", value=0.0, key="t_pm"), "sigma": c2.number_input("σ", 0.0001, 1000.0, 1.0, key="t_ps")}
        elif dist == "Exponencial":
            params = {"lambda": c1.number_input("λ", 0.0001, 1000.0, 1.0, key="t_pel")}
        else:
            params = {"a": c1.number_input("a", value=0.0, key="t_pua"), "b": c2.number_input("b", value=1.0, key="t_pub")}
        c3, c4, c5 = st.columns(3)
        xv = c3.number_input("x", value=2.0, key="t_px")
        av, bv = c4.number_input("a (intervalo)", value=-1.0, key="t_pa"), c5.number_input("b (intervalo)", value=1.0, key="t_pb")
        if W.btn("Calcular probabilidades", key="t_pgo", type="primary"):
            def go_():
                st.dataframe(pd.DataFrame({"valor": N.distribution(dist, params, xv, av, bv)}))
            W.run_guarded(go_)
