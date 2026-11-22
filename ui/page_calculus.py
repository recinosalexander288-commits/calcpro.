"""Página: módulo de cálculo (diferencial, integral, multivariable, vectorial)."""
from __future__ import annotations

import sympy as sp
import streamlit as st

from core import calculus
from core.parser import (ExpressionError, parse_equation, parse_expression, parse_symbol, parse_symbols,
                         parse_vector, split_top_level)
from core.calculator import format_number
from tutor import derivatives, integrals, limits, multiple, vector
from tutor.model import L
from . import widgets as W

OPS = {
    "Diferencial": ["Derivada", "Derivada de orden superior", "Derivada parcial", "Derivada implícita", "Derivada direccional",
                    "Gradiente", "Divergencia", "Rotacional", "Jacobiano", "Hessiano", "Puntos críticos (máx/mín/silla)",
                    "Optimización con restricciones (Lagrange)", "Límite"],
    "Integral": ["Integral indefinida", "Integral definida", "Integral impropia", "Integral doble", "Integral triple",
                 "Integral de línea (escalar)", "Integral de línea (campo vectorial)", "Integral de superficie (escalar)",
                 "Flujo de un campo vectorial", "Longitud de arco", "Área de una superficie z = f(x, y)", "Volumen bajo z = f(x, y)"],
}


def _out(settings, kind, problem_expr, sol=None, latex=None, copy=None, topic=None, key="c", exact=True):
    """Presenta el resultado según el modo (Estudiante/Profesional)."""
    res_latex = latex if latex is not None else sol.result
    copy_text = copy if copy is not None else res_latex
    W.add_history(kind, problem_expr, res_latex, copy_text)
    if settings.mode == "Estudiante" and sol is not None:
        W.render_solution(sol, settings.level, key=key)
        if topic:
            W.similar_panel(topic, key)
    else:
        W.show_result(res_latex, copy_text, None, exact if sol is None else sol.exact)
        if sol is not None:
            for n in sol.notes:
                st.info(n)
            if sol.verification:
                st.caption(sol.verification)


def _num_or_exact(d: dict, settings):
    """Texto LaTeX + aviso a partir del dict que devuelve calculus.definite/iterated."""
    if d.get("exact") is not None:
        return L(d["exact"]), True, d.get("note", "")
    if d.get("numeric") is not None:
        v, e = d["numeric"]
        return rf"\approx {format_number(v, settings.precision)}", False, d.get("note", "")
    return r"\text{no se pudo evaluar}", False, d.get("note", "")


def _limits_input(prefix, n, defaults):
    """Entradas de límites; devuelve lista [(var, lo, hi)] con la variable MÁS INTERNA primero."""
    out = []
    st.caption("Ingrese los límites de la integral más interna a la más externa.")
    for i in range(n):
        c0, c1, c2 = st.columns([1, 2, 2])
        v = c0.text_input(f"Variable {i + 1}", defaults[i][0], key=f"{prefix}_v{i}")
        lo = c1.text_input(f"Inferior {i + 1}", defaults[i][1], key=f"{prefix}_lo{i}")
        hi = c2.text_input(f"Superior {i + 1}", defaults[i][2], key=f"{prefix}_hi{i}")
        out.append((parse_symbol(v), parse_expression(lo, extra_symbols=("theta", "phi", "rho")),
                    parse_expression(hi, extra_symbols=("theta", "phi", "rho"))))
    return out


def render(settings) -> None:
    st.title("Cálculo")
    area = st.radio("Área", list(OPS), horizontal=True)
    op = st.selectbox("Operación", OPS[area])
    ang = settings.angle if op not in () else "rad"
    key = "calc_" + op[:6]

    def P(text, **k):
        return parse_expression(text, "rad", **k)

    # ───────── DIFERENCIAL ─────────
    if op in ("Derivada", "Derivada de orden superior", "Derivada parcial"):
        f_txt = st.text_input("f", "x^2*sin(x)" if op != "Derivada parcial" else "x^2*y + sin(x*y)", key=key + "f")
        v_txt = st.text_input("Variable de derivación", "x", key=key + "v")
        order = st.number_input("Orden", 1, 6, 2 if "superior" in op else 1) if "superior" in op else 1
        alt = settings.mode == "Estudiante" and op == "Derivada" and st.checkbox("Mostrar método alternativo (definición por límite)")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                f, x = P(f_txt), parse_symbol(v_txt)
                sol = derivatives.derivative_solution(f, x, int(order))
                _out(settings, "Derivada", f_txt, sol, topic="derivada", key=key)
                if alt:
                    st.markdown("---")
                    st.subheader("Método alternativo")
                    W.render_solution(derivatives.derivative_by_definition(f, x), settings.level, key=key + "alt")
            W.run_guarded(go)

    elif op == "Derivada implícita":
        eq_txt = st.text_input("Ecuación F(x, y) = 0", "x^2 + y^2 = 25", key=key + "e")
        if W.btn("Calcular dy/dx", key=key + "go", type="primary"):
            def go():
                eq = parse_equation(eq_txt)
                x, y = sp.symbols("x y")
                F = eq.lhs - eq.rhs
                d = calculus.implicit_derivative(eq, y, x)
                sol = __import__("tutor.model", fromlist=["Solution"]).Solution(
                    topic="Cálculo diferencial · derivación implícita", problem=rf"{L(eq.lhs)}={L(eq.rhs)}",
                    goal="Hallar dy/dx sin despejar y.", method="dy/dx = −F_x / F_y",
                    given=[f"$F(x,y)={L(sp.simplify(F))}$"])
                Fx, Fy = sp.diff(F, x), sp.diff(F, y)
                St = __import__("tutor.model", fromlist=["Step"]).Step
                sol.steps = [
                    St("Pasar todo a un lado", "", rf"F(x,y)={L(sp.simplify(F))}=0", r"F(x,y)=0", "", "Se obtiene una curva de nivel F=0."),
                    St("Derivadas parciales", "", rf"F_x={L(Fx)},\quad F_y={L(Fy)}", r"\frac{dy}{dx}=-\frac{F_x}{F_y}",
                       "Se deriva F respecto de x (y constante) y respecto de y (x constante).",
                       "Al derivar F(x,y(x))=0 con la regla de la cadena, F_x + F_y·y′ = 0."),
                    St("Despejar dy/dx", "", rf"\frac{{dy}}{{dx}}=-\frac{{{L(Fx)}}}{{{L(Fy)}}}={L(d)}", "", "Se divide entre F_y (debe ser ≠ 0).", "")]
                sol.result = rf"\frac{{dy}}{{dx}}={L(d)}"
                _out(settings, "Derivada implícita", eq_txt, sol, key=key)
            W.run_guarded(go)

    elif op == "Derivada direccional":
        f_txt = st.text_input("f(x, y, …)", "x^2*y + z", key=key + "f")
        vs = st.text_input("Variables", "x, y, z", key=key + "vs")
        d_txt = st.text_input("Dirección v", "(1, 2, 2)", key=key + "d")
        p_txt = st.text_input("Punto (opcional)", "(1, 1, 1)", key=key + "p")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                vs_ = parse_symbols(vs)
                f = P(f_txt)
                d = parse_vector(d_txt, len(vs_))
                pt = dict(zip(vs_, parse_vector(p_txt, len(vs_)))) if p_txt.strip() else None
                _out(settings, "Derivada direccional", f_txt, vector.directional_solution(f, vs_, list(d), pt), key=key)
            W.run_guarded(go)

    elif op in ("Gradiente", "Hessiano"):
        f_txt = st.text_input("f", "x^2*y + sin(z)*y", key=key + "f")
        vs = st.text_input("Variables", "x, y, z", key=key + "vs")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                vs_, f = parse_symbols(vs), P(f_txt)
                fn = vector.gradient_solution if op == "Gradiente" else vector.hessian_solution
                _out(settings, op, f_txt, fn(f, vs_), key=key)
            W.run_guarded(go)

    elif op in ("Divergencia", "Rotacional", "Jacobiano"):
        F_txt = st.text_input("Campo / función vectorial F", "(x*y, y*z, z*x)", key=key + "F")
        vs = st.text_input("Variables", "x, y, z", key=key + "vs")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                vs_ = parse_symbols(vs)
                F = parse_vector(F_txt)
                fn = {"Divergencia": vector.divergence_solution, "Rotacional": vector.curl_solution,
                      "Jacobiano": vector.jacobian_solution}[op]
                _out(settings, op, F_txt, fn(F, vs_), key=key)
            W.run_guarded(go)

    elif op.startswith("Puntos críticos"):
        f_txt = st.text_input("f", "x^3 - 3*x + y^2", key=key + "f")
        vs = st.text_input("Variables", "x, y", key=key + "vs")
        if W.btn("Hallar y clasificar", key=key + "go", type="primary"):
            W.run_guarded(lambda: _out(settings, "Puntos críticos", f_txt,
                                       vector.critical_points_solution(P(f_txt), parse_symbols(vs)), key=key))

    elif op.startswith("Optimización"):
        f_txt = st.text_input("Función objetivo f", "x*y", key=key + "f")
        g_txt = st.text_input("Restricciones g = 0 (una por línea, puede escribir g = c)", "x + y = 10", key=key + "g")
        vs = st.text_input("Variables", "x, y", key=key + "vs")
        if W.btn("Optimizar", key=key + "go", type="primary"):
            def go():
                cons = []
                for line in g_txt.splitlines():
                    if line.strip():
                        eq = parse_equation(line)
                        cons.append(sp.simplify(eq.lhs - eq.rhs))
                if not cons:
                    raise ExpressionError("Ingrese al menos una restricción.")
                _out(settings, "Lagrange", f_txt, vector.lagrange_solution(P(f_txt), cons, parse_symbols(vs)), key=key)
            W.run_guarded(go)

    elif op == "Límite":
        f_txt = st.text_input("f(x)", "(sin(3x))/x", key=key + "f")
        c1, c2, c3 = st.columns([2, 2, 2])
        v_txt = c1.text_input("Variable", "x", key=key + "v")
        a_txt = c2.text_input("Tiende a (use oo para ∞)", "0", key=key + "a")
        side = c3.selectbox("Dirección", ["Ambos lados", "Por la derecha (+)", "Por la izquierda (−)"], key=key + "s")
        pref = st.selectbox("Método preferido", ["auto", "algebraico", "lhopital"], key=key + "m") if settings.mode == "Estudiante" else "auto"
        if W.btn("Calcular límite", key=key + "go", type="primary"):
            def go():
                d = {"Ambos lados": "+-", "Por la derecha (+)": "+", "Por la izquierda (−)": "-"}[side]
                sol = limits.limit_solution(P(f_txt), parse_symbol(v_txt), P(a_txt), d, pref)
                _out(settings, "Límite", f"lim {v_txt}->{a_txt} {f_txt}", sol, topic="limite", key=key)
            W.run_guarded(go)

    # ───────────── INTEGRAL ─────────────
    elif op in ("Integral indefinida", "Integral definida", "Integral impropia"):
        f_txt = st.text_input("Integrando f(x)", "x*exp(x)" if op == "Integral indefinida" else "exp(-x^2)", key=key + "f")
        v_txt = st.text_input("Variable", "x", key=key + "v")
        a_txt = b_txt = None
        if op != "Integral indefinida":
            c1, c2 = st.columns(2)
            a_txt = c1.text_input("Límite inferior", "-oo" if op == "Integral impropia" else "0", key=key + "a")
            b_txt = c2.text_input("Límite superior", "oo" if op == "Integral impropia" else "1", key=key + "b")
        pref = st.selectbox("Método preferido (alternativo)", ["auto", "sustitución", "partes"], key=key + "m") if settings.mode == "Estudiante" else "auto"
        if W.btn("Integrar", key=key + "go", type="primary"):
            def go():
                f, x = P(f_txt), parse_symbol(v_txt)
                a = P(a_txt) if a_txt else None
                b = P(b_txt) if b_txt else None
                from core.safety import run_with_timeout
                sol = run_with_timeout(integrals.integral_solution, 40, f, x, a, b, pref)
                label = f"∫ {f_txt} d{v_txt}" + (f" [{a_txt},{b_txt}]" if a_txt else "")
                _out(settings, op, label, sol, topic="integral", key=key)
            W.run_guarded(go)

    elif op in ("Integral doble", "Integral triple"):
        dim = 2 if "doble" in op else 3
        system = st.selectbox("Sistema de coordenadas", ["cartesianas", "polares", "cilíndricas", "esféricas"] if dim == 3 else ["cartesianas", "polares"], key=key + "sys")
        if dim == 3 and system == "polares":
            system = "cartesianas"
        defaults = {
            "cartesianas": ([("x", "0", "y"), ("y", "0", "1")] if dim == 2 else [("x", "0", "1"), ("y", "0", "1"), ("z", "0", "1")]),
            "polares": [("r", "0", "2"), ("theta", "0", "2*pi")],
            "cilíndricas": [("z", "0", "4"), ("r", "0", "2"), ("theta", "0", "2*pi")],
            "esféricas": [("rho", "0", "2"), ("phi", "0", "pi"), ("theta", "0", "2*pi")],
        }[system]
        f_def = {"cartesianas": "x*y" if dim == 2 else "x+y+z", "polares": "x^2 + y^2", "cilíndricas": "z", "esféricas": "1"}[system]
        f_txt = st.text_input("Integrando (en cartesianas x, y, z — se convierte automáticamente)", f_def, key=key + "f")
        conv = st.checkbox("El integrando ya está escrito en las nuevas variables (r, θ, z, ρ, φ)", key=key + "cv") if system != "cartesianas" else False
        lims = None
        try:
            lims = _limits_input(key + system, dim, defaults)
        except ExpressionError as e:
            st.error(f"⚠️ {e}")
        if W.btn("Calcular", key=key + "go", type="primary") and lims:
            def go():
                from core.safety import run_with_timeout
                f = P(f_txt, extra_symbols=("theta", "phi", "rho"))
                sol = run_with_timeout(multiple.multiple_integral_solution, 40, f, lims, system, conv)
                _out(settings, op, f_txt, sol, key=key)
            W.run_guarded(go)

    elif op.startswith("Integral de línea"):
        scalar = "escalar" in op
        n = st.radio("Dimensión", [2, 3], horizontal=True, index=1, key=key + "n")
        r_txt = st.text_input("Curva r(t)", "(cos(t), sin(t), t)" if n == 3 else "(cos(t), sin(t))", key=key + "r")
        c1, c2 = st.columns(2)
        a_txt, b_txt = c1.text_input("t desde", "0", key=key + "a"), c2.text_input("t hasta", "2*pi", key=key + "b")
        f_txt = st.text_input("f(x, y, z)" if scalar else "Campo F", "x^2 + y^2 + z" if scalar else "(-y, x, z)", key=key + "f")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                t = sp.Symbol("t")
                r = list(parse_vector(r_txt, n))
                a, b = P(a_txt), P(b_txt)
                if scalar:
                    res = calculus.line_integral_scalar(P(f_txt), r, t, a, b)
                    title, formula = "Integral de línea escalar", r"\int_C f\,ds=\int_a^b f(\mathbf r(t))\,\|\mathbf r'(t)\|\,dt"
                else:
                    res = calculus.line_integral_vector(parse_vector(f_txt, n), r, t, a, b)
                    title, formula = "Integral de línea de un campo", r"\int_C\mathbf F\cdot d\mathbf r=\int_a^b\mathbf F(\mathbf r(t))\cdot\mathbf r'(t)\,dt"
                txt, exact, note = _num_or_exact(res["result"], settings)
                from tutor.model import Solution, Step
                sol = Solution(topic=f"Cálculo vectorial · {title.lower()}", problem=formula, goal="Parametrizar la curva y reducir a una integral en t.",
                               method="Parametrización", exact=exact, notes=[note] if note else [])
                sol.steps = [Step("Parametrización", "", rf"\mathbf r(t)={L(sp.Matrix(r))}", "", f"t ∈ [{a_txt}, {b_txt}].", ""),
                             Step("Integrando en términos de t", "", L(res["integrand"]), formula, "Se sustituye x(t), y(t), z(t) y se multiplica por el factor de arco / r′(t).", "Convierte la integral de línea en una integral ordinaria."),
                             Step("Integrar en t", "", txt, "", "Se evalúa la integral definida.", "")]
                sol.result = txt
                _out(settings, title, f_txt, sol, key=key, exact=exact)
            W.run_guarded(go)

    elif op in ("Integral de superficie (escalar)", "Flujo de un campo vectorial"):
        scalar = op.startswith("Integral")
        r_txt = st.text_input("Superficie r(u, v) = (x, y, z)", "(sin(u)*cos(v), sin(u)*sin(v), cos(u))", key=key + "r")
        c1, c2 = st.columns(2)
        ul, vl = c1.text_input("u ∈ [a, b]", "0, pi", key=key + "u"), c2.text_input("v ∈ [c, d]", "0, 2*pi", key=key + "vv")
        f_txt = st.text_input("f(x, y, z)" if scalar else "Campo F", "z^2" if scalar else "(x, y, z)", key=key + "f")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                u, v = sp.symbols("u v")
                r = list(parse_vector(r_txt, 3))
                (u0, u1), (v0, v1) = [tuple(P(s) for s in split_top_level(t)) for t in (ul, vl)]
                if scalar:
                    res = calculus.surface_integral_scalar(P(f_txt), r, u, v, (u0, u1), (v0, v1))
                else:
                    res = calculus.flux_integral(parse_vector(f_txt, 3), r, u, v, (u0, u1), (v0, v1))
                txt, exact, note = _num_or_exact(res["result"], settings)
                from tutor.model import Solution, Step
                sol = Solution(topic="Cálculo vectorial · " + ("integral de superficie" if scalar else "flujo"),
                               problem=r"\iint_S f\,dS=\iint f(\mathbf r)\,\|\mathbf r_u\times\mathbf r_v\|\,du\,dv" if scalar else r"\iint_S\mathbf F\cdot d\mathbf S=\iint\mathbf F(\mathbf r)\cdot(\mathbf r_u\times\mathbf r_v)\,du\,dv",
                               method="Parametrización de la superficie", exact=exact, notes=[note] if note else [])
                sol.steps = [Step("Vector normal", "", rf"\mathbf r_u\times\mathbf r_v={L(res['normal'])}", r"\mathbf n=\mathbf r_u\times\mathbf r_v", "Se derivan r respecto de u y de v y se hace el producto vectorial.", "Es perpendicular a la superficie; su norma es el factor de área dS."),
                             Step("Integrando en (u, v)", "", L(res["integrand"]), "", "Se sustituye la parametrización en el integrando.", ""),
                             Step("Integral doble", "", txt, "", "Se integra sobre el dominio de los parámetros.", "")]
                sol.result = txt
                _out(settings, op, f_txt, sol, key=key, exact=exact)
            W.run_guarded(go)

    elif op == "Longitud de arco":
        kind = st.radio("Curva", ["y = f(x)", "r(t) paramétrica"], horizontal=True, key=key + "k")
        if kind == "y = f(x)":
            f_txt = st.text_input("f(x)", "x^2", key=key + "f")
        else:
            f_txt = st.text_input("r(t)", "(cos(t), sin(t), t)", key=key + "f")
        c1, c2 = st.columns(2)
        a_txt, b_txt = c1.text_input("Desde", "0", key=key + "a"), c2.text_input("Hasta", "1", key=key + "b")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                a, b = P(a_txt), P(b_txt)
                if kind == "y = f(x)":
                    x = sp.Symbol("x")
                    res = calculus.arc_length_graph(P(f_txt), x, a, b)
                else:
                    res = calculus.arc_length_param(list(parse_vector(f_txt)), sp.Symbol("t"), a, b)
                txt, exact, note = _num_or_exact(res["result"], settings)
                from tutor.model import Solution, Step
                sol = Solution(topic="Aplicaciones de la integral · longitud de arco", problem=r"s=\int_a^b\sqrt{1+\left(f'(x)\right)^{2}}\,dx" if kind == "y = f(x)" else r"s=\int_a^b\|\mathbf r'(t)\|\,dt",
                               method="Integral de la rapidez", exact=exact, notes=[note] if note else [])
                sol.steps = [Step("Rapidez / elemento de arco", "", L(res["speed"]), "", "Se deriva y se calcula la norma.", "ds = (rapidez)·dt"),
                             Step("Integrar", "", txt, "", "Se integra entre los límites.", "")]
                sol.result = rf"s={txt}"
                _out(settings, "Longitud de arco", f_txt, sol, key=key, exact=exact)
            W.run_guarded(go)

    else:  # área de superficie / volumen
        f_txt = st.text_input("f(x, y)", "x^2 + y^2", key=key + "f")
        c1, c2 = st.columns(2)
        xl, yl = c1.text_input("x ∈ [a, b]", "0, 1", key=key + "x"), c2.text_input("y ∈ [c, d]", "0, 1", key=key + "y")
        if W.btn("Calcular", key=key + "go", type="primary"):
            def go():
                x, y = sp.symbols("x y")
                xs = tuple(P(s) for s in split_top_level(xl))
                ys = tuple(P(s) for s in split_top_level(yl))
                f = P(f_txt)
                if op.startswith("Área"):
                    res = calculus.surface_area_graph(f, x, y, xs, ys)
                    expl = ("dS", res["dS"], r"A=\iint\sqrt{1+f_x^{2}+f_y^{2}}\,dA")
                else:
                    res = calculus.volume_under_graph(f, x, y, xs, ys)
                    expl = ("f", f, r"V=\iint f(x,y)\,dA")
                txt, exact, note = _num_or_exact(res["result"], settings)
                from tutor.model import Solution, Step
                sol = Solution(topic="Aplicaciones de la integral doble", problem=expl[2], method="Integral doble", exact=exact, notes=[note] if note else [])
                sol.steps = [Step("Integrando", "", L(expl[1]), expl[2], "", ""), Step("Integral iterada", "", txt, "", "Se integra primero en x y luego en y.", "")]
                sol.result = txt
                _out(settings, op, f_txt, sol, key=key, exact=exact)
            W.run_guarded(go)
