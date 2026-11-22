"""Página: álgebra, ecuaciones, polinomios y álgebra lineal."""
from __future__ import annotations

import sympy as sp
import streamlit as st

from core import algebra, linalg
from core.parser import ExpressionError, parse_equation, parse_expression, parse_matrix, parse_symbol, parse_symbols
from tutor import equations
from tutor.model import L
from . import widgets as W


def render(settings) -> None:
    st.title("Álgebra y álgebra lineal")
    t1, t2, t3, t4 = st.tabs(["Simplificar / factorizar", "Ecuaciones y sistemas", "Polinomios", "Matrices"])

    with t1:
        e_txt = st.text_input("Expresión", "(x^2 - 1)/(x - 1) + sin(x)^2 + cos(x)^2", key="al_e")
        op = st.selectbox("Operación", list(algebra.SIMPLIFY_OPS), key="al_op")
        if W.btn("Aplicar", key="al_go", type="primary"):
            def go():
                e = parse_expression(e_txt, settings.angle)
                r = algebra.apply_operation(op, e)
                W.show_result(rf"{L(e)}\;\longrightarrow\;{L(r)}", str(r), None, True)
                W.add_history(op, e_txt, L(r), str(r))
            W.run_guarded(go)

    with t2:
        mode = st.radio("Tipo", ["Una ecuación", "Sistema de ecuaciones"], horizontal=True, key="al_m")
        if mode == "Una ecuación":
            eq_txt = st.text_input("Ecuación", "x^2 - 5x + 6 = 0", key="al_eq")
            c1, c2 = st.columns(2)
            var = c1.text_input("Incógnita", "x", key="al_var")
            dom = c2.radio("Dominio", ["real", "complejo"], horizontal=True, key="al_dom")
            if W.btn("Resolver", key="al_solve", type="primary"):
                def go():
                    sol = equations.equation_solution(parse_equation(eq_txt, settings.angle), parse_symbol(var), "real" if dom == "real" else "complex")
                    W.add_history("Ecuación", eq_txt, sol.result, sol.result)
                    if settings.mode == "Estudiante":
                        W.render_solution(sol, settings.level, key="al_eqsol")
                    else:
                        W.show_result(sol.result, sol.result, None, sol.exact)
                W.run_guarded(go)
        else:
            sys_txt = st.text_area("Una ecuación por línea", "2x + y = 5\nx - y = 1", key="al_sys", height=100)
            vs = st.text_input("Incógnitas", "x, y", key="al_sv")
            if W.btn("Resolver sistema", key="al_ssolve", type="primary"):
                def go():
                    eqs = [parse_equation(l) for l in sys_txt.splitlines() if l.strip()]
                    vars_ = parse_symbols(vs)
                    if not eqs:
                        raise ExpressionError("Ingrese al menos una ecuación.")
                    is_linear = all(sp.Poly(sp.expand(e.lhs - e.rhs), *vars_).total_degree() <= 1 for e in eqs
                                    if sp.expand(e.lhs - e.rhs).is_polynomial(*vars_)) and all(sp.expand(e.lhs - e.rhs).is_polynomial(*vars_) for e in eqs)
                    if is_linear:
                        sol = equations.linear_system_solution(eqs, vars_)
                        W.add_history("Sistema lineal", sys_txt.replace("\n", "; "), sol.result, sol.result)
                        if settings.mode == "Estudiante":
                            W.render_solution(sol, settings.level, key="al_sysol")
                        else:
                            W.show_result(sol.result, sol.result, None, True)
                    else:
                        res = algebra.solve_system(eqs, vars_)["solutions"]
                        if not res:
                            st.warning("No se encontraron soluciones.")
                        for s in res:
                            W.show_result(",\\;".join(f"{L(k)}={L(val)}" for k, val in s.items()), str(s), None, True)
                        st.info("Sistema no lineal: resuelto simbólicamente (sin desglose paso a paso).")
                W.run_guarded(go)

    with t3:
        p_txt = st.text_input("Polinomio", "x^3 - 6x^2 + 11x - 6", key="al_p")
        pv = st.text_input("Variable", "x", key="al_pv")
        if W.btn("Analizar polinomio", key="al_pgo", type="primary"):
            def go():
                info = algebra.polynomial_info(parse_expression(p_txt), parse_symbol(pv))
                st.latex(rf"\text{{grado}}={info['grado']}")
                st.latex(rf"\text{{coeficientes}}={L(info['coeficientes'])}")
                st.latex(rf"\text{{factorizado}}:\;{L(info['factorizado'])}")
                st.latex(r"\text{raíces}:\;" + r",\;".join(f"{L(r)}\\ (m={m})" for r, m in info["raices"].items()))
                st.caption("Raíces numéricas: " + ", ".join(str(r) for r in info["raices_numericas"]))
            W.run_guarded(go)

    with t4:
        c1, c2 = st.columns(2)
        a_txt = c1.text_area("Matriz A (filas con «;» o salto de línea, columnas con «,»)", "2, 1, 0\n1, 3, 1\n0, 1, 2", key="al_A", height=110)
        b_txt = c2.text_area("Matriz B (si la operación la necesita)", "1, 0, 1\n0, 1, 0\n1, 0, 1", key="al_B", height=110)
        op = st.selectbox("Operación", linalg.OPERATIONS, key="al_mop")
        n = st.number_input("n (para Aⁿ)", 1, 20, 2, key="al_n") if op.startswith("Potencia") else 2
        if W.btn("Calcular", key="al_mgo", type="primary"):
            def go():
                A = parse_matrix(a_txt)
                B = parse_matrix(b_txt) if op in linalg.NEEDS_B else None
                if op == "Determinante" and settings.mode == "Estudiante":
                    sol = equations.determinant_solution(A)
                    W.render_solution(sol, settings.level, key="al_det")
                    W.add_history("Determinante", a_txt.replace("\n", "; "), sol.result)
                    return
                for title, tex in linalg.operate(op, A, B, int(n)):
                    st.markdown(f"**{title}**")
                    st.latex(tex)
                W.add_history(op, a_txt.replace("\n", "; "), tex)
                if op.startswith("Forma escalonada") and settings.mode == "Estudiante":
                    _, steps = equations.gauss_jordan(A)
                    with st.expander("Ver las operaciones por filas (Gauss-Jordan)"):
                        for i, s in enumerate(steps, 1):
                            st.markdown(f"**{i}. {s.title}**")
                            st.latex(s.after)
            W.run_guarded(go)
