"""Ecuaciones, sistemas lineales (Gauss-Jordan) y determinantes paso a paso."""
from __future__ import annotations

import sympy as sp

from core import algebra
from core.parser import ExpressionError
from .model import COMMON_ERRORS, THEORY, Solution, Step, L, equivalent


def equation_solution(eq: sp.Eq, x, domain: str = "real") -> Solution:
    expr = sp.expand(eq.lhs - eq.rhs)
    sol = Solution(topic="Álgebra · resolución de ecuaciones", problem=rf"{L(eq.lhs)}={L(eq.rhs)}",
                   given=[f"Incógnita: ${L(x)}$", f"Dominio: {'reales' if domain == 'real' else 'complejos'}"],
                   goal=f"Hallar los valores de ${L(x)}$ que cumplen la ecuación.",
                   theory=THEORY["ecuacion"], errors=COMMON_ERRORS["ecuacion"])
    steps = [Step("Llevar todo a un lado", rf"{L(eq.lhs)}={L(eq.rhs)}", rf"{L(expr)}=0", r"A=B\iff A-B=0",
                  "Se pasan todos los términos al lado izquierdo y se desarrolla.",
                  "Una ecuación igualada a cero permite usar factorización o fórmulas de raíces.")]
    res = algebra.solve_equation(eq, x, domain)

    if expr.is_polynomial(x) and expr.has(x):
        p = sp.Poly(expr, x)
        deg = p.degree()
        if deg == 1:
            a, b = p.all_coeffs()
            steps.append(Step("Despejar la incógnita", rf"{L(a)}{L(x)}+{_p(b)}=0", rf"{L(x)}=-\frac{{{L(b)}}}{{{L(a)}}}={L(sp.simplify(-b / a))}",
                              r"ax+b=0\Rightarrow x=-\frac{b}{a}",
                              f"Se resta ${L(b)}$ en ambos lados y se divide entre ${L(a)}$.",
                              "Se realizan operaciones inversas para dejar x sola."))
            sol.method = "Ecuación lineal"
        elif deg == 2:
            a, b, c = p.all_coeffs()
            disc = sp.simplify(b ** 2 - 4 * a * c)
            steps.append(Step("Identificar coeficientes", rf"{L(expr)}=0", rf"a={L(a)},\;b={L(b)},\;c={L(c)}",
                              r"ax^{2}+bx+c=0", "Se comparan los coeficientes con la forma general.", ""))
            steps.append(Step("Calcular el discriminante", "", rf"\Delta=b^{{2}}-4ac=({L(b)})^{{2}}-4({L(a)})({L(c)})={L(disc)}",
                              r"\Delta=b^{2}-4ac",
                              "Se sustituyen los coeficientes.",
                              "El signo de Δ indica cuántas soluciones reales hay: Δ>0 dos, Δ=0 una doble, Δ<0 ninguna real (dos complejas)."))
            steps.append(Step("Aplicar la fórmula general", "", rf"{L(x)}=\frac{{-({L(b)})\pm\sqrt{{{L(disc)}}}}}{{2({L(a)})}}",
                              r"x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}", "Se reemplazan a, b y Δ.", "Es válida para toda ecuación cuadrática con a ≠ 0."))
            sol.method = "Fórmula general"
            if disc.is_negative and domain == "real":
                sol.notes.append("Δ < 0: no hay soluciones reales. Cambie el dominio a complejos para verlas.")
        else:
            fl = sp.factor(expr)
            if fl != expr:
                steps.append(Step("Factorizar el polinomio", rf"{L(expr)}=0", rf"{L(fl)}=0", r"a\cdot b=0\Rightarrow a=0\;\text{o}\;b=0",
                                  "Se factoriza (factor común, diferencia de cuadrados, raíces racionales...).",
                                  "Propiedad del producto nulo: un producto es cero sólo si algún factor es cero."))
            sol.method = f"Polinomio de grado {deg}"
    else:
        sol.method = "Resolución simbólica general"
        steps.append(Step("Ecuación no polinómica", rf"{L(expr)}=0", "", "",
                          "No es un polinomio en la incógnita; se resuelve con técnicas simbólicas (SymPy) o, si no hay forma cerrada, numéricas.",
                          "Se declara para no inventar pasos."))

    sols = res["solutions"]
    if res["set"] is not None and not sols:
        sol.result = rf"{L(x)}\in {L(res['set'])}"
    elif sols:
        body = r",\;".join(f"{L(x)}={L(s)}" for s in sols)
        sol.result = body + ("" if res["exact"] else r"\;\;(\text{aproximado})")
    else:
        sol.result = r"\text{Sin soluciones en el dominio elegido}"
    sol.exact = res["exact"]
    if res["note"]:
        sol.notes.append(res["note"])
    steps.append(Step("Solución", "", sol.result, "", "Se reúnen las soluciones obtenidas.", ""))
    sol.steps = steps

    checks = []
    for s in sols:
        try:
            r = sp.simplify(expr.subs(x, s))
            checks.append(f"${L(x)}={L(s)}$: $f={L(r)}$" if r == 0 or abs(complex(sp.N(r))) < 1e-8 else f"⚠ ${L(x)}={L(s)}$ no verifica")
        except Exception:  # noqa: BLE001
            pass
    sol.verification = ("✔ Sustitución en la ecuación (resultado 0): " + "; ".join(checks)) if checks else ""
    sol.final = sols  # type: ignore[attr-defined]
    return sol


def _p(v) -> str:
    return f"({L(v)})"


def gauss_jordan(M: sp.Matrix, pivot_cols: int | None = None):
    M = sp.Matrix(M)
    rows, cols = M.shape
    pivot_cols = cols if pivot_cols is None else pivot_cols
    steps, r = [], 0
    for c in range(pivot_cols):
        if r >= rows:
            break
        piv = next((i for i in range(r, rows) if M[i, c] != 0), None)
        if piv is None:
            continue
        if piv != r:
            before = M.copy()
            M.row_swap(piv, r)
            steps.append(Step(rf"Intercambiar F{r + 1} ↔ F{piv + 1}", L(before), L(M), r"F_i\leftrightarrow F_j",
                              f"La columna {c + 1} tiene un cero en la fila {r + 1}; se sube una fila con pivote no nulo.",
                              "Se necesita un pivote distinto de cero."))
        if M[r, c] != 1:
            before, k = M.copy(), sp.nsimplify(1 / M[r, c])
            M[r, :] = M[r, :] * k
            steps.append(Step(rf"F{r + 1} ← ({L(k)})·F{r + 1}", L(before), L(M), r"F_i\leftarrow kF_i",
                              f"Se divide la fila {r + 1} entre el pivote para obtener un 1 principal.",
                              "Un pivote igual a 1 facilita eliminar en las demás filas."))
        for i in range(rows):
            if i != r and M[i, c] != 0:
                before, k = M.copy(), M[i, c]
                M[i, :] = (M[i, :] - k * M[r, :]).applyfunc(sp.nsimplify)
                steps.append(Step(rf"F{i + 1} ← F{i + 1} − ({L(k)})·F{r + 1}", L(before), L(M), r"F_i\leftarrow F_i-kF_j",
                                  f"Se elimina la columna {c + 1} en la fila {i + 1}.",
                                  "Hacer ceros arriba y abajo del pivote es el objetivo de Gauss-Jordan."))
        r += 1
    return M, steps


def linear_system_solution(eqs: list, variables: list) -> Solution:
    A, b = sp.linear_eq_to_matrix([e.lhs - e.rhs for e in eqs], variables)
    aug = A.row_join(b)
    sol = Solution(topic="Álgebra lineal · sistema de ecuaciones lineales",
                   problem=r"\begin{cases}" + r"\\".join(f"{L(e.lhs)}={L(e.rhs)}" for e in eqs) + r"\end{cases}",
                   given=[f"Incógnitas: {', '.join('$' + L(v) + '$' for v in variables)}", f"{A.rows} ecuaciones, {A.cols} incógnitas"],
                   goal="Hallar el conjunto solución.", method="Eliminación de Gauss-Jordan sobre la matriz aumentada.",
                   theory=THEORY["algebra_lineal"], errors=COMMON_ERRORS["algebra_lineal"])
    sol.steps.append(Step("Matriz aumentada", "", L(aug), r"[A\,|\,b]", "Se escriben los coeficientes y los términos independientes.",
                          "Las operaciones por filas sobre [A|b] no cambian el conjunto solución."))
    R, steps = gauss_jordan(aug, pivot_cols=A.cols)
    sol.steps += steps
    solset = sp.linsolve((A, b), *variables)
    if solset == sp.S.EmptySet:
        sol.result = r"\text{Sistema incompatible: no tiene solución}"
        sol.steps.append(Step("Interpretar", L(R), "", "", "Hay una fila [0 … 0 | k] con k ≠ 0, que equivale a 0 = k.", "Es una contradicción."))
    else:
        s = list(solset)[0]
        sol.result = r",\;".join(f"{L(v)}={L(val)}" for v, val in zip(variables, s))
        free = [v for v in variables if any(sp.sympify(val).has(v) for val in s)]
        sol.steps.append(Step("Interpretar la forma escalonada reducida", L(R), sol.result, "",
                              "Se lee cada incógnita de su fila pivote" + (f"; las variables libres son {', '.join(map(L, free))}." if free else "."),
                              "Cada fila pivote expresa una variable en términos de las libres."))
        if free:
            sol.notes.append("El sistema tiene infinitas soluciones (variables libres).")
        else:
            subs = dict(zip(variables, s))
            ok = all(sp.simplify((e.lhs - e.rhs).subs(subs)) == 0 for e in eqs)
            sol.verification = "✔ Sustituyendo la solución en las ecuaciones originales se obtiene 0 = 0 en todas." if ok else "⚠ La sustitución no verifica."
    return sol


def determinant_solution(A: sp.Matrix) -> Solution:
    if A.rows != A.cols:
        raise ExpressionError("El determinante requiere una matriz cuadrada.")
    n = A.rows
    sol = Solution(topic="Álgebra lineal · determinante", problem=rf"\det{L(A)}", goal="Calcular el determinante.",
                   theory="det(A) es el factor de escala del volumen que produce la transformación lineal A; vale 0 si y sólo si A es singular.",
                   errors=COMMON_ERRORS["algebra_lineal"])
    if n == 1:
        sol.steps.append(Step("Matriz 1×1", "", L(A[0, 0]), r"\det[a]=a", "", ""))
    elif n == 2:
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        sol.steps.append(Step("Fórmula 2×2", L(A), rf"({L(a)})({L(d)})-({L(b)})({L(c)})", r"\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc",
                              "Producto de la diagonal principal menos producto de la secundaria.", "Es la definición para 2×2."))
    else:
        terms = []
        for j in range(n):
            minor = A.minor_submatrix(0, j)
            terms.append(rf"{'+' if j % 2 == 0 else '-'}\,({L(A[0, j])})\cdot\det{L(minor)}")
        sol.steps.append(Step("Desarrollo por cofactores a lo largo de la fila 1", L(A), "".join(terms),
                              r"\det A=\sum_{j}(-1)^{1+j}a_{1j}M_{1j}",
                              "Se multiplica cada elemento de la primera fila por su cofactor (signo alternado × menor).",
                              "Reduce un determinante n×n a determinantes (n−1)×(n−1)."))
        if n > 3:
            sol.notes.append("Para n > 3 se muestra el primer desarrollo; los menores se evalúan internamente.")
    d = sp.simplify(A.det())
    sol.steps.append(Step("Resultado", "", L(d), "", "Se evalúan y suman los términos.", ""))
    sol.result = rf"\det{L(A)}={L(d)}"
    sol.verification = "✔ Calculado de forma exacta (aritmética racional/simbólica)."
    sol.final = d  # type: ignore[attr-defined]
    return sol
