"""Cálculo vectorial y optimización con pasos (gradiente, div, rot, Jacobiano, Hessiano, críticos, Lagrange)."""
from __future__ import annotations

import sympy as sp

from core import calculus
from .model import COMMON_ERRORS, THEORY, Solution, Step, L


def _partials(f, vars_):
    return [Step(f"Derivada parcial respecto de ${L(v)}$", L(sp.Derivative(f, v)), L(sp.diff(f, v)), "",
                 "Se deriva respecto de esa variable tratando las demás como constantes.",
                 "Cada componente mide la razón de cambio en una dirección coordenada.") for v in vars_]


def gradient_solution(f, vars_) -> Solution:
    g = calculus.gradient(f, vars_)
    sol = Solution(topic="Cálculo vectorial · gradiente", problem=rf"\nabla f,\quad f={L(f)}",
                   given=[f"$f={L(f)}$", f"Variables: {', '.join('$' + L(v) + '$' for v in vars_)}"],
                   goal="Calcular ∇f.", method="Derivadas parciales", theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    sol.steps = [Step("Definición", "", r"\nabla f=\left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right)", r"\nabla f=\left(\frac{\partial f}{\partial x_1},\dots\right)",
                      "El gradiente reúne las derivadas parciales en un vector.", "Apunta hacia la dirección de máximo crecimiento de f.")]
    sol.steps += _partials(f, vars_)
    sol.steps.append(Step("Armar el vector", "", L(g), "", "Se escriben las parciales como componentes.", ""))
    sol.result = rf"\nabla f={L(g)}"
    sol.final = g  # type: ignore[attr-defined]
    return sol


def divergence_solution(F, vars_) -> Solution:
    F = sp.Matrix(F)
    d = calculus.divergence(F, vars_)
    sol = Solution(topic="Cálculo vectorial · divergencia", problem=rf"\nabla\cdot\mathbf{{F}},\quad \mathbf{{F}}={L(F)}",
                   goal="Calcular ∇·F.", method="Suma de derivadas parciales de cada componente",
                   theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    sol.steps = [Step("Definición", "", r"\nabla\cdot\mathbf F=\frac{\partial F_1}{\partial x}+\frac{\partial F_2}{\partial y}+\frac{\partial F_3}{\partial z}", "",
                      "La divergencia es un escalar: suma de ∂Fᵢ/∂xᵢ.", "Mide cuánto «se expande» el campo en un punto.")]
    for i, v in enumerate(vars_):
        sol.steps.append(Step(f"∂F{i + 1}/∂{L(v)}", L(sp.Derivative(F[i], v)), L(sp.diff(F[i], v)), "", "", ""))
    sol.steps.append(Step("Sumar", "", L(sp.simplify(sum(sp.diff(F[i], v) for i, v in enumerate(vars_)))), "", "Se suman las tres derivadas.", ""))
    sol.result = rf"\nabla\cdot\mathbf F={L(d)}"
    sol.final = d  # type: ignore[attr-defined]
    return sol


def curl_solution(F, vars_) -> Solution:
    F = sp.Matrix(F)
    c = calculus.curl(F, vars_)
    sol = Solution(topic="Cálculo vectorial · rotacional", problem=rf"\nabla\times\mathbf{{F}},\quad \mathbf{{F}}={L(F)}",
                   goal="Calcular ∇×F.", method="Determinante simbólico", theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    if len(F) == 3:
        x, y, z = vars_
        sol.steps = [
            Step("Definición", "", r"\nabla\times\mathbf F=\begin{vmatrix}\mathbf i&\mathbf j&\mathbf k\\\partial_x&\partial_y&\partial_z\\F_1&F_2&F_3\end{vmatrix}", "",
                 "Se expande el determinante simbólico.", "Mide la tendencia del campo a «girar» alrededor de un punto."),
            Step("Componente i", "", rf"\partial_y F_3-\partial_z F_2={L(sp.diff(F[2], y))}-({L(sp.diff(F[1], z))})", "", "", ""),
            Step("Componente j", "", rf"\partial_z F_1-\partial_x F_3={L(sp.diff(F[0], z))}-({L(sp.diff(F[2], x))})", "", "", ""),
            Step("Componente k", "", rf"\partial_x F_2-\partial_y F_1={L(sp.diff(F[1], x))}-({L(sp.diff(F[0], y))})", "", "", ""),
        ]
    sol.result = rf"\nabla\times\mathbf F={L(c)}"
    sol.final = c  # type: ignore[attr-defined]
    return sol


def jacobian_solution(F, vars_) -> Solution:
    J = calculus.jacobian(F, vars_)
    sol = Solution(topic="Cálculo vectorial · matriz Jacobiana", problem=rf"J=\frac{{\partial \mathbf F}}{{\partial \mathbf x}},\quad \mathbf F={L(sp.Matrix(F))}",
                   goal="Calcular la matriz de derivadas parciales.", method="J[i,j] = ∂Fᵢ/∂xⱼ", theory=THEORY["vectorial"])
    sol.steps = [Step("Definición", "", r"J_{ij}=\frac{\partial F_i}{\partial x_j}", "", "Cada fila es el gradiente de una componente.", "")]
    for i, comp in enumerate(F):
        sol.steps.append(Step(f"Fila {i + 1}: gradiente de ${L(comp)}$", "", L(sp.Matrix([[sp.diff(comp, v) for v in vars_]])), "", "", ""))
    sol.result = rf"J={L(J)}" + (rf",\quad \det J={L(sp.simplify(J.det()))}" if J.is_square else "")
    sol.final = J  # type: ignore[attr-defined]
    return sol


def hessian_solution(f, vars_) -> Solution:
    H = calculus.hessian(f, vars_)
    sol = Solution(topic="Cálculo vectorial · matriz Hessiana", problem=rf"H(f),\quad f={L(f)}", goal="Calcular la matriz de segundas derivadas.",
                   method="H[i,j] = ∂²f/∂xᵢ∂xⱼ", theory=THEORY["vectorial"])
    sol.steps = [Step("Definición", "", r"H_{ij}=\frac{\partial^{2}f}{\partial x_i\,\partial x_j}", "", "Es simétrica si las segundas derivadas son continuas (teorema de Clairaut).", "")]
    for i, vi in enumerate(vars_):
        for j, vj in enumerate(vars_):
            if j >= i:
                sol.steps.append(Step(rf"$\partial^2 f/\partial {L(vi)}\partial {L(vj)}$", L(sp.Derivative(f, vi, vj)), L(H[i, j]), "", "", ""))
    sol.result = rf"H={L(H)}"
    sol.final = H  # type: ignore[attr-defined]
    return sol


def directional_solution(f, vars_, direction, point=None) -> Solution:
    val, u = calculus.directional_derivative(f, vars_, direction, point)
    g = calculus.gradient(f, vars_)
    sol = Solution(topic="Cálculo vectorial · derivada direccional", problem=rf"D_{{\mathbf u}}f,\quad f={L(f)}",
                   given=[f"Dirección: ${L(sp.Matrix(direction))}$"], goal="Calcular D_u f = ∇f · u con u unitario.",
                   method="Gradiente y vector unitario", theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    sol.steps = [
        Step("Gradiente", "", L(g), r"\nabla f", "Primero se calcula ∇f.", ""),
        Step("Vector unitario", rf"\mathbf v={L(sp.Matrix(direction))}", rf"\mathbf u=\frac{{\mathbf v}}{{\|\mathbf v\|}}={L(u)}", r"\mathbf u=\mathbf v/\|\mathbf v\|",
             "La fórmula exige un vector de longitud 1.", "Si v no es unitario el resultado quedaría multiplicado por |v|."),
        Step("Producto punto", "", rf"D_{{\mathbf u}}f=\nabla f\cdot\mathbf u={L(sp.simplify(g.dot(u)))}", r"D_{\mathbf u}f=\nabla f\cdot\mathbf u", "", "")]
    if point:
        sol.steps.append(Step("Evaluar en el punto", "", L(val), "", f"Se sustituye {point}.", ""))
    sol.result = rf"D_{{\mathbf u}}f={L(val)}"
    sol.final = val  # type: ignore[attr-defined]
    return sol


def critical_points_solution(f, vars_) -> Solution:
    cp = calculus.critical_points(f, vars_)
    sol = Solution(topic="Cálculo diferencial · puntos críticos y clasificación", problem=rf"f={L(f)}",
                   goal="Hallar y clasificar los puntos donde ∇f = 0.", method="Gradiente nulo + criterio de la segunda derivada (Hessiana)",
                   theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    sol.steps.append(Step("Gradiente", "", L(sp.Matrix(cp["gradient"])), r"\nabla f=\mathbf 0", "Se calculan todas las derivadas parciales.", "En un extremo local interior las parciales valen 0."))
    sol.steps.append(Step("Resolver ∇f = 0", "", r",\;".join(f"{L(g)}=0" for g in cp["gradient"]), "", "Se resuelve el sistema (SymPy).", "Sus soluciones reales son los candidatos."))
    sol.steps.append(Step("Matriz Hessiana", "", L(cp["hessian"]), r"H_{ij}=\partial_{ij}f", "Se calcula la matriz de segundas derivadas.", "Informa la curvatura en cada punto crítico."))
    lines = []
    for p in cp["points"]:
        pt = L(sp.Matrix([p["point"][v] for v in vars_]).T)
        extra = ""
        if len(vars_) == 2:
            D = sp.simplify(p["hessian"].det())
            extra = rf",\;D=f_{{xx}}f_{{yy}}-f_{{xy}}^{{2}}={L(D)},\;f_{{xx}}={L(p['hessian'][0, 0])}"
        sol.steps.append(Step(f"Clasificar {pt}", L(p["hessian"]), rf"\text{{{p['kind']}}},\;f={L(p['value'])}" + extra,
                              r"D>0,\,f_{xx}>0\Rightarrow\text{mín};\;D>0,\,f_{xx}<0\Rightarrow\text{máx};\;D<0\Rightarrow\text{silla}" if len(vars_) == 2 else "",
                              "Se evalúa la Hessiana en el punto y se aplica el criterio de la segunda derivada.", "Distingue mínimos, máximos y puntos silla."))
        lines.append(rf"{pt}:\ \text{{{p['kind']}}},\ f={L(p['value'])}")
    sol.result = r"\\".join(lines) if lines else r"\text{No hay puntos críticos reales aislados}"
    if cp["skipped"]:
        sol.notes.append("Se omitieron familias de soluciones no aisladas (curvas de puntos críticos).")
    sol.final = cp  # type: ignore[attr-defined]
    return sol


def lagrange_solution(f, constraints, vars_) -> Solution:
    lg = calculus.lagrange(f, constraints, vars_)
    sol = Solution(topic="Optimización · multiplicadores de Lagrange", problem=rf"\text{{extremos de }} f={L(f)}\ \text{{ sujeto a }}\ " + r",\ ".join(f"{L(g)}=0" for g in constraints),
                   goal="Hallar los puntos candidatos a extremo con restricciones.", method="∇f = λ∇g junto con g = 0", theory=THEORY["vectorial"], errors=COMMON_ERRORS["vectorial"])
    sol.steps = [
        Step("Lagrangiano", "", rf"\mathcal L={L(lg['lagrangian'])}", r"\mathcal L=f-\lambda g", "Se incorpora la restricción con un multiplicador.", "En un extremo condicionado ∇f es paralelo a ∇g."),
        Step("Sistema de ecuaciones", "", r"\begin{cases}" + r"\\".join(f"{L(e)}=0" for e in lg["equations"]) + r"\end{cases}", r"\nabla\mathcal L=\mathbf 0,\; g=0",
             "Se anulan las derivadas parciales de ℒ y se exige la restricción.", "Son las condiciones necesarias de primer orden."),
    ]
    lines = []
    for s in lg["solutions"]:
        pt = L(sp.Matrix([s["point"][v] for v in vars_]).T)
        sol.steps.append(Step(f"Candidato {pt}", "", rf"f={L(s['value'])}", "", "Se evalúa f en la solución del sistema.", ""))
        lines.append(rf"{pt}:\ f={L(s['value'])}")
    if lg["solutions"]:
        vals = [s["value"] for s in lg["solutions"]]
        try:
            vf = [float(sp.N(v)) for v in vals]
            sol.steps.append(Step("Comparar valores", "", rf"f_{{\max}}={L(vals[vf.index(max(vf))])},\;f_{{\min}}={L(vals[vf.index(min(vf))])}", "",
                                  "Si la región es compacta, el mayor y el menor valor entre los candidatos son el máximo y el mínimo absolutos.", ""))
        except Exception:  # noqa: BLE001
            pass
    sol.result = r"\\".join(lines) if lines else r"\text{Sin soluciones reales}"
    sol.final = lg  # type: ignore[attr-defined]
    return sol
