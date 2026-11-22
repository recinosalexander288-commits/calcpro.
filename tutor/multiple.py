"""Integrales dobles/triples (cartesianas, polares, cilíndricas, esféricas) paso a paso."""
from __future__ import annotations

import sympy as sp

from core import calculus
from .model import COMMON_ERRORS, THEORY, Solution, Step, L

_SYSTEM_FORMULAS = {
    "polares": (r"x=r\cos\theta,\quad y=r\sin\theta", r"J=r", r"dA=r\,dr\,d\theta"),
    "cilíndricas": (r"x=r\cos\theta,\quad y=r\sin\theta,\quad z=z", r"J=r", r"dV=r\,dz\,dr\,d\theta"),
    "esféricas": (r"x=\rho\sin\varphi\cos\theta,\quad y=\rho\sin\varphi\sin\theta,\quad z=\rho\cos\varphi",
                  r"J=\rho^{2}\sin\varphi", r"dV=\rho^{2}\sin\varphi\,d\rho\,d\theta\,d\varphi"),
}


def _nest(limits, integrand) -> str:
    """LaTeX de la integral iterada. `limits` con la variable MÁS INTERNA primero."""
    out = ""
    for v, lo, hi in reversed(limits):
        out += rf"\int_{{{L(lo)}}}^{{{L(hi)}}}"
    inner = L(integrand) + "".join(rf"\,d{L(v)}" for v, _, _ in limits)
    return out + inner


def multiple_integral_solution(f, limits, system: str = "cartesianas", already_converted: bool = False) -> Solution:
    """limits: [(var, lo, hi), ...] con la variable MÁS INTERNA primero."""
    f = sp.sympify(f)
    dim = len(limits)
    kind = {2: "doble", 3: "triple"}.get(dim, "múltiple")
    sol = Solution(topic=f"Cálculo multivariable · integral {kind} ({system})",
                   problem=_nest(limits, f),
                   given=[f"Integrando: ${L(f)}$"] + [f"${L(v)}$ desde ${L(lo)}$ hasta ${L(hi)}$" for v, lo, hi in limits],
                   goal="Evaluar la integral iterada, de la integral más interna a la más externa.",
                   method="Teorema de Fubini (integral iterada)" + ("" if system == "cartesianas" else " con cambio de variables"),
                   theory=THEORY["multiple"], errors=COMMON_ERRORS["multiple"])
    sol.steps.append(Step("Región de integración y orden", "", _nest(limits, f), r"\iint_R f\,dA=\int_a^b\int_{g_1(x)}^{g_2(x)}f\,dy\,dx",
                          "Se integra primero respecto de " + ", luego de ".join(f"${L(v)}$" for v, _, _ in limits) + ".",
                          "Los límites de una integral interna pueden depender de las variables externas, pero nunca al revés."))

    g = f
    if system != "cartesianas":
        conv, jac_txt, d_txt = _SYSTEM_FORMULAS[system]
        sol.steps.append(Step("Fórmulas de conversión", "", conv, conv, f"Sistema de coordenadas: {system}.",
                              "Las regiones con simetría circular/esférica tienen límites constantes en estas coordenadas."))
        new_vars, subs, jac, _desc = calculus.coordinate_map(system)
        jmat = sp.Matrix([[sp.diff(e, v) for v in new_vars] for e in [subs.get(s, s) for s in sp.symbols("x y z")[:len(new_vars)]]])
        det = sp.simplify(jmat.det())
        sol.steps.append(Step("Jacobiano", L(jmat), rf"\left|J\right|={L(sp.simplify(sp.Abs(det)))}", r"J=\left|\frac{\partial(x,y,z)}{\partial(r,\theta,z)}\right|",
                              f"Se calcula el determinante de la matriz de derivadas parciales: {jac_txt}.",
                              "El Jacobiano compensa la deformación del elemento de área/volumen al cambiar de coordenadas."))
        sol.steps.append(Step("Elemento diferencial", "", d_txt, d_txt, "El elemento de área/volumen en las nuevas coordenadas.", ""))
        if not already_converted:
            g = calculus.to_coordinates(f, system)[0]
            sol.steps.append(Step("Expresar el integrando en las nuevas variables", L(f), L(g), "",
                                  "Se sustituyen x, y, z por sus expresiones y se simplifica con identidades trigonométricas.",
                                  "El integrando debe escribirse en las mismas variables que los límites."))
        g = sp.simplify(g * jac)
        sol.steps.append(Step("Multiplicar por el Jacobiano", "", _nest(limits, g), "", "Integrando final = f · J.", ""))

    current = g
    remaining = list(limits)
    exact_ok = True
    for k, (v, lo, hi) in enumerate(limits):
        try:
            anti = sp.integrate(current, v)
            if anti.has(sp.Integral):
                raise ValueError
            evaluated = sp.simplify(anti.subs(v, hi) - anti.subs(v, lo))
            sol.steps.append(Step(f"Integrar respecto de ${L(v)}$ (integral {'más interna' if k == 0 else f'nº {k + 1}'})",
                                  rf"\int_{{{L(lo)}}}^{{{L(hi)}}}{L(current)}\,d{L(v)}",
                                  rf"\left[{L(anti)}\right]_{{{L(lo)}}}^{{{L(hi)}}}={L(evaluated)}",
                                  r"\int_a^b f\,dv=F(b)-F(a)",
                                  f"Las demás variables se tratan como constantes. Antiderivada: ${L(anti)}$; luego se evalúa en los límites.",
                                  "Cada integración elimina una variable hasta quedar un número."))
            current = evaluated
        except Exception:  # noqa: BLE001
            exact_ok = False
            sol.steps.append(Step(f"Integrar respecto de ${L(v)}$", rf"\int {L(current)}\,d{L(v)}", "", "",
                                  "No se encontró antiderivada elemental en este paso: se pasa a integración numérica.", ""))
            break
        remaining = remaining[1:]

    if exact_ok:
        sol.result = rf"{_nest(limits, f)}={L(current)}"
        sol.final = current  # type: ignore[attr-defined]
        try:
            num, err = calculus.numeric_integral_nd(g, limits)
            ok = abs(num - float(sp.N(current))) <= 1e-6 * (1 + abs(num))
            sol.verification = (f"✔ Verificación numérica independiente (nquad): {num:.10g} ≈ {float(sp.N(current)):.10g}." if ok else
                                f"⚠ nquad da {num:.10g}, distinto del valor simbólico {float(sp.N(current)):.10g}.")
        except Exception:  # noqa: BLE001
            sol.verification = "La verificación numérica no fue posible."
    else:
        res = calculus.iterated(g, limits)
        sol.exact = False
        if res["numeric"] is not None:
            val, err = res["numeric"]
            sol.steps.append(Step("Aproximación numérica", "", rf"\approx {val:.12g}", "", res["note"], ""))
            sol.result = rf"{_nest(limits, f)}\approx {val:.12g}"
            sol.verification = "Resultado APROXIMADO (no exacto)."
        else:
            sol.result = r"\text{No se pudo evaluar}"
            sol.verification = res["note"]
    return sol
