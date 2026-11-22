"""Límites paso a paso. Cada método se acepta sólo si reproduce el valor verificado."""
from __future__ import annotations

import sympy as sp

from .model import COMMON_ERRORS, THEORY, Solution, Step, L, equivalent


def _finite(v) -> bool:
    return v is not None and not v.has(sp.nan, sp.zoo, sp.oo, -sp.oo)


def _is_inf(v) -> bool:
    return v in (sp.oo, -sp.oo)


def _form(f, x, a, d) -> str:
    """Clasifica la forma indeterminada (cadena vacía si no la hay)."""
    try:
        num, den = sp.fraction(sp.together(f))
        if den != 1:
            ln, ld = sp.limit(num, x, a, dir=d), sp.limit(den, x, a, dir=d)
            if ln == 0 and ld == 0:
                return "0/0"
            if _is_inf(ln) and _is_inf(ld):
                return "∞/∞"
        if f.is_Add:
            ls = [sp.limit(t, x, a, dir=d) for t in f.args]
            if any(v == sp.oo for v in ls) and any(v == -sp.oo for v in ls):
                return "∞−∞"
        if f.is_Mul:
            ls = [sp.limit(t, x, a, dir=d) for t in f.args]
            if any(v == 0 for v in ls) and any(_is_inf(v) for v in ls):
                return "0·∞"
        if f.is_Pow:
            b, e = f.as_base_exp()
            lb, le = sp.limit(b, x, a, dir=d), sp.limit(e, x, a, dir=d)
            if lb == 1 and _is_inf(le):
                return "1^∞"
            if lb == 0 and le == 0:
                return "0^0"
            if _is_inf(lb) and le == 0:
                return "∞^0"
    except Exception:  # noqa: BLE001
        pass
    return ""


def limit_solution(f, x, a, direction: str = "+-", prefer: str = "auto") -> Solution:
    f, a = sp.sympify(f), sp.sympify(a)
    dtxt = {"+-": "", "+": "^{+}", "-": "^{-}"}[direction]
    problem = rf"\lim_{{{L(x)}\to {L(a)}{dtxt}}} {L(f)}"
    sol = Solution(topic="Límites", problem=problem,
                   given=[f"Función: $f({L(x)})={L(f)}$", f"Punto: ${L(x)}\\to {L(a)}$" + (" (lateral)" if direction != "+-" else "")],
                   goal="Determinar el valor al que se aproxima f(x) (o si diverge / no existe).",
                   theory=THEORY["limite"], errors=COMMON_ERRORS["limite"])
    true = sp.limit(f, x, a, dir=direction)
    steps: list[Step] = []

    # 1) sustitución directa
    direct = None
    if not a.has(sp.oo, -sp.oo):
        try:
            direct = f.subs(x, a)
        except Exception:  # noqa: BLE001
            direct = None
    if direct is not None and _finite(direct) and equivalent(direct, true):
        steps.append(Step("Sustitución directa", problem, L(sp.simplify(direct)), r"\lim_{x\to a}f(x)=f(a)\;\text{si } f \text{ es continua en } a",
                          f"Se reemplaza ${L(x)}={L(a)}$ y el resultado es un número finito.",
                          "Si la función es continua en el punto, el límite coincide con el valor de la función."))
        sol.method = "Sustitución directa"
        return _finish(sol, steps, true, f, x, a, direction)

    form = _form(f, x, a, direction)
    if form:
        form_tex = form.replace("∞", r"\infty ").replace("−", "-").replace("·", r"\cdot ")
        steps.append(Step("Identificar la indeterminación", problem, r"\text{forma indeterminada: }\;" + form_tex,
                          "", "Al sustituir directamente (o al evaluar los límites de las partes) se obtiene una forma indeterminada.",
                          "Una indeterminación no es un resultado: hay que transformar la expresión."))
    else:
        steps.append(Step("Sustitución directa", problem, "", "",
                          "La sustitución directa no da un valor finito." if direct is None or not _finite(direct) else "La sustitución no es concluyente.",
                          "Se analiza el comportamiento cerca del punto."))

    def accepted(expr_val) -> bool:
        return expr_val is not None and (equivalent(expr_val, true) if _finite(true) and _finite(expr_val) else expr_val == true)

    methods = []
    if prefer in ("auto", "algebraico"):
        methods += ["cancel", "radsimp"]
    if prefer in ("auto", "lhopital"):
        methods += ["lhopital"]
    if form in ("1^∞", "0^0", "∞^0"):
        methods = ["log"] + methods
    if form in ("0·∞", "∞−∞"):
        methods = ["together"] + methods

    for m in methods:
        n = len(steps)
        try:
            if m == "cancel":
                g = sp.cancel(sp.together(f))
                if g == f:
                    continue
                nu, de = sp.fraction(sp.together(f))
                steps.append(Step("Factorizar y cancelar", L(f), rf"\frac{{{L(sp.factor(nu))}}}{{{L(sp.factor(de))}}}",
                                  r"a^{2}-b^{2}=(a-b)(a+b)\;;\;\text{factor común}",
                                  "Se factorizan numerador y denominador para localizar el factor que causa la indeterminación.",
                                  "El factor común que se anula en el punto puede cancelarse porque x ≠ a mientras x → a."))
                steps.append(Step("Cancelar el factor común", "", L(g), "", "Se simplifica la fracción.",
                                  "La función simplificada coincide con la original salvo en el punto."))
                val = g.subs(x, a) if not a.has(sp.oo, -sp.oo) else sp.limit(g, x, a, dir=direction)
                if accepted(val):
                    steps.append(Step("Sustituir en la expresión simplificada", L(g), L(sp.simplify(val)), "",
                                      f"Se evalúa en ${L(x)}={L(a)}$.", "Ya no hay indeterminación."))
                    sol.method = "Factorización y cancelación"
                    return _finish(sol, steps, true, f, x, a, direction)
            elif m == "radsimp":
                if not f.has(sp.sqrt) and not any(p.is_Pow and not p.exp.is_Integer for p in f.atoms(sp.Pow)):
                    continue
                g = sp.radsimp(f)
                if g == f:
                    continue
                steps.append(Step("Racionalizar (multiplicar por el conjugado)", L(f), L(g),
                                  r"(a-b)(a+b)=a^{2}-b^{2}",
                                  "Se multiplica numerador y denominador por el conjugado de la expresión con raíz.",
                                  "Así desaparecen las raíces y aparece un factor que puede cancelarse."))
                g2 = sp.cancel(sp.together(g))
                steps.append(Step("Simplificar", L(g), L(g2), "", "Se desarrolla y se cancela el factor común.", ""))
                val = g2.subs(x, a) if not a.has(sp.oo, -sp.oo) else sp.limit(g2, x, a, dir=direction)
                if accepted(val):
                    steps.append(Step("Sustituir", L(g2), L(sp.simplify(val)), "", "Se evalúa el límite.", ""))
                    sol.method = "Racionalización"
                    return _finish(sol, steps, true, f, x, a, direction)
            elif m == "together":
                g = sp.together(f)
                if g == f:
                    continue
                steps.append(Step("Reescribir como una sola fracción", L(f), L(g), "",
                                  "Se combinan los términos en un cociente para poder aplicar los métodos de 0/0 o ∞/∞.",
                                  "Las formas ∞−∞ y 0·∞ se convierten en cocientes."))
                sub = limit_solution(g, x, a, direction, prefer)
                if accepted(sp.sympify(sol_val(sub))):
                    steps += sub.steps
                    sol.method = "Reescritura + " + sub.method
                    return _finish(sol, steps, true, f, x, a, direction)
            elif m == "lhopital" and form in ("0/0", "∞/∞"):
                nu, de = sp.fraction(sp.together(f))
                cn, cd = nu, de
                for k in range(1, 4):
                    dn, dd = sp.diff(cn, x), sp.diff(cd, x)
                    steps.append(Step(f"Regla de L'Hôpital (aplicación {k})", rf"\frac{{{L(cn)}}}{{{L(cd)}}}",
                                      rf"\frac{{{L(dn)}}}{{{L(dd)}}}",
                                      r"\lim\frac{f}{g}=\lim\frac{f'}{g'}\;\text{si la forma es }\tfrac00\text{ o }\tfrac{\infty}{\infty}",
                                      "Se deriva el numerador y el denominador POR SEPARADO.",
                                      "Con forma 0/0 o ∞/∞ y si el nuevo límite existe, ambos límites coinciden."))
                    val = sp.limit(dn / dd, x, a, dir=direction)
                    if accepted(val) and (_finite(val) or _is_inf(val)):
                        steps.append(Step("Evaluar el nuevo límite", "", L(val), "", "El límite del cociente de derivadas existe.", ""))
                        sol.method = "Regla de L'Hôpital"
                        return _finish(sol, steps, true, f, x, a, direction)
                    cn, cd = dn, dd
            elif m == "log" and f.is_Pow:
                b, e = f.as_base_exp()
                g = e * sp.log(b)
                steps.append(Step("Tomar logaritmo natural", L(f), rf"\ln L=\lim {L(g)}", r"L=e^{\lim g\ln f}",
                                  "Se escribe f^g = e^{g ln f} y se calcula el límite del exponente.",
                                  "Las formas 1^∞, 0^0, ∞^0 se convierten en 0·∞."))
                sub = limit_solution(g, x, a, direction, prefer)
                subv = sp.sympify(sol_val(sub))
                if accepted(sp.exp(subv)):
                    steps += sub.steps
                    steps.append(Step("Deshacer el logaritmo", rf"\ln L={L(subv)}", rf"L=e^{{{L(subv)}}}={L(sp.exp(subv))}", "",
                                      "Se aplica la exponencial a ambos lados.", ""))
                    sol.method = "Logaritmo natural"
                    return _finish(sol, steps, true, f, x, a, direction)
        except Exception:  # noqa: BLE001
            pass
        del steps[n:]

    steps.append(Step("Resultado por el algoritmo de límites de SymPy", problem, L(true), "",
                      "Ningún método algebraico programado reprodujo el valor verificado; el límite lo calcula SymPy (series de Gruntz).",
                      "Se declara para no inventar un procedimiento que no corresponde."))
    sol.method = "Algoritmo general de SymPy"
    sol.notes.append("El desglose didáctico no estuvo disponible para este límite.")
    return _finish(sol, steps, true, f, x, a, direction)


def sol_val(sub: Solution):
    return getattr(sub, "final", sp.nan)


def _finish(sol: Solution, steps, true, f, x, a, direction) -> Solution:
    sol.steps = steps
    sol.final = true  # type: ignore[attr-defined]
    dtxt = {"+-": "", "+": "^{+}", "-": "^{-}"}[direction]
    sol.result = rf"\lim_{{{L(x)}\to {L(a)}{dtxt}}} {L(f)}={L(true)}"
    if direction == "+-":
        try:
            lp, lm = sp.limit(f, x, a, dir="+"), sp.limit(f, x, a, dir="-")
            sol.verification = (f"✔ Límites laterales: por la derecha ${L(lp)}$, por la izquierda ${L(lm)}$ → "
                                + ("coinciden, el límite es válido." if lp == lm else "NO coinciden: el límite bilateral no existe."))
        except Exception:  # noqa: BLE001
            sol.verification = "Resultado verificado con el algoritmo de SymPy."
    else:
        sol.verification = "Resultado verificado con el algoritmo de SymPy."
    return sol
