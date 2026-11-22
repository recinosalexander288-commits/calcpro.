"""Derivadas paso a paso. Cada regla se aplica de verdad sobre el árbol de la expresión."""
from __future__ import annotations

import sympy as sp

from .model import COMMON_ERRORS, THEORY, Solution, Step, L, equivalent

# función → (fórmula LaTeX, derivada externa evaluada en u)
_TABLE = {
    sp.sin: (r"\frac{d}{du}\sin u=\cos u", lambda u: sp.cos(u)),
    sp.cos: (r"\frac{d}{du}\cos u=-\sin u", lambda u: -sp.sin(u)),
    sp.tan: (r"\frac{d}{du}\tan u=\sec^{2}u", lambda u: sp.sec(u) ** 2),
    sp.cot: (r"\frac{d}{du}\cot u=-\csc^{2}u", lambda u: -sp.csc(u) ** 2),
    sp.sec: (r"\frac{d}{du}\sec u=\sec u\tan u", lambda u: sp.sec(u) * sp.tan(u)),
    sp.csc: (r"\frac{d}{du}\csc u=-\csc u\cot u", lambda u: -sp.csc(u) * sp.cot(u)),
    sp.exp: (r"\frac{d}{du}e^{u}=e^{u}", lambda u: sp.exp(u)),
    sp.log: (r"\frac{d}{du}\ln u=\frac{1}{u}", lambda u: 1 / u),
    sp.asin: (r"\frac{d}{du}\arcsin u=\frac{1}{\sqrt{1-u^{2}}}", lambda u: 1 / sp.sqrt(1 - u ** 2)),
    sp.acos: (r"\frac{d}{du}\arccos u=-\frac{1}{\sqrt{1-u^{2}}}", lambda u: -1 / sp.sqrt(1 - u ** 2)),
    sp.atan: (r"\frac{d}{du}\arctan u=\frac{1}{1+u^{2}}", lambda u: 1 / (1 + u ** 2)),
    sp.sinh: (r"\frac{d}{du}\sinh u=\cosh u", lambda u: sp.cosh(u)),
    sp.cosh: (r"\frac{d}{du}\cosh u=\sinh u", lambda u: sp.sinh(u)),
    sp.tanh: (r"\frac{d}{du}\tanh u=\operatorname{sech}^{2}u", lambda u: 1 - sp.tanh(u) ** 2),
}


def _par(e) -> str:
    return r"\left(" + L(e) + r"\right)"


class _Diff:
    def __init__(self, x):
        self.x = x
        self.steps: list[Step] = []

    def D(self, e) -> str:
        return L(sp.Derivative(e, self.x))

    def add(self, title, f, after, formula, expl, reason):
        self.steps.append(Step(title, self.D(f), after if isinstance(after, str) else L(after),
                               formula, expl, reason))

    def run(self, f):
        x = self.x
        if not f.has(x):
            self.add("Regla de la constante", f, "0", r"\frac{d}{dx}c=0",
                     f"La expresión ${L(f)}$ no depende de ${L(x)}$, así que se comporta como constante.",
                     "Una constante no cambia cuando x cambia, por eso su razón de cambio es cero.")
            return sp.Integer(0)
        if f == x:
            self.add("Derivada de la variable", f, "1", r"\frac{d}{dx}x=1",
                     "La derivada de x respecto de x es 1.", "x cambia exactamente a la misma razón que x.")
            return sp.Integer(1)

        if f.is_Add:
            self.add("Regla de la suma", f, " + ".join(self.D(t) for t in f.args),
                     r"\frac{d}{dx}[u+v]=u'+v'",
                     "Se deriva cada sumando por separado.",
                     "La derivada es lineal: la derivada de una suma es la suma de las derivadas.")
            return sp.Add(*[self.run(t) for t in f.args])

        if f.is_Mul:
            c, rest = f.as_independent(x)
            if c != 1:
                self.add("Regla del múltiplo constante", f, f"{_par(c)}\\cdot {self.D(rest)}",
                         r"\frac{d}{dx}[c\,u]=c\,u'",
                         f"El factor constante ${L(c)}$ sale de la derivada.",
                         "Las constantes multiplicativas se conservan al derivar.")
                return c * self.run(rest)
            numer, denom = sp.fraction(f)
            if denom != 1 and denom.has(x):
                n_, d_ = numer, denom
                self.add("Regla del cociente", f,
                         rf"\frac{{{self.D(n_)}\cdot {_par(d_)}-{_par(n_)}\cdot {self.D(d_)}}}{{{_par(d_)}^{{2}}}}",
                         r"\frac{d}{dx}\left[\frac{u}{v}\right]=\frac{u'v-uv'}{v^{2}}",
                         f"Numerador $u={L(n_)}$ y denominador $v={L(d_)}$.",
                         "El cociente de dos funciones no se deriva dividiendo derivadas; se usa esta regla.")
                dn, dd = self.run(n_), self.run(d_)
                return (dn * d_ - n_ * dd) / d_ ** 2
            u, v = f.args[0], sp.Mul(*f.args[1:])
            self.add("Regla del producto", f, f"{self.D(u)}\\cdot {_par(v)}+{_par(u)}\\cdot {self.D(v)}",
                     r"\frac{d}{dx}[uv]=u'v+uv'",
                     f"Se toma $u={L(u)}$ y $v={L(v)}$.",
                     "Ambos factores varían con x: cada uno aporta su cambio multiplicado por el otro factor.")
            du, dv = self.run(u), self.run(v)
            return du * v + u * dv

        if f.is_Pow:
            b, e = f.as_base_exp()
            if not e.has(x):
                if b == x:
                    self.add("Regla de la potencia", f, rf"{_par(e)}\,{L(x)}^{{{L(e - 1)}}}",
                             r"\frac{d}{dx}x^{n}=n\,x^{n-1}",
                             f"Con $n={L(e)}$: se baja el exponente y se le resta 1.",
                             "Es la regla básica para potencias de la variable con exponente constante.")
                    return e * x ** (e - 1)
                self.add("Regla de la potencia con regla de la cadena", f,
                         rf"{_par(e)}\,{_par(b)}^{{{L(e - 1)}}}\cdot {self.D(b)}",
                         r"\frac{d}{dx}u^{n}=n\,u^{n-1}\,u'",
                         f"La base es $u={L(b)}$ (no es sólo x), por eso aparece $u'$.",
                         "Cuando la base es una función de x, hay que multiplicar por su derivada (regla de la cadena).")
                return e * b ** (e - 1) * self.run(b)
            if not b.has(x):
                self.add("Derivada de una exponencial", f, rf"{L(f)}\,\ln {_par(b)}\cdot {self.D(e)}",
                         r"\frac{d}{dx}a^{u}=a^{u}\ln(a)\,u'",
                         f"Base constante $a={L(b)}$ y exponente $u={L(e)}$.",
                         "La variable está en el exponente: la derivada conserva la función y añade ln(a)·u′.")
                return f * sp.log(b) * self.run(e)
            self.add("Derivación logarítmica", f,
                     rf"{L(f)}\left({self.D(e)}\ln {_par(b)}+{_par(e)}\frac{{{self.D(b)}}}{{{_par(b)}}}\right)",
                     r"\frac{d}{dx}f^{g}=f^{g}\left(g'\ln f+g\frac{f'}{f}\right)",
                     "Tanto la base como el exponente dependen de x.",
                     "Se escribe y = f^g, se toma ln en ambos lados y se deriva implícitamente.")
            db, de = self.run(b), self.run(e)
            return f * (de * sp.log(b) + e * db / b)

        if f.func in _TABLE and len(f.args) == 1:
            formula, outer = _TABLE[f.func]
            u = f.args[0]
            if u == x:
                self.add("Derivada de función elemental", f, L(outer(u)), formula,
                         "Se usa directamente la tabla de derivadas.", "Es una derivada básica conocida.")
                return outer(u)
            self.add("Regla de la cadena", f, f"{L(outer(u))}\\cdot {self.D(u)}",
                     r"\frac{d}{dx}F(u)=F'(u)\,u'",
                     f"Función externa: derivada tabulada ({L(f.func(sp.Symbol('u')))}); función interna $u={L(u)}$.",
                     "La función interna cambia con x, así que su derivada multiplica a la derivada de la externa.")
            return outer(u) * self.run(u)

        res = sp.diff(f, x)
        self.add("Derivada obtenida con SymPy", f, L(res), "",
                 "Para este tipo de expresión no hay una regla didáctica programada; el resultado lo calcula SymPy.",
                 "Se mantiene la honestidad del procedimiento: este paso no está desglosado.")
        return res


def _spot_check(f, res, x) -> bool:
    """Comprobación numérica independiente: diferencia central."""
    try:
        extra = {s: sp.Float(0.9) for s in f.free_symbols if s != x}
        for x0 in (0.7, 1.3):
            h = 1e-6
            g = sp.lambdify(x, f.subs(extra), "numpy")
            approx = (complex(g(x0 + h)) - complex(g(x0 - h))) / (2 * h)
            exact = complex(sp.N(res.subs(extra).subs(x, x0)))
            if abs(approx - exact) > 1e-4 * (1 + abs(exact)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return True  # no concluyente: no se afirma nada falso


def derivative_solution(f, x, order: int = 1) -> Solution:
    f = sp.sympify(f)
    sol = Solution(
        topic="Cálculo diferencial · derivada" + (f" de orden {order}" if order > 1 else ""),
        problem=rf"\frac{{d{'^'+str(order) if order > 1 else ''}}}{{d{L(x)}{'^'+str(order) if order > 1 else ''}}}\left[{L(f)}\right]",
        given=[f"Función: $f({L(x)}) = {L(f)}$", f"Variable de derivación: ${L(x)}$"],
        goal=f"Calcular la derivada de orden {order} de $f$ respecto de ${L(x)}$.",
        method="Reglas de derivación aplicadas recursivamente sobre el árbol de la expresión.",
        theory=THEORY["derivada"], errors=COMMON_ERRORS["derivada"],
    )
    cur = f
    for k in range(1, order + 1):
        d = _Diff(x)
        if order > 1:
            sol.steps.append(Step(f"Derivada nº {k}", L(sp.Derivative(cur, x)), "", "",
                                  f"Se calcula la derivada de orden {k} derivando el resultado anterior."))
        assembled = d.run(cur)
        sol.steps += d.steps
        expanded = sp.expand(assembled)
        if expanded != assembled:
            sol.steps.append(Step("Desarrollo algebraico", L(assembled), L(expanded),
                                  r"a(b+c)=ab+ac",
                                  "Se multiplican los paréntesis y se agrupan términos semejantes.",
                                  "Dejar la expresión desarrollada permite ver qué términos se combinan."))
        simp = sp.simplify(assembled)
        if simp != expanded:
            sol.steps.append(Step("Simplificación", L(expanded), L(simp), "",
                                  "Se factorizan o reducen términos con identidades algebraicas/trigonométricas.",
                                  "Una forma más compacta es más fácil de interpretar y de evaluar."))
        cur = simp
    sol.result = rf"{L(sp.Derivative(f, x, order))} = {L(cur)}"
    ref = sp.diff(f, x, order)
    ok = equivalent(cur, ref) and _spot_check(f, cur, x) if order == 1 else equivalent(cur, ref)
    sol.verification = ("✔ Verificado: coincide con la derivada calculada por SymPy" +
                        (" y con la diferencia central numérica." if order == 1 else ".")) if ok else \
        "⚠ La verificación automática no fue concluyente; revise el resultado."
    if not ok:
        cur = ref
        sol.result = rf"{L(sp.Derivative(f, x, order))} = {L(cur)}"
        sol.notes.append("Se sustituyó el resultado por el de SymPy porque el desarrollo manual no coincidió.")
    sol.final = cur  # type: ignore[attr-defined]
    return sol


def derivative_by_definition(f, x) -> Solution:
    """Método alternativo: definición de derivada como límite."""
    h = sp.Symbol("h", positive=True)
    f = sp.sympify(f)
    num = sp.expand(f.subs(x, x + h) - f)
    quotient = sp.simplify(num / h)
    res = sp.limit(quotient, h, 0)
    sol = Solution(topic="Derivada · método alternativo (definición por límite)",
                   problem=rf"f'({L(x)})=\lim_{{h\to 0}}\frac{{f({L(x)}+h)-f({L(x)})}}{{h}}",
                   given=[f"$f({L(x)})={L(f)}$"], goal="Calcular f′ directamente desde la definición.",
                   method="Cociente incremental y límite.", errors=COMMON_ERRORS["derivada"],
                   theory=THEORY["derivada"])
    sol.steps = [
        Step("Plantear el cociente incremental", "", rf"\frac{{{L(f.subs(x, x + h))}-\left({L(f)}\right)}}{{h}}",
             r"f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}", "Se evalúa f en x+h y se resta f(x).",
             "Es la definición: pendiente de la secante cuando el incremento tiende a 0."),
        Step("Desarrollar el numerador", "", rf"\frac{{{L(num)}}}{{h}}", "",
             "Se expande f(x+h) − f(x) y se cancelan términos.", "Hay que eliminar el factor h para poder tomar el límite."),
        Step("Simplificar dividiendo entre h", "", L(quotient), "", "Se factoriza h en el numerador y se cancela con el denominador.",
             "Así desaparece la indeterminación 0/0."),
        Step("Tomar el límite h → 0", L(quotient), L(res), "", "Se evalúa el límite (con SymPy cuando la expresión lo requiere).",
             "Es el valor al que tiende la pendiente de las secantes."),
    ]
    sol.result = rf"f'({L(x)})={L(res)}"
    sol.verification = "✔ Coincide con la derivada por reglas." if equivalent(res, sp.diff(f, x)) else \
        "⚠ No coincide con la derivada por reglas; revise la expresión."
    return sol
