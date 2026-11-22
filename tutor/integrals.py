"""Integrales paso a paso: tabla, linealidad, sustitución, partes y fracciones parciales.

Cada intento se VERIFICA derivando el resultado; si falla, se descartan sus pasos y se
prueba otro método. Si ningún método didáctico funciona, se usa el algoritmo general de
SymPy y se declara explícitamente que ese tramo no está desglosado.
"""
from __future__ import annotations

import sympy as sp

from core import calculus
from .model import COMMON_ERRORS, THEORY, Solution, Step, L, equivalent, fresh_symbol


class _GiveUp(Exception):
    pass


def _par(e) -> str:
    return r"\left(" + L(e) + r"\right)"


class _Integ:
    def __init__(self, x, prefer: str = "auto"):
        self.x = x
        self.prefer = prefer
        self.steps: list[Step] = []
        self.used_fallback = False
        self.methods: list[str] = []

    def I(self, e, v=None) -> str:
        return L(sp.Integral(e, v or self.x))

    def add(self, title, before, after, formula, expl, reason):
        self.steps.append(Step(title, before, after if isinstance(after, str) else L(after), formula, expl, reason))

    # ── tabla ──
    def _table(self, f, x):
        a = sp.Wild("a", exclude=[x])
        if f.is_Pow:
            b, e = f.as_base_exp()
            if b == x and not e.has(x):
                if e == -1:
                    return sp.log(sp.Abs(x)), r"\int \frac{1}{x}dx=\ln|x|+C", "potencia con n = −1"
                return x ** (e + 1) / (e + 1), r"\int x^{n}dx=\frac{x^{n+1}}{n+1}+C\;(n\neq-1)", "regla de la potencia"
            if not b.has(x) and e == x and b != sp.E:
                return f / sp.log(b), r"\int a^{x}dx=\frac{a^{x}}{\ln a}+C", "exponencial de base a"
            if b == sp.sec(x) and e == 2:
                return sp.tan(x), r"\int \sec^{2}x\,dx=\tan x+C", "tabla"
            if b == sp.csc(x) and e == 2:
                return -sp.cot(x), r"\int \csc^{2}x\,dx=-\cot x+C", "tabla"
            if e == -1 and sp.simplify(b - (1 + x ** 2)) == 0:
                return sp.atan(x), r"\int\frac{dx}{1+x^{2}}=\arctan x+C", "tabla"
            if e == -sp.Rational(1, 2) and sp.simplify(b - (1 - x ** 2)) == 0:
                return sp.asin(x), r"\int\frac{dx}{\sqrt{1-x^{2}}}=\arcsin x+C", "tabla"
        if len(f.args) == 1 and f.args[0] == x:
            tab = {
                sp.exp: (sp.exp(x), r"\int e^{x}dx=e^{x}+C"),
                sp.sin: (-sp.cos(x), r"\int \sin x\,dx=-\cos x+C"),
                sp.cos: (sp.sin(x), r"\int \cos x\,dx=\sin x+C"),
                sp.sinh: (sp.cosh(x), r"\int \sinh x\,dx=\cosh x+C"),
                sp.cosh: (sp.sinh(x), r"\int \cosh x\,dx=\sinh x+C"),
            }
            if f.func in tab:
                return tab[f.func][0], tab[f.func][1], "tabla"
        if sp.simplify(f - sp.sec(x) * sp.tan(x)) == 0:
            return sp.sec(x), r"\int \sec x\tan x\,dx=\sec x+C", "tabla"
        if sp.simplify(f - sp.csc(x) * sp.cot(x)) == 0:
            return -sp.csc(x), r"\int \csc x\cot x\,dx=-\csc x+C", "tabla"
        return None

    def _valid(self, res, f, x) -> bool:
        try:
            return equivalent(sp.diff(res, x), f)
        except Exception:  # noqa: BLE001
            return False

    def _attempt(self, fn, f, x, depth):
        n = len(self.steps)
        m = len(self.methods)
        try:
            r = fn(f, x, depth)
        except _GiveUp:
            r = None
        if r is not None and self._valid(r, f, x):
            return r
        del self.steps[n:]
        del self.methods[m:]
        return None

    # ── sustitución ──
    def _usub(self, f, x, depth):
        if depth >= 4:
            return None
        cands, seen = [], set()
        for sub in sp.preorder_traversal(f):
            if sub == f or sub == x or not sub.has(x) or sub.is_Number or sub in seen:
                continue
            if sub.is_Mul and not sub.is_Add:
                continue
            seen.add(sub)
            cands.append(sub)
        cands.sort(key=lambda e: sp.count_ops(e))
        for u in cands[:8]:
            du = sp.diff(u, x)
            if du == 0:
                continue
            U = fresh_symbol(f, u)
            ratio = sp.simplify(f / du)
            g = ratio.subs(u, U)
            if g.has(x):
                sols = sp.solve(sp.Eq(u, U), x)
                if len(sols) != 1:
                    continue
                g = sp.simplify(ratio.subs(x, sols[0]))
            if g.has(x):
                continue
            n, m = len(self.steps), len(self.methods)
            self.methods.append("sustitución")
            self.add(f"Sustitución  {L(U)} = {L(u)}", self.I(f),
                     rf"\int {L(g)}\,d{L(U)}",
                     rf"\int f(g(x))\,g'(x)\,dx=\int f({L(U)})\,d{L(U)}",
                     f"Se elige ${L(U)}={L(u)}$; entonces $d{L(U)}={L(du)}\\,dx$. Al reescribir, la variable x desaparece.",
                     "Se escoge como u la parte interna cuya derivada aparece (salvo constantes) en el integrando.")
            try:
                inner = self.solve(g, U, depth + 1)
            except _GiveUp:
                del self.steps[n:], self.methods[m:]
                continue
            res = inner.subs(U, u)
            if self._valid(res, f, x):
                self.add("Regresar a la variable original", L(inner), L(res), "",
                         f"Se reemplaza ${L(U)}={L(u)}$.", "La respuesta debe expresarse en la variable del problema.")
                return res
            del self.steps[n:], self.methods[m:]
        return None

    # ── partes ──
    @staticmethod
    def _liate(g, x) -> int:
        if g.has(sp.log):
            return 1
        if g.has(sp.asin) or g.has(sp.acos) or g.has(sp.atan):
            return 2
        if g.is_polynomial(x) or (g.is_Pow and g.base == x):
            return 3
        if g.has(sp.sin) or g.has(sp.cos) or g.has(sp.tan) or g.has(sp.sinh) or g.has(sp.cosh):
            return 4
        if g.has(sp.exp):
            return 5
        return 6

    def _parts(self, f, x, depth):
        if depth >= 3:
            return None
        fs = list(f.args) if f.is_Mul else [f]
        ranks = [self._liate(g, x) for g in fs]
        i = min(range(len(fs)), key=lambda k: ranks[k])
        if len(fs) == 1 and ranks[i] > 2:
            return None
        u = fs[i]
        dv = sp.Mul(*(fs[:i] + fs[i + 1:])) if len(fs) > 1 else sp.Integer(1)
        if sp.count_ops(dv) > 12:
            return None
        v = sp.integrate(dv, x)
        if v.has(sp.Integral):
            return None
        du = sp.diff(u, x)
        remaining = sp.simplify(v * du)
        if sp.count_ops(remaining) > sp.count_ops(f) + 6:
            return None
        self.methods.append("integración por partes")
        self.add("Integración por partes", self.I(f),
                 f"{_par(u)}\\cdot {_par(v)}-\\int {L(remaining)}\\,d{L(x)}",
                 r"\int u\,dv=uv-\int v\,du",
                 f"Con LIATE: $u={L(u)}$, $dv={L(dv)}\\,d{L(x)}$. Entonces $du={L(du)}\\,d{L(x)}$ y $v=\\int dv={L(v)}$.",
                 "Se elige u de modo que al derivarla se simplifique y dv sea fácil de integrar.")
        inner = self.solve(remaining, x, depth + 1)
        return u * v - inner

    # ── núcleo ──
    def solve(self, f, x, depth: int = 0):
        f = sp.sympify(f)
        if depth > 6:
            raise _GiveUp
        if not f.has(x):
            r = f * x
            self.add("Integral de una constante", self.I(f, x), L(r), r"\int c\,dx=c\,x",
                     "El integrando no depende de la variable de integración.", "Se integra una constante multiplicándola por x.")
            self.methods.append("tabla")
            return r

        if f.is_Add:
            self.add("Linealidad de la integral", self.I(f, x), " + ".join(self.I(t, x) for t in f.args),
                     r"\int[u+v]\,dx=\int u\,dx+\int v\,dx",
                     "Se integra cada sumando por separado.", "La integral de una suma es la suma de las integrales.")
            self.methods.append("linealidad")
            return sp.Add(*[self._solve_or_fallback(t, x, depth + 1) for t in f.args])

        c, rest = f.as_independent(x)
        if c != 1:
            self.add("Constante fuera de la integral", self.I(f, x), f"{_par(c)}\\cdot {self.I(rest, x)}",
                     r"\int c\,u\,dx=c\int u\,dx", f"El factor ${L(c)}$ no depende de ${L(x)}$.",
                     "Las constantes multiplicativas pueden sacarse de la integral.")
            return c * self._solve_or_fallback(rest, x, depth + 1)

        t = self._table(f, x)
        if t is not None:
            r, formula, name = t
            self.methods.append("tabla")
            self.add("Integral inmediata (tabla)", self.I(f, x), L(r), formula,
                     f"Coincide con una integral de tabla ({name}).", "Es una antiderivada básica conocida.")
            return r

        order = [self._usub, self._parts] if self.prefer != "partes" else [self._parts, self._usub]
        if self.prefer == "sustitución":
            order = [self._usub]
        for fn in order:
            r = self._attempt(fn, f, x, depth)
            if r is not None:
                return r

        if f.is_rational_function(x):
            ap = sp.apart(f, x)
            if ap != f and ap.is_Add:
                n, m = len(self.steps), len(self.methods)
                self.methods.append("fracciones parciales")
                self.add("Fracciones parciales", self.I(f, x), L(ap), r"\frac{P(x)}{Q(x)}=\sum\frac{A_i}{(x-r_i)^{k}}",
                         "El denominador se factoriza y la fracción se descompone en fracciones simples.",
                         "Cada fracción simple es una integral inmediata o una sustitución sencilla.")
                try:
                    r = self.solve(ap, x, depth + 1)
                    if self._valid(r, f, x):
                        return r
                except _GiveUp:
                    pass
                del self.steps[n:], self.methods[m:]

        r = sp.integrate(f, x)
        if r.has(sp.Integral):
            raise _GiveUp
        self.used_fallback = True
        self.methods.append("algoritmo simbólico general")
        self.add("Resolución con el algoritmo simbólico general", self.I(f, x), L(r), "",
                 "Este tramo no tiene un desglose didáctico programado; lo resuelve el algoritmo general de SymPy "
                 "(heurísticas + algoritmo de Risch).",
                 "Se declara explícitamente para no inventar pasos que no corresponden al resultado.")
        return r

    def _solve_or_fallback(self, f, x, depth):
        try:
            return self.solve(f, x, depth)
        except _GiveUp:
            r = sp.integrate(f, x)
            if r.has(sp.Integral):
                raise
            self.used_fallback = True
            self.add("Término resuelto con el algoritmo general", self.I(f, x), L(r), "",
                     "Este término no tiene desglose didáctico; lo resuelve SymPy.", "Se declara por transparencia.")
            return r


def antiderivative_steps(f, x, prefer: str = "auto"):
    eng = _Integ(x, prefer)
    try:
        F = eng.solve(f, x)
    except _GiveUp:
        return None, eng
    return F, eng


def integral_solution(f, x, a=None, b=None, prefer: str = "auto") -> Solution:
    f = sp.sympify(f)
    definite = a is not None and b is not None
    improper = definite and (sp.sympify(a).has(sp.oo, -sp.oo) or sp.sympify(b).has(sp.oo, -sp.oo))
    if definite:
        kind = "impropia" if improper else "definida"
        problem = rf"\int_{{{L(a)}}}^{{{L(b)}}} {L(f)}\,d{L(x)}"
    else:
        kind = "indefinida"
        problem = rf"\int {L(f)}\,d{L(x)}"
    sol = Solution(topic=f"Cálculo integral · integral {kind}", problem=problem,
                   given=[f"Integrando: $f({L(x)})={L(f)}$", f"Variable: ${L(x)}$"] +
                         ([f"Límites: de ${L(a)}$ a ${L(b)}$"] if definite else []),
                   goal="Hallar una antiderivada" + (" y evaluarla en los límites." if definite else " (más la constante C)."),
                   theory=THEORY["integral"], errors=COMMON_ERRORS["integral"])

    F, eng = antiderivative_steps(f, x, prefer)
    sol.steps = list(eng.steps)

    if F is None:
        sol.method = "Sin antiderivada elemental conocida"
        if definite:
            d = calculus.definite(f, x, a, b)
            sol.notes.append(d["note"])
            if d["numeric"] is not None:
                val, err = d["numeric"]
                sol.exact = False
                sol.steps.append(Step("Aproximación numérica", "", rf"\approx {val:.12g}",
                                      r"\int_a^b f\,dx\approx\sum w_i f(x_i)",
                                      "Cuadratura adaptativa de Gauss-Kronrod (QUADPACK).",
                                      "Sin antiderivada elemental, el valor sólo puede aproximarse."))
                sol.result = rf"{problem}\approx {val:.12g}\quad(\text{{error estimado}}\approx {err:.1e})"
                sol.verification = "Resultado APROXIMADO (no exacto)."
            elif d["exact"] is not None:
                sol.result = rf"{problem}={L(d['exact'])}"
                sol.verification = "Resultado simbólico de SymPy."
            else:
                sol.result = r"\text{No se pudo evaluar}"
                sol.verification = d["note"]
        else:
            sol.result = r"\text{Sin antiderivada elemental conocida}"
            sol.verification = "Puede intentar la versión definida para obtener una aproximación numérica."
        return sol

    sol.method = " + ".join(dict.fromkeys(eng.methods)) or "tabla"
    if eng.used_fallback:
        sol.notes.append("Parte del procedimiento se resolvió con el algoritmo simbólico general de SymPy "
                         "(declarado en el paso correspondiente).")
    Fs = sp.simplify(F)
    if not definite:
        sol.result = rf"{problem}={L(F)}+C"
        sol.verification = ("✔ Verificación por derivación: " + rf"$\frac{{d}}{{d{L(x)}}}\left[{L(Fs)}\right]$ coincide con el integrando."
                            if equivalent(sp.diff(F, x), f) else "⚠ La verificación por derivación no fue concluyente.")
        sol.steps.append(Step("Añadir la constante de integración", L(F), rf"{L(F)}+C", r"\int f\,dx=F(x)+C",
                              "Toda antiderivada se define salvo una constante.",
                              "La derivada de una constante es cero, así que hay infinitas antiderivadas."))
        return sol

    # definida / impropia: Teorema Fundamental con límites laterales
    a_, b_ = sp.sympify(a), sp.sympify(b)

    def eval_end(p, side):
        if p in (sp.oo, -sp.oo):
            return sp.limit(F, x, p)
        try:
            val = F.subs(x, p)
            if val.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
                raise ValueError
            return sp.simplify(val)
        except Exception:  # noqa: BLE001
            return sp.limit(F, x, p, dir=side)

    try:
        Fb, Fa = eval_end(b_, "-"), eval_end(a_, "+")
    except Exception:  # noqa: BLE001
        Fb = Fa = None
    if Fb is None or any(v.has(sp.nan, sp.zoo) for v in (Fa, Fb)):
        d = calculus.definite(f, x, a, b)
        sol.notes.append("No se pudieron evaluar los extremos de forma exacta. " + d["note"])
        if d["exact"] is not None:
            sol.result = rf"{problem}={L(d['exact'])}"
        elif d["numeric"] is not None:
            sol.exact = False
            sol.result = rf"{problem}\approx {d['numeric'][0]:.12g}"
        else:
            sol.result = r"\text{No se pudo evaluar}"
        return sol

    if improper:
        sol.steps.append(Step("Integral impropia: usar límites", problem,
                              rf"\lim_{{t\to {L(b_ if b_.has(sp.oo, -sp.oo) else a_)}}}\left[F(t)\right]",
                              r"\int_a^{\infty}f\,dx=\lim_{t\to\infty}\int_a^t f\,dx",
                              "Un límite de integración infinito se reemplaza por una variable t que tiende a él.",
                              "La integral sólo existe si ese límite es finito."))
    sol.steps.append(Step("Teorema Fundamental del Cálculo", problem, rf"F({L(b_)})-F({L(a_)})",
                          r"\int_a^b f\,dx=F(b)-F(a)",
                          f"Se evalúa la antiderivada $F({L(x)})={L(F)}$ en los extremos.",
                          "Si F′ = f y f es continua en [a, b], la integral definida es F(b) − F(a)."))
    sol.steps.append(Step("Sustituir los límites", "", rf"\left({L(Fb)}\right)-\left({L(Fa)}\right)", "",
                          f"$F({L(b_)})={L(Fb)}$ y $F({L(a_)})={L(Fa)}$.", "Se evalúa en cada extremo y se resta."))
    val = sp.simplify(Fb - Fa)
    sol.steps.append(Step("Resultado", rf"\left({L(Fb)}\right)-\left({L(Fa)}\right)", L(val), "",
                          "Se simplifica la resta.", "Es el valor exacto de la integral."))
    if val in (sp.oo, -sp.oo):
        sol.result = rf"{problem}=\infty\quad\text{{(la integral diverge)}}"
        sol.verification = "La integral impropia diverge."
        return sol
    sol.result = rf"{problem}={L(val)}"
    try:
        num, err = calculus.numeric_integral_1d(f, x, a_, b_)
        ok = abs(num - float(sp.N(val))) <= 1e-6 * (1 + abs(num))
        sol.verification = (f"✔ Verificación numérica independiente (cuadratura): {num:.10g}  ≈  {float(sp.N(val)):.10g}."
                            if ok else f"⚠ La cuadratura numérica da {num:.10g}, distinta del valor simbólico {float(sp.N(val)):.10g}: "
                                       "revise singularidades dentro del intervalo.")
    except Exception:  # noqa: BLE001
        sol.verification = "La verificación numérica no fue posible (posible singularidad o divergencia)."
    sol.final = val  # type: ignore[attr-defined]
    return sol
