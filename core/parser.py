"""Análisis sintáctico SEGURO de expresiones matemáticas.

Nunca se llama a eval()/exec() sobre el texto original del usuario. El texto se
valida carácter por carácter, se normaliza y sólo entonces se entrega al
analizador de SymPy con un espacio de nombres cerrado (sin builtins y sin acceso
a atributos).
"""
from __future__ import annotations

import re
from tokenize import TokenError

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class ExpressionError(ValueError):
    """Error de sintaxis o de dominio con un mensaje apto para el usuario."""


MAX_LEN = 600
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

DEFAULT_SYMBOLS = [
    "x", "y", "z", "t", "u", "v", "w", "r", "s", "a", "b", "c", "d", "h", "k",
    "m", "n", "p", "q", "theta", "phi", "rho", "alpha", "beta", "lam", "mu", "omega",
]

_UNICODE = {
    "π": "pi", "∞": "oo", "θ": "theta", "φ": "phi", "ρ": "rho", "α": "alpha",
    "β": "beta", "λ": "lam", "μ": "mu", "ω": "omega", "×": "*", "÷": "/",
    "−": "-", "·": "*", "²": "**2", "³": "**3",
}

_ALLOWED = re.compile(r"[0-9A-Za-z_+\-*/()\s,.=!]")

_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
    "Lambda": sp.Lambda,
    "factorial": sp.factorial,
    "factorial2": sp.factorial2,
}


def namespace(angle_mode: str = "rad") -> dict:
    """Espacio de nombres cerrado: sólo funciones y constantes matemáticas."""
    ns: dict = {n: sp.Symbol(n) for n in DEFAULT_SYMBOLS}
    ns.update(pi=sp.pi, e=sp.E, i=sp.I, oo=sp.oo, inf=sp.oo, infinity=sp.oo)

    k = sp.pi / 180
    deg = angle_mode == "deg"

    def direct(f):
        return (lambda a: f(a * k)) if deg else f

    def inverse(f):
        return (lambda a: f(a) / k) if deg else f

    for n in ("sin", "cos", "tan", "cot", "sec", "csc"):
        ns[n] = direct(getattr(sp, n))
    for n in ("asin", "acos", "atan", "acot", "asec", "acsc"):
        f = inverse(getattr(sp, n))
        ns[n] = f
        ns["arc" + n[1:]] = f
    for n in ("sinh", "cosh", "tanh", "coth", "sech", "csch"):
        ns[n] = getattr(sp, n)
    for n in ("asinh", "acosh", "atanh", "acoth", "asech", "acsch"):
        f = getattr(sp, n)
        ns[n] = f
        ns["arc" + n[1:]] = f

    def log(x, base=None):
        # Convención de calculadora: log(x) = base 10; ln(x) = natural.
        return sp.log(x, 10) if base is None else sp.log(x, base)

    ns.update(
        exp=sp.exp, ln=sp.log, log=log,
        log10=lambda x: sp.log(x, 10), log2=lambda x: sp.log(x, 2),
        sqrt=sp.sqrt, cbrt=sp.cbrt, root=sp.root,
        abs=sp.Abs, Abs=sp.Abs, sign=sp.sign,
        floor=sp.floor, ceil=sp.ceiling, ceiling=sp.ceiling,
        factorial=sp.factorial, Gamma=sp.gamma, binomial=sp.binomial,
        gcd=sp.gcd, lcm=sp.lcm, mod=sp.Mod,
        re=sp.re, im=sp.im, conj=sp.conjugate, arg=sp.arg,
        max=sp.Max, min=sp.Min, Max=sp.Max, Min=sp.Min,
        erf=sp.erf, Heaviside=sp.Heaviside, DiracDelta=sp.DiracDelta,
        rad=lambda a: a * sp.pi / 180, deg_=lambda a: a * 180 / sp.pi,
    )
    ns["deg"] = ns.pop("deg_")
    return ns


def split_top_level(s: str, sep: str = ",") -> list[str]:
    """Divide por `sep` ignorando separadores dentro de paréntesis."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _preprocess(text: str) -> str:
    if text is None or not str(text).strip():
        raise ExpressionError("La expresión está vacía.")
    s = str(text).strip()
    if len(s) > MAX_LEN:
        raise ExpressionError(f"La expresión es demasiado larga (máximo {MAX_LEN} caracteres).")

    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*([0-9]+(?:\.[0-9]+)?|[A-Za-z_]\w*)", r"sqrt(\1)", s)
    for k, v in _UNICODE.items():
        s = s.replace(k, v)
    s = s.replace("^", "**")

    # |x| -> Abs(x)  (de dentro hacia fuera)
    for _ in range(10):
        new = re.sub(r"\|([^|]*)\|", r"Abs(\1)", s)
        if new == s:
            break
        s = new
    # 25% -> (25/100)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)

    for ch in s:
        if not _ALLOWED.fullmatch(ch):
            raise ExpressionError(f"Carácter no permitido: «{ch}». Use sólo números, letras, + - * / ^ ( ) , . ! y símbolos matemáticos soportados.")
    if "__" in s or re.search(r"[A-Za-z_)]\s*\.\s*[A-Za-z_]", s):
        raise ExpressionError("Sintaxis no permitida (acceso a atributos).")

    depth = 0
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if depth < 0:
            break
    if depth != 0:
        raise ExpressionError("Paréntesis desbalanceados: revise que cada «(» tenga su «)».")

    # 2x -> 2*x, 3(x+1) -> 3*(x+1)  (sin romper 1.5e-3)
    s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)(?![eE][+-]?\d)(?=[A-Za-z_(])", r"\1*", s)
    return s


def _parse(s: str, angle_mode: str, extra_symbols) -> sp.Basic:
    ns = namespace(angle_mode)
    for n in extra_symbols or ():
        ns.setdefault(n, sp.Symbol(n))
    try:
        expr = parse_expr(s, local_dict=ns, global_dict=dict(_GLOBALS),
                          transformations=TRANSFORMS, evaluate=True)
    except ZeroDivisionError as exc:
        raise ExpressionError("División entre cero.") from exc
    except RecursionError as exc:
        raise ExpressionError("La expresión es demasiado compleja.") from exc
    except (SyntaxError, TokenError, TypeError, AttributeError, NameError, ValueError, sp.SympifyError) as exc:
        raise ExpressionError(
            "No se pudo interpretar la expresión. Revise paréntesis, operadores y nombres de funciones "
            f"(detalle técnico: {type(exc).__name__})."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - cualquier otro fallo del analizador
        raise ExpressionError(f"Expresión no válida ({type(exc).__name__}).") from exc

    if not isinstance(expr, sp.Basic):
        try:
            expr = sp.sympify(expr)
        except Exception as exc:  # noqa: BLE001
            raise ExpressionError("La expresión no produjo un objeto matemático.") from exc
    if expr.has(sp.zoo, sp.nan):
        raise ExpressionError("La expresión no está definida (división entre cero o forma indeterminada).")
    return expr


def parse_expression(text: str, angle_mode: str = "rad", extra_symbols=()) -> sp.Basic:
    s = _preprocess(text)
    if "=" in s:
        raise ExpressionError("Aquí se espera una expresión sin «=». Para ecuaciones use el módulo de Álgebra.")
    return _parse(s, angle_mode, extra_symbols)


def parse_equation(text: str, angle_mode: str = "rad", extra_symbols=()) -> sp.Eq:
    s = _preprocess(text)
    if s.count("=") > 1:
        raise ExpressionError("La ecuación debe tener un solo signo «=».")
    if "=" in s:
        lhs, rhs = s.split("=")
        return sp.Eq(_parse(lhs, angle_mode, extra_symbols), _parse(rhs, angle_mode, extra_symbols))
    return sp.Eq(_parse(s, angle_mode, extra_symbols), 0)


def parse_symbol(text: str) -> sp.Symbol:
    name = (text or "").strip()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ExpressionError(f"«{name}» no es un nombre de variable válido.")
    return sp.Symbol(name)


def parse_symbols(text: str) -> list[sp.Symbol]:
    names = [p for p in re.split(r"[,\s]+", (text or "").strip()) if p]
    if not names:
        raise ExpressionError("Indique al menos una variable.")
    return [parse_symbol(n) for n in names]


def _strip_outer(s: str) -> str:
    s = s.strip()
    if s.startswith("<") and s.endswith(">"):
        return s[1:-1]
    while s and s[0] in "([" and s[-1] in ")]":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            depth += ch in "(["
            depth -= ch in ")]"
            if depth == 0 and i < len(s) - 1:
                ok = False
                break
        if not ok:
            break
        s = s[1:-1].strip()
    return s


def parse_vector(text: str, dim: int | None = None, angle_mode: str = "rad", extra_symbols=()) -> sp.Matrix:
    parts = [p for p in split_top_level(_strip_outer(text)) if p != ""]
    if not parts:
        raise ExpressionError("Vector vacío.")
    comps = [parse_expression(p, angle_mode, extra_symbols) for p in parts]
    if dim is not None and len(comps) != dim:
        raise ExpressionError(f"Se esperaban {dim} componentes y se recibieron {len(comps)}.")
    return sp.Matrix(comps)


def parse_matrix(text: str, angle_mode: str = "rad") -> sp.Matrix:
    s = (text or "").strip()
    s = re.sub(r"\]\s*,\s*\[", ";", s)
    s = s.replace("[", "").replace("]", "")
    rows = [r for r in re.split(r"[;\n]", s) if r.strip()]
    if not rows:
        raise ExpressionError("Matriz vacía.")
    data = []
    for r in rows:
        data.append([parse_expression(c, angle_mode) for c in split_top_level(r) if c != ""])
    width = {len(r) for r in data}
    if len(width) != 1:
        raise ExpressionError("Todas las filas de la matriz deben tener el mismo número de columnas.")
    return sp.Matrix(data)
