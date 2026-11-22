"""Generador de ejercicios similares. Las respuestas se calculan con SymPy al generar el ejercicio."""
from __future__ import annotations

import random

import sympy as sp

x = sp.Symbol("x")


def _rand(rng, lo=1, hi=5):
    return rng.randint(lo, hi)


def _derivatives(rng):
    a, b, n = _rand(rng), _rand(rng, 2, 6), _rand(rng, 2, 5)
    return rng.choice([
        a * x ** n + b * sp.sin(a * x),
        x ** n * sp.exp(a * x),
        sp.sin(a * x ** 2 + b),
        sp.log(x ** 2 + a),
        (x ** 2 + a) / (x + b),
        sp.exp(x) * sp.cos(b * x),
        sp.sqrt(a * x ** 2 + b),
    ])


def _integrals(rng):
    a, b, n = _rand(rng), _rand(rng, 2, 4), _rand(rng, 2, 5)
    return rng.choice([
        a * x ** n + b * x,
        x * sp.exp(a * x),
        x * sp.cos(a * x),
        2 * a * x * sp.exp(x ** 2),
        sp.sin(a * x) * sp.cos(a * x),
        1 / (x ** 2 + a ** 2),
        x * sp.log(x),
        (2 * x + a) / (x ** 2 + a * x + b),
    ])


def _limits(rng):
    a = _rand(rng, 1, 4)
    return rng.choice([
        ((x ** 2 - a ** 2) / (x - a), a),
        (sp.sin(a * x) / x, 0),
        ((sp.sqrt(x + a ** 2) - a) / x, 0),
        ((1 + a / x) ** x, sp.oo),
        ((sp.exp(a * x) - 1) / x, 0),
        ((x ** 3 + a * x) / (2 * x ** 3 - 1), sp.oo),
    ])


def similar_exercises(topic: str, n: int = 3, seed: int | None = None) -> list[dict]:
    """Devuelve [{'statement': latex, 'answer': latex}, ...]."""
    rng = random.Random(seed)
    out, tries = [], 0
    while len(out) < n and tries < 60:
        tries += 1
        try:
            if topic == "derivada":
                f = _derivatives(rng)
                out.append({"statement": rf"\dfrac{{d}}{{dx}}\left[{sp.latex(f)}\right]",
                            "answer": sp.latex(sp.simplify(sp.diff(f, x)))})
            elif topic == "integral":
                f = _integrals(rng)
                F = sp.integrate(f, x)
                if F.has(sp.Integral):
                    continue
                out.append({"statement": rf"\int {sp.latex(f)}\,dx", "answer": sp.latex(sp.simplify(F)) + "+C"})
            elif topic == "limite":
                f, a = _limits(rng)
                res = sp.limit(f, x, a)
                out.append({"statement": rf"\lim_{{x\to {sp.latex(a)}}} {sp.latex(f)}", "answer": sp.latex(res)})
            else:
                return []
        except Exception:  # noqa: BLE001
            continue
    return out
