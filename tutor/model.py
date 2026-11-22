"""Modelo de datos común del Modo Estudiante."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import sympy as sp

LEVELS = ["Básico", "Intermedio", "Avanzado", "Universitario completo"]
DEFAULT_LEVEL = LEVELS[-1]


def L(e) -> str:
    """LaTeX de un objeto SymPy (o texto ya en LaTeX)."""
    return e if isinstance(e, str) else sp.latex(e)


@dataclass
class Step:
    title: str
    before: str = ""        # LaTeX antes de aplicar la regla
    after: str = ""         # LaTeX después
    formula: str = ""       # LaTeX de la fórmula/propiedad usada
    explanation: str = ""   # qué se hizo (Markdown)
    reason: str = ""        # por qué se hizo


@dataclass
class Solution:
    topic: str
    problem: str                      # LaTeX
    given: list[str] = field(default_factory=list)
    goal: str = ""
    method: str = ""
    theory: str = ""
    steps: list[Step] = field(default_factory=list)
    result: str = ""                  # LaTeX
    verification: str = ""
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    exact: bool = True


COMMON_ERRORS = {
    "derivada": [
        "Olvidar la regla de la cadena: la derivada de sen(g(x)) es cos(g(x))·g′(x), no sólo cos(g(x)).",
        "Derivar un producto como producto de derivadas: (uv)′ ≠ u′v′.",
        "En el cociente, invertir el orden del numerador: es (u′v − uv′)/v².",
        "Derivar aᵘ como si fuera una potencia (u·aᵘ⁻¹): una exponencial se deriva como aᵘ·ln(a)·u′.",
    ],
    "integral": [
        "Olvidar la constante de integración +C en integrales indefinidas.",
        "Al sustituir u, no cambiar también el diferencial (dx → du/u′) o no regresar a la variable original.",
        "En integrales definidas con sustitución: no cambiar los límites o mezclar límites en x con una antiderivada en u.",
        "Integrar por partes eligiendo mal u (use LIATE) y obtener una integral más difícil.",
        "Aplicar ∫1/x dx = ln|x| sin el valor absoluto.",
    ],
    "limite": [
        "Concluir que un límite «no existe» cuando sólo se obtuvo una forma indeterminada (0/0, ∞/∞...).",
        "Aplicar L'Hôpital sin verificar que la forma sea 0/0 o ∞/∞.",
        "Derivar el cociente completo en L'Hôpital en lugar de derivar numerador y denominador por separado.",
        "Ignorar los límites laterales cuando hay valor absoluto, partes enteras o funciones definidas por tramos.",
    ],
    "ecuacion": [
        "Dividir ambos lados entre una expresión que puede valer cero (se pierden soluciones).",
        "Elevar al cuadrado ambos lados y no comprobar las soluciones (aparecen raíces extrañas).",
        "Errores de signo al aplicar la fórmula general: −b ± √(b² − 4ac).",
    ],
    "multiple": [
        "Dibujar mal la región y poner límites incorrectos o en el orden equivocado.",
        "Olvidar el Jacobiano (r en polares/cilíndricas, ρ² senφ en esféricas) al cambiar de coordenadas.",
        "Confundir las convenciones de θ y φ en coordenadas esféricas.",
        "Integrar en el orden incorrecto: los límites de la integral interna pueden depender de las externas, nunca al revés.",
    ],
    "vectorial": [
        "Confundir gradiente (campo vectorial), divergencia (escalar) y rotacional (campo vectorial).",
        "Calcular un punto crítico y no clasificarlo con la matriz Hessiana.",
        "En Lagrange: olvidar incluir la restricción como ecuación del sistema.",
    ],
    "algebra_lineal": [
        "Multiplicar matrices sin comprobar dimensiones o conmutar el producto (AB ≠ BA en general).",
        "Intercambiar filas en Gauss-Jordan y olvidar que eso cambia el signo del determinante.",
    ],
}

THEORY = {
    "derivada": ("La derivada f′(x) = lim_{h→0} [f(x+h) − f(x)]/h existe en x si ese límite es finito. "
                 "Las reglas de derivación (suma, producto, cociente, cadena) se deducen de esta definición y valen "
                 "donde las funciones involucradas son derivables."),
    "integral": ("Teorema Fundamental del Cálculo: si F′ = f en [a, b] y f es continua, entonces ∫ₐᵇ f(x)dx = F(b) − F(a). "
                 "La antiderivada no es única: dos antiderivadas difieren en una constante."),
    "limite": ("lim_{x→a} f(x) = L si f(x) se acerca a L cuando x se acerca a a por ambos lados. "
               "Una forma indeterminada no es un valor: obliga a transformar la expresión."),
    "ecuacion": "Resolver una ecuación es hallar los valores de la incógnita que la convierten en una igualdad verdadera.",
    "multiple": ("Teorema de Fubini: si f es continua en una región acotada, la integral múltiple puede calcularse como "
                 "integral iterada. El cambio de variables multiplica por |det J| (valor absoluto del Jacobiano)."),
    "vectorial": "Los operadores ∇, ∇·, ∇× combinan derivadas parciales de campos escalares y vectoriales.",
    "algebra_lineal": "Las operaciones elementales por filas no cambian el conjunto solución de un sistema lineal.",
}


def equivalent(a, b, trials: int = 5) -> bool:
    """¿Son a y b la misma expresión? Primero simbólico; si no concluye, numérico en puntos aleatorios."""
    a, b = sp.sympify(a), sp.sympify(b)
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    syms = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(1234)
    ok = 0
    for _ in range(trials * 3):
        pt = {s: sp.Float(rng.uniform(0.3, 2.0)) for s in syms}
        try:
            va, vb = complex(sp.N(a.subs(pt))), complex(sp.N(b.subs(pt)))
        except Exception:  # noqa: BLE001
            continue
        if abs(va - vb) > 1e-7 * (1 + abs(va)):
            return False
        ok += 1
        if ok >= trials:
            return True
    return ok >= 3


def fresh_symbol(*exprs, candidates=("u", "w", "s", "q", "p")) -> sp.Symbol:
    used = set()
    for e in exprs:
        used |= {s.name for s in sp.sympify(e).free_symbols}
    for c in candidates:
        if c not in used:
            return sp.Symbol(c)
    return sp.Symbol("u0")
