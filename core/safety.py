"""Límite de tiempo para operaciones simbólicas costosas."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from .parser import ExpressionError

_POOL = ThreadPoolExecutor(max_workers=4)


def run_with_timeout(fn, seconds: int = 30, *args, **kwargs):
    """Ejecuta fn(*args) y devuelve su resultado o lanza ExpressionError si tarda demasiado.

    Nota: un hilo de Python no puede matarse; si se agota el tiempo, la interfaz
    recupera el control pero el cálculo puede seguir consumiendo CPU hasta terminar.
    """
    fut = _POOL.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=seconds)
    except FuturesTimeout:
        fut.cancel()
        raise ExpressionError(
            f"La operación superó {seconds} s y se canceló. Pruebe con una expresión más simple "
            "o use la aproximación numérica."
        )
