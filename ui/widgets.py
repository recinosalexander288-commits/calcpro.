"""Componentes compartidos: estado, barra lateral, historial, render de soluciones."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from core.parser import ExpressionError
from tutor.exercises import similar_exercises
from tutor.model import DEFAULT_LEVEL, LEVELS, Solution


@dataclass
class Settings:
    mode: str          # "Estudiante" | "Profesional"
    level: str
    angle: str         # "rad" | "deg"
    precision: int
    scientific: bool
    theme: str         # "Claro" | "Oscuro"


# ───────────────────────── utilidades de API de Streamlit ─────────────────────────
def btn(label: str, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)


def plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def download(label, data, file_name, mime):
    try:
        st.download_button(label, data, file_name=file_name, mime=mime, width="stretch")
    except TypeError:
        st.download_button(label, data, file_name=file_name, mime=mime, use_container_width=True)


# ───────────────────────── estado ─────────────────────────
def state_init() -> None:
    defaults = {
        "history": [], "favorites": [], "memory": "0", "expr": "", "last_exact": None,
        "mode": "Estudiante", "level": DEFAULT_LEVEL, "angle": "rad", "precision": 10,
        "scientific": False, "theme": "Oscuro", "ans": "0",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def sidebar() -> Settings:
    sb = st.sidebar
    sb.markdown("## ∑ CalcPro")
    sb.caption("Calculadora científica · CAS · graficador · tutor")
    sb.radio("Modo", ["Estudiante", "Profesional"], key="mode", horizontal=True,
             help="Estudiante: procedimiento completo y explicado. Profesional: resultado rápido.")
    if st.session_state.mode == "Estudiante":
        sb.selectbox("Nivel de explicación", LEVELS, key="level")
    sb.radio("Ángulos", ["rad", "deg"], key="angle", horizontal=True,
             format_func=lambda a: "Radianes" if a == "rad" else "Grados")
    sb.slider("Cifras significativas", 3, 50, key="precision")
    sb.checkbox("Notación científica", key="scientific")
    sb.radio("Tema", ["Oscuro", "Claro"], key="theme", horizontal=True)
    return Settings(st.session_state.mode, st.session_state.level, st.session_state.angle,
                    int(st.session_state.precision), bool(st.session_state.scientific), st.session_state.theme)


# ───────────────────────── historial / favoritos ─────────────────────────
def add_history(kind: str, expr: str, result_latex: str, result_text: str = "") -> None:
    st.session_state.history.insert(0, {
        "hora": datetime.now().strftime("%H:%M:%S"), "tipo": kind, "expresión": expr,
        "latex": result_latex, "texto": result_text or result_latex,
    })
    del st.session_state.history[200:]


def _recall(expr: str) -> None:
    st.session_state.expr = expr


def history_panel() -> None:
    hist, favs = st.session_state.history, st.session_state.favorites
    t1, t2 = st.tabs([f"Historial ({len(hist)})", f"Favoritos ({len(favs)})"])
    with t1:
        if not hist:
            st.caption("Aún no hay operaciones.")
        for i, h in enumerate(hist[:30]):
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.markdown(f"`{h['hora']}` **{h['tipo']}** — `{h['expresión']}`")
            c2.button("↺", key=f"rec_{i}", help="Recuperar", on_click=_recall, args=(h["expresión"],))
            if c3.button("★", key=f"fav_{i}", help="Guardar en favoritos"):
                if h not in favs:
                    favs.append(h)
                    st.rerun()
            st.latex(h["latex"])
        if hist:
            _export(hist)
            if btn("Borrar historial", key="clear_hist"):
                st.session_state.history = []
                st.rerun()
    with t2:
        if not favs:
            st.caption("Marque ★ en el historial para guardar operaciones favoritas.")
        for i, h in enumerate(favs):
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.markdown(f"**{h['tipo']}** — `{h['expresión']}`")
            c2.button("↺", key=f"frec_{i}", on_click=_recall, args=(h["expresión"],))
            if c3.button("✕", key=f"fdel_{i}"):
                favs.pop(i)
                st.rerun()
            st.latex(h["latex"])


def _export(hist) -> None:
    c1, c2, c3 = st.columns(3)
    txt = "\n".join(f"[{h['hora']}] {h['tipo']}: {h['expresión']}  →  {h['texto']}" for h in hist)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["hora", "tipo", "expresion", "resultado_latex"])
    for h in hist:
        w.writerow([h["hora"], h["tipo"], h["expresión"], h["latex"]])
    tex = "\\documentclass{article}\\usepackage{amsmath}\\begin{document}\n" + "\n".join(
        f"\\noindent\\texttt{{{h['expresión'].replace('_', ' ')}}}\\[{h['latex']}\\]" for h in hist) + "\n\\end{document}"
    with c1:
        download("TXT", txt, "historial.txt", "text/plain")
    with c2:
        download("CSV", buf.getvalue(), "historial.csv", "text/csv")
    with c3:
        download("LaTeX", tex, "historial.tex", "text/plain")
    download("JSON", json.dumps(hist, ensure_ascii=False, indent=2), "historial.json", "application/json")


# ───────────────────────── render de soluciones ─────────────────────────
def _safe_latex(s: str) -> None:
    try:
        st.latex(s)
    except Exception:  # noqa: BLE001
        st.code(s)


def render_solution(sol: Solution, level: str, key: str = "sol") -> None:
    """Muestra el procedimiento según el nivel (Básico … Universitario completo)."""
    full = level == LEVELS[3]
    adv = level in (LEVELS[2], LEVELS[3])
    mid = level != LEVELS[0]

    show = st.toggle("Mostrar procedimiento", value=True, key=f"{key}_show")
    if not show:
        return
    expand_all = st.checkbox("Expandir todos los pasos", value=full, key=f"{key}_exp")

    st.markdown(f"##### 📘 {sol.topic}")
    if mid:
        _safe_latex(sol.problem)
    if adv:
        for g in sol.given:
            st.markdown(f"- {g}")
        if sol.goal:
            st.markdown(f"**Qué se busca:** {sol.goal}")
        if sol.method:
            st.markdown(f"**Método seleccionado:** {sol.method}")
    if full and sol.theory:
        st.info(f"**Fundamento teórico.** {sol.theory}")

    for i, s in enumerate(sol.steps, 1):
        with st.expander(f"Paso {i}: {s.title}", expanded=expand_all):
            if s.explanation:
                st.markdown(s.explanation)
            if mid and s.formula:
                st.caption("Fórmula / propiedad utilizada")
                _safe_latex(s.formula)
            if adv and s.before:
                st.caption("Antes")
                _safe_latex(s.before)
            if s.after:
                st.caption("Después" if adv and s.before else "Resultado del paso")
                _safe_latex(s.after)
            if adv and s.reason:
                st.caption(f"❓ ¿Por qué? {s.reason}")

    st.markdown("**Resultado final**")
    _safe_latex(sol.result)
    if not sol.exact:
        st.warning("Este resultado es una APROXIMACIÓN numérica, no un valor exacto.")
    for n in sol.notes:
        st.info(n)
    if sol.verification and (mid or True):
        (st.success if sol.verification.startswith("✔") else st.warning)(sol.verification)
    if full and sol.errors:
        with st.expander("⚠️ Errores frecuentes en este tema"):
            for e in sol.errors:
                st.markdown(f"- {e}")


def similar_panel(topic: str, key: str) -> None:
    """Botón para generar ejercicios de práctica similares."""
    if topic not in ("derivada", "integral", "limite"):
        return
    with st.expander("🎯 Generar ejercicios similares para practicar"):
        if btn("Generar 3 ejercicios", key=f"{key}_gen"):
            st.session_state[f"{key}_ex"] = similar_exercises(topic, 3)
        for i, ex in enumerate(st.session_state.get(f"{key}_ex", []), 1):
            st.markdown(f"**Ejercicio {i}**")
            _safe_latex(ex["statement"])
            with st.expander("Ver respuesta"):
                _safe_latex(ex["answer"])


def show_result(result_latex: str, copy_text: str, approx: str | None = None, exact: bool = True) -> None:
    st.markdown('<div class="calc-card">', unsafe_allow_html=True)
    chips = '<span class="calc-chip">' + ("EXACTO" if exact else "APROXIMADO") + "</span>"
    st.markdown(chips, unsafe_allow_html=True)
    _safe_latex(result_latex)
    if approx:
        st.caption(f"Aproximación decimal: {approx}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.code(copy_text, language=None)  # el icono de la esquina copia al portapapeles


def run_guarded(fn, *args, **kwargs):
    """Ejecuta fn mostrando errores matemáticos de forma comprensible. Devuelve None si falló."""
    try:
        return fn(*args, **kwargs)
    except ExpressionError as e:
        st.error(f"⚠️ {e}")
    except (ZeroDivisionError, OverflowError) as e:
        st.error(f"⚠️ Error numérico: {type(e).__name__}. Revise divisiones entre cero o valores demasiado grandes.")
    except Exception as e:  # noqa: BLE001
        st.error(f"⚠️ No se pudo completar la operación ({type(e).__name__}). Revise la sintaxis o simplifique la expresión.")
    return None
