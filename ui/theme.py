"""Tema claro/oscuro mediante CSS (Streamlit no permite cambiar su tema en caliente)."""
from __future__ import annotations

import streamlit as st

_BASE = """
<style>
.stApp, section[data-testid="stSidebar"], .stButton button, input, textarea, [data-baseweb="select"] > div {
    transition: background-color .25s ease, color .25s ease, border-color .25s ease, transform .12s ease, box-shadow .2s ease;
}
.stButton button { border-radius: 10px; font-weight: 600; }
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,.18); }
.stButton button:active { transform: translateY(1px); }
.calc-card { border-radius: 16px; padding: 14px 18px; margin: 6px 0 14px 0; animation: fadein .35s ease; }
.calc-display { font-family: "Cambria Math","STIX Two Math","Times New Roman",serif; font-size: 1.15rem; opacity:.85; }
.calc-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px; }
@keyframes fadein { from {opacity:0; transform: translateY(4px);} to {opacity:1; transform:none;} }
@media (max-width: 768px) {
  .block-container { padding: 1rem .6rem 4rem .6rem !important; }
  .stButton button { padding: .35rem .2rem; font-size: .95rem; }
}
</style>
"""

_DARK = """
<style>
.stApp { background: #0e1320; color: #e8ecf5; }
section[data-testid="stSidebar"] { background: #141a2b; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div[data-testid="stMarkdownContainer"] { color: #e8ecf5; }
.stTextInput input, .stTextArea textarea, .stNumberInput input { background: #1b2338 !important; color: #f2f5fb !important; border: 1px solid #2c3858 !important; }
[data-baseweb="select"] > div { background: #1b2338 !important; color: #f2f5fb !important; border-color: #2c3858 !important; }
.stButton button { background: #1d2740; color: #e8ecf5; border: 1px solid #2f3d63; }
.stButton button:hover { border-color: #4f8cff; color: #fff; }
.calc-card { background: linear-gradient(145deg,#161e33,#101728); border: 1px solid #263154; }
.calc-chip { background:#223056; color:#9cc0ff; }
div[data-testid="stExpander"] { background:#121a2d; border:1px solid #243052; border-radius:12px; }
.katex { color: #f2f5fb; }
</style>
"""

_LIGHT = """
<style>
.stApp { background: #f4f7fc; color: #162036; }
section[data-testid="stSidebar"] { background: #e9eef8; }
.stButton button { background: #ffffff; color: #1a2540; border: 1px solid #cdd7ea; }
.stButton button:hover { border-color: #3b74e6; color: #0d1a3a; }
.calc-card { background: linear-gradient(145deg,#ffffff,#eef3fb); border: 1px solid #d5deef; }
.calc-chip { background:#dfe9ff; color:#2556b8; }
div[data-testid="stExpander"] { background:#ffffff; border:1px solid #d8e0f0; border-radius:12px; }
</style>
"""


def apply(theme: str) -> None:
    st.markdown(_BASE + (_DARK if theme == "Oscuro" else _LIGHT), unsafe_allow_html=True)
