"""Graficador: 2D, paramétricas, polares, implícitas, 3D, campos vectoriales, regiones."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import sympy as sp

from core.parser import ExpressionError

PALETTE = ["#4f8cff", "#ff6b6b", "#2ecc71", "#f5a623", "#a66cff", "#17c3b2", "#ff8fab", "#8d99ae"]
GAP = 1e6  # valores mayores se tratan como asíntotas


def _fn(expr, vars_):
    f = sp.lambdify(vars_, expr, modules=["numpy"])

    def call(*args):
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(f(*args), dtype=float)
            except (TypeError, ValueError):
                out = np.asarray(f(*args), dtype=complex)
                out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
        out = np.broadcast_to(out, np.broadcast(*args).shape).astype(float).copy()
        out[~np.isfinite(out)] = np.nan
        out[np.abs(out) > GAP] = np.nan
        return out
    return call


def _layout(fig, title="", dark=False, equal=False):
    tmpl = "plotly_dark" if dark else "plotly_white"
    fig.update_layout(template=tmpl, title=title, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.15), height=560, hovermode="closest")
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_functions_2d(items, x, xmin, xmax, points=None, show_derivative=False, show_integral=False, dark=False,
                      n: int = 2000):
    """items: [(expr, color, nombre)]. points: [(x0, y0, etiqueta)]"""
    if xmin >= xmax:
        raise ExpressionError("El intervalo es inválido: x mín debe ser menor que x máx.")
    xs = np.linspace(xmin, xmax, n)
    fig = go.Figure()
    for i, (expr, color, name) in enumerate(items):
        ys = _fn(expr, x)(xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name or sp.sstr(expr),
                                 line=dict(color=color or PALETTE[i % len(PALETTE)], width=2.4), connectgaps=False))
        if i == 0 and show_derivative:
            fig.add_trace(go.Scatter(x=xs, y=_fn(sp.diff(expr, x), x)(xs), mode="lines", name="f′(x)",
                                     line=dict(color="#ff6b6b", dash="dash")))
        if i == 0 and show_integral:
            from scipy.integrate import cumulative_trapezoid
            yv = np.nan_to_num(ys, nan=0.0)
            fig.add_trace(go.Scatter(x=xs, y=cumulative_trapezoid(yv, xs, initial=0), mode="lines",
                                     name=f"∫ f dx (desde {xmin:g})", line=dict(color="#2ecc71", dash="dot")))
    for px, py, label in points or []:
        fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers+text", text=[label], textposition="top center",
                                 marker=dict(size=10, color="#f5a623"), name=label or "punto", showlegend=bool(label)))
    return _layout(fig, "Gráfica de funciones", dark)


def plot_parametric_2d(xt, yt, t, t0, t1, color=None, dark=False):
    ts = np.linspace(t0, t1, 2000)
    fig = go.Figure(go.Scatter(x=_fn(xt, t)(ts), y=_fn(yt, t)(ts), mode="lines", line=dict(color=color or PALETTE[0], width=2.6),
                               name="curva paramétrica"))
    return _layout(fig, "Curva paramétrica", dark, equal=True)


def plot_polar(r_expr, th, t0, t1, color=None, dark=False):
    ts = np.linspace(t0, t1, 3000)
    r = _fn(r_expr, th)(ts)
    fig = go.Figure(go.Scatter(x=r * np.cos(ts), y=r * np.sin(ts), mode="lines", line=dict(color=color or PALETTE[0], width=2.6),
                               name="r(θ)"))
    return _layout(fig, "Curva polar", dark, equal=True)


def plot_implicit(F, x, y, xr, yr, color=None, dark=False, n=500):
    xs, ys = np.linspace(*xr, n), np.linspace(*yr, n)
    X, Y = np.meshgrid(xs, ys)
    Z = _fn(F, (x, y))(X, Y)
    fig = go.Figure(go.Contour(x=xs, y=ys, z=Z, showscale=False, hoverinfo="skip",
                               contours=dict(start=0, end=0, size=1, coloring="lines"),
                               line=dict(color=color or PALETTE[0], width=3)))
    return _layout(fig, "Curva implícita F(x, y) = 0", dark, equal=True)


def plot_level_curves(f, x, y, xr, yr, levels=15, dark=False, n=300):
    xs, ys = np.linspace(*xr, n), np.linspace(*yr, n)
    X, Y = np.meshgrid(xs, ys)
    fig = go.Figure(go.Contour(x=xs, y=ys, z=_fn(f, (x, y))(X, Y), ncontours=levels, colorscale="Viridis",
                               contours=dict(showlabels=True)))
    return _layout(fig, "Curvas de nivel", dark, equal=True)


def plot_surface(f, x, y, xr, yr, dark=False, n=90, colorscale="Viridis"):
    xs, ys = np.linspace(*xr, n), np.linspace(*yr, n)
    X, Y = np.meshgrid(xs, ys)
    fig = go.Figure(go.Surface(x=X, y=Y, z=_fn(f, (x, y))(X, Y), colorscale=colorscale))
    fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return _layout(fig, "Superficie z = f(x, y)", dark)


def plot_param_surface(X_, Y_, Z_, u, v, ur, vr, dark=False, n=80, colorscale="Plasma", title="Superficie paramétrica"):
    us, vs = np.linspace(*ur, n), np.linspace(*vr, n)
    U, V = np.meshgrid(us, vs)
    fig = go.Figure(go.Surface(x=_fn(X_, (u, v))(U, V), y=_fn(Y_, (u, v))(U, V), z=_fn(Z_, (u, v))(U, V), colorscale=colorscale))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return _layout(fig, title, dark)


def plot_cylindrical_surface(r_expr, th, z, thr, zr, dark=False):
    """r = f(θ, z)"""
    r_, t_, z_ = sp.symbols("r_ t_ z_")
    X = r_expr * sp.cos(th)
    Y = r_expr * sp.sin(th)
    return plot_param_surface(X, Y, z, th, z, thr, zr, dark, title="Superficie en coordenadas cilíndricas r = f(θ, z)")


def plot_spherical_surface(rho_expr, th, ph, thr, phr, dark=False):
    """ρ = f(θ, φ), con φ medido desde el eje +z."""
    X = rho_expr * sp.sin(ph) * sp.cos(th)
    Y = rho_expr * sp.sin(ph) * sp.sin(th)
    Z = rho_expr * sp.cos(ph)
    return plot_param_surface(X, Y, Z, th, ph, thr, phr, dark, title="Superficie en coordenadas esféricas ρ = f(θ, φ)")


def plot_curve_3d(xt, yt, zt, t, t0, t1, color=None, dark=False):
    ts = np.linspace(t0, t1, 1500)
    fig = go.Figure(go.Scatter3d(x=_fn(xt, t)(ts), y=_fn(yt, t)(ts), z=_fn(zt, t)(ts), mode="lines",
                                 line=dict(color=color or PALETTE[0], width=6)))
    fig.update_layout(scene=dict(aspectmode="data"))
    return _layout(fig, "Curva paramétrica 3D", dark)


def plot_vector_field_2d(P, Q, x, y, xr, yr, density=18, dark=False):
    import plotly.figure_factory as ff
    xs, ys = np.linspace(*xr, density), np.linspace(*yr, density)
    X, Y = np.meshgrid(xs, ys)
    U, V = _fn(P, (x, y))(X, Y), _fn(Q, (x, y))(X, Y)
    mag = np.nanmax(np.hypot(U, V)) or 1.0
    step = (xr[1] - xr[0]) / density
    fig = ff.create_quiver(X, Y, np.nan_to_num(U), np.nan_to_num(V), scale=0.8 * step / mag,
                           line=dict(color=PALETTE[0], width=1.5), name="F")
    return _layout(fig, "Campo vectorial 2D", dark, equal=True)


def plot_vector_field_3d(P, Q, R, x, y, z, xr, yr, zr, density=7, dark=False):
    g = [np.linspace(*r, density) for r in (xr, yr, zr)]
    X, Y, Z = np.meshgrid(*g)
    fig = go.Figure(go.Cone(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                            u=np.nan_to_num(_fn(P, (x, y, z))(X, Y, Z)).ravel(),
                            v=np.nan_to_num(_fn(Q, (x, y, z))(X, Y, Z)).ravel(),
                            w=np.nan_to_num(_fn(R, (x, y, z))(X, Y, Z)).ravel(),
                            sizemode="scaled", sizeref=1.2, colorscale="Turbo"))
    fig.update_layout(scene=dict(aspectmode="data"))
    return _layout(fig, "Campo vectorial 3D", dark)


def plot_region_between(f1, f2, x, a, b, dark=False):
    """Región entre y = f1(x) (abajo) y y = f2(x) (arriba) para a ≤ x ≤ b."""
    xs = np.linspace(a, b, 800)
    y1, y2 = _fn(f1, x)(xs), _fn(f2, x)(xs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=y1, mode="lines", name="y = límite inferior", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(x=xs, y=y2, mode="lines", name="y = límite superior", line=dict(color=PALETTE[1]),
                             fill="tonexty", fillcolor="rgba(79,140,255,0.25)"))
    fig.add_vline(x=a, line_dash="dot")
    fig.add_vline(x=b, line_dash="dot")
    return _layout(fig, "Región de integración", dark)


def plot_polar_region(r1, r2, th, t0, t1, dark=False):
    """Región polar r1(θ) ≤ r ≤ r2(θ), t0 ≤ θ ≤ t1."""
    ts = np.linspace(t0, t1, 600)
    a, b = _fn(r1, th)(ts), _fn(r2, th)(ts)
    xs = np.concatenate([b * np.cos(ts), (a * np.cos(ts))[::-1]])
    ys = np.concatenate([b * np.sin(ts), (a * np.sin(ts))[::-1]])
    fig = go.Figure(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(79,140,255,0.3)", line=dict(color=PALETTE[0]),
                               name="región polar"))
    return _layout(fig, "Región de integración en polares", dark, equal=True)


def plot_points_3d(points, dark=False):
    """points: [(x, y, z, etiqueta)] con proyecciones sobre los ejes."""
    fig = go.Figure()
    for i, (px, py, pz, label) in enumerate(points):
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode="markers+text", text=[label], marker=dict(size=6, color=c), name=label))
        fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, pz], mode="lines", line=dict(color=c, width=4), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[px, px], y=[py, py], z=[0, pz], mode="lines", line=dict(color=c, dash="dash"), showlegend=False))
        fig.add_trace(go.Scatter3d(x=[0, px], y=[0, py], z=[0, 0], mode="lines", line=dict(color=c, dash="dot"), showlegend=False))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return _layout(fig, "Puntos en 3D", dark)
