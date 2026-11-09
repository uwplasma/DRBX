"""Offscreen PyVista helpers for the example figures (optional: ``pip install "drbx[viz]"``)."""

from __future__ import annotations

import numpy as np

try:
    import pyvista as pv
except ImportError as error:  # pragma: no cover - optional dependency
    raise ImportError('drbx.viz needs PyVista: pip install "drbx[viz]"') from error

pv.OFF_SCREEN = True


def scene(size=(900, 600), background="white") -> "pv.Plotter":
    """Offscreen plotter with anti-aliasing and a white background."""

    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background(background)
    plotter.enable_anti_aliasing("ssaa")
    return plotter


def add_tubes(plotter, curves, radius=0.01, color="saddlebrown", closed=True, **kwargs):
    """Draw each ``(n, 3)`` curve as a smooth tube (coils, field lines)."""

    for curve in curves:
        points = np.asarray(curve, dtype=float)
        if closed:
            points = np.vstack([points, points[:1]])
        plotter.add_mesh(pv.Spline(points, len(points)).tube(radius=radius), color=color, smooth_shading=True, **kwargs)


def add_surface(plotter, xyz, scalars=None, close_second_axis=True, **kwargs):
    """Draw a structured ``(n1, n2, 3)`` surface, optionally closed periodically along axis 1."""

    xyz = np.asarray(xyz, dtype=float)
    if close_second_axis:
        xyz = np.concatenate([xyz, xyz[:, :1]], axis=1)
        if scalars is not None:
            scalars = np.concatenate([scalars, scalars[:, :1]], axis=1)
    grid = pv.StructuredGrid(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    if scalars is not None:
        grid["s"] = np.asarray(scalars, dtype=float).ravel(order="F")
        kwargs.setdefault("scalars", "s")
    return plotter.add_mesh(grid, smooth_shading=True, **kwargs)


def image(plotter) -> np.ndarray:
    """Render and return the RGB image array."""

    plotter.show(auto_close=False)
    array = plotter.screenshot(return_img=True)
    plotter.close()
    return array
