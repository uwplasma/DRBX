"""Rotating-ellipse FCI flagship: a genuinely non-axisymmetric benchmark.

The classical rotating-ellipse (``l = 2``) stellarator is a torus whose
elliptical cross-section rotates as it is followed toroidally, so the metric
depends on all three logical coordinates -- the canonical minimal
non-axisymmetric field. This demo shows, on that geometry:

1. The rotating flux surfaces themselves (nested ellipses whose major axis turns
   with the toroidal angle), built from an analytic embedding whose metric is
   obtained by automatic differentiation -- no hand-derived metric.
2. Second-order convergence of the FCI parallel gradient on this geometry, for
   both the direct ``b^i d_i f`` operator and the traced-field-line operator
   ``grad_parallel_op_fci`` (the FCI-specific path that follows field lines
   between toroidal planes).

Everything is a pure-JAX, ``jit``/``grad``-transparent construction; the same
autodiff that builds the metric makes the geometry differentiable with respect
to its shape (see ``tests/test_rotating_ellipse_fci.py`` for the shape-gradient
gate).

Run:

    PYTHONPATH=src python examples/stellarator/rotating_ellipse_fci_demo.py

writes ``output/rotating_ellipse_fci/`` with a two-panel PNG (rotating flux
surfaces + parallel-operator convergence) and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_drb.geometry import (  # noqa: E402
    FciGeometry3D,
    build_rotating_ellipse_geometry,
    logical_grid_from_axis_vectors,
    rotating_ellipse_position,
)
from jax_drb.native import LocalStencil1D, LocalStencil3D  # noqa: E402
from jax_drb.native.fci_operators import grad_parallel_op_direct, grad_parallel_op_fci  # noqa: E402

X_MIN, X_MAX = 0.2, 1.0
ELONGATION = 0.35
N_FIELD_PERIODS = 1
IOTA = 0.9
C_PHI = 3.0
R0 = 3.0
MMS_M, MMS_N = 2, 1
RESOLUTIONS = (16, 24, 32, 48)
GEOMETRY_KWARGS = dict(
    r0=R0, x_min=X_MIN, x_max=X_MAX, elongation=ELONGATION,
    n_field_periods=N_FIELD_PERIODS, iota=IOTA, c_phi=C_PHI,
)
OUTPUT_DIR = Path("output/rotating_ellipse_fci")


def _mms_field(x, theta, zeta, *, derivatives=False):
    envelope = jnp.sin(jnp.pi * (x - X_MIN) / (X_MAX - X_MIN))
    poloidal = jnp.cos(MMS_M * theta)
    toroidal = jnp.sin(MMS_N * zeta)
    field = envelope * poloidal * toroidal
    if not derivatives:
        return field
    envelope_x = (jnp.pi / (X_MAX - X_MIN)) * jnp.cos(jnp.pi * (x - X_MIN) / (X_MAX - X_MIN))
    return (
        field,
        envelope_x * poloidal * toroidal,
        -MMS_M * envelope * jnp.sin(MMS_M * theta) * toroidal,
        MMS_N * envelope * poloidal * jnp.cos(MMS_N * zeta),
    )


def _sample(x_axis, y_axis, z_axis):
    x, y, z = jnp.broadcast_arrays(
        jnp.asarray(x_axis)[:, None, None],
        jnp.asarray(y_axis)[None, :, None],
        jnp.asarray(z_axis)[None, None, :],
    )
    return _mms_field(x, y, z)


def _expected_grad_parallel(geometry: FciGeometry3D) -> jnp.ndarray:
    logical = logical_grid_from_axis_vectors(
        geometry.grid.x.centers, geometry.grid.y.centers, geometry.grid.z.centers
    )
    _, field_x, field_theta, field_zeta = _mms_field(
        logical[..., 0], logical[..., 1], logical[..., 2], derivatives=True
    )
    df = jnp.stack((field_x, field_theta, field_zeta), axis=-1)
    return jnp.einsum("...i,...i->...", geometry.cell_bfield.b_contra, df)


def _direct_stencil(geometry: FciGeometry3D) -> LocalStencil3D:
    xc, yc, zc = geometry.grid.x.centers, geometry.grid.y.centers, geometry.grid.z.centers
    xf, yf, zf = geometry.grid.x.faces, geometry.grid.y.faces, geometry.grid.z.faces

    def neighbors(centers, faces, periodic):
        if periodic:
            period = faces[-1] - faces[0]
            return (
                jnp.concatenate((centers[-1:] - period, centers[:-1])),
                jnp.concatenate((centers[1:], centers[:1] + period)),
            )
        return (
            jnp.concatenate((jnp.array([2.0 * faces[0] - centers[0]]), centers[:-1])),
            jnp.concatenate((centers[1:], jnp.array([2.0 * faces[-1] - centers[-1]]))),
        )

    xm, xp = neighbors(xc, xf, False)
    ym, yp = neighbors(yc, yf, True)
    zm, zp = neighbors(zc, zf, True)
    shape = geometry.shape
    center = _sample(xc, yc, zc)

    def axis(minus, plus, dmin, dplus):
        return LocalStencil1D(center=center, minus=minus, plus=plus,
                              dx_min=jnp.broadcast_to(dmin, shape), dx_plus=jnp.broadcast_to(dplus, shape))

    return LocalStencil3D(
        x=axis(_sample(xm, yc, zc), _sample(xp, yc, zc), (xc - xm)[:, None, None], (xp - xc)[:, None, None]),
        y=axis(_sample(xc, ym, zc), _sample(xc, yp, zc), (yc - ym)[None, :, None], (yp - yc)[None, :, None]),
        z=axis(_sample(xc, yc, zm), _sample(xc, yc, zp), (zc - zm)[None, None, :], (zp - zc)[None, None, :]),
    )


def _field_line_stencil(geometry: FciGeometry3D) -> LocalStencil1D:
    maps = geometry.maps
    return LocalStencil1D(
        center=_sample(geometry.grid.x.centers, geometry.grid.y.centers, geometry.grid.z.centers),
        minus=_mms_field(maps.backward_endpoint_x, maps.backward_endpoint_y, maps.backward_endpoint_z),
        plus=_mms_field(maps.forward_endpoint_x, maps.forward_endpoint_y, maps.forward_endpoint_z),
        dx_min=jnp.asarray(maps.backward_length, dtype=jnp.float64),
        dx_plus=jnp.asarray(maps.forward_length, dtype=jnp.float64),
    )


def _interior_rms(actual, expected) -> float:
    return float(jnp.sqrt(jnp.mean(((actual - expected)[1:-1, :, :]) ** 2)))


def convergence_study() -> dict:
    direct_errors, fci_errors = [], []
    for resolution in RESOLUTIONS:
        shape = (resolution, resolution, resolution)
        geometry = build_rotating_ellipse_geometry(shape, **GEOMETRY_KWARGS)
        expected = _expected_grad_parallel(geometry)
        direct_errors.append(_interior_rms(grad_parallel_op_direct(_direct_stencil(geometry), geometry), expected))

        traced = build_rotating_ellipse_geometry(shape, construct_fci_maps=True, map_substeps=8, **GEOMETRY_KWARGS)
        fci_errors.append(_interior_rms(grad_parallel_op_fci(_field_line_stencil(traced), traced), expected))
        print(f"res={resolution:3d}: direct rms={direct_errors[-1]:.4e}  fci-traced rms={fci_errors[-1]:.4e}")

    log_res = np.log(np.asarray(RESOLUTIONS, dtype=np.float64))
    direct_order = float(-np.polyfit(log_res, np.log(direct_errors), 1)[0])
    fci_order = float(-np.polyfit(log_res, np.log(fci_errors), 1)[0])
    print(f"convergence order: direct={direct_order:.3f}  fci-traced={fci_order:.3f}")
    return {
        "resolutions": list(RESOLUTIONS),
        "direct_rms_errors": direct_errors,
        "fci_traced_rms_errors": fci_errors,
        "direct_convergence_order": direct_order,
        "fci_traced_convergence_order": fci_order,
    }


def _cross_section(geometry: FciGeometry3D, z_index: int):
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    surfaces = []
    zeta = float(geometry.grid.z.centers[z_index])
    for x in np.asarray(geometry.grid.x.centers)[::2]:
        position = np.asarray(rotating_ellipse_position(
            jnp.asarray(x), jnp.asarray(theta), jnp.asarray(zeta),
            r0=R0, elongation=ELONGATION, n_field_periods=N_FIELD_PERIODS,
        ))
        major_radius = np.hypot(position[:, 0], position[:, 1])
        surfaces.append((major_radius - R0, position[:, 2]))
    return zeta, surfaces


def plot_summary(study: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    geometry = build_rotating_ellipse_geometry((16, 48, 48), **GEOMETRY_KWARGS)
    z_indices = [int(round(frac * (geometry.shape[2] - 1))) for frac in (0.0, 0.25, 0.5, 0.75)]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(z_indices)))
    for color, z_index in zip(colors, z_indices):
        zeta, surfaces = _cross_section(geometry, z_index)
        outer = surfaces[-1]
        axes[0].plot(outer[0], outer[1], color=color, lw=2.0, label=f"zeta = {zeta:.2f}")
        for radial, vertical in surfaces[:-1]:
            axes[0].plot(radial, vertical, color=color, lw=0.7, alpha=0.5)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("R - R0")
    axes[0].set_ylabel("Z")
    axes[0].set_title("Rotating-ellipse flux surfaces vs toroidal angle")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.4)

    res = np.asarray(study["resolutions"], dtype=np.float64)
    axes[1].loglog(res, study["direct_rms_errors"], "o-", label=f"direct, order {study['direct_convergence_order']:.2f}")
    axes[1].loglog(res, study["fci_traced_rms_errors"], "s-", label=f"FCI traced, order {study['fci_traced_convergence_order']:.2f}")
    axes[1].loglog(res, study["direct_rms_errors"][0] * (res / res[0]) ** -2.0, "--", color="gray", label="slope -2")
    axes[1].set_xlabel("resolution")
    axes[1].set_ylabel("parallel-gradient rms error")
    axes[1].set_title("FCI parallel-gradient convergence")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Rotating-ellipse FCI: non-axisymmetric geometry and parallel-operator convergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study = convergence_study()

    geometry = build_rotating_ellipse_geometry((16, 16, 16), **GEOMETRY_KWARGS)
    g_cov = geometry.cell_metric.g_cov
    study["non_axisymmetry_zeta_variation"] = float(
        jnp.max(jnp.std(g_cov, axis=2)) / (jnp.mean(jnp.abs(g_cov)) + 1e-30)
    )

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(study, indent=2))
    plot_summary(study, OUTPUT_DIR / "rotating_ellipse_fci.png")
    print(f"wrote {OUTPUT_DIR / 'rotating_ellipse_fci.png'} and summary.json")


if __name__ == "__main__":
    main()
