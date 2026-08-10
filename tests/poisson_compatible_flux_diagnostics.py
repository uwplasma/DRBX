"""Standalone diagnostics for the experimental compatible-flux bracket.

Run from ``DRBX`` with::

    XDG_CACHE_HOME=/tmp/drbx-cache conda run -n 2D_fci \
        python tests/poisson_compatible_flux_diagnostics.py

This is deliberately a report, not a test: the compatible-flux prototype is
not expected to satisfy every continuous identity exactly on a finite grid.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SRC = ROOT / "src"
for path in (TESTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from axis_regular_operator_support import polar_fixture, scalar_field_halo  # noqa: E402
from test_fci_operators_domain_decomp import _build_domain, _build_local_geometry  # noqa: E402
from drbx.geometry import (  # noqa: E402
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_stencil_from_field,
)
from drbx.native.fci_operators import (  # noqa: E402
    local_poisson_bracket_compatible_flux_op,
    local_poisson_bracket_op,
)


def _owned(value, geometry):
    return jnp.asarray(value)[geometry.layout.owned_slices_cell]


def _weights(geometry):
    spacing = geometry.spacing
    return (
        jnp.asarray(geometry.cell_metric.J_owned, dtype=jnp.float64)
        * jnp.asarray(spacing.dx_owned, dtype=jnp.float64)
        * jnp.asarray(spacing.dy_owned, dtype=jnp.float64)
        * jnp.asarray(spacing.dz_owned, dtype=jnp.float64)
    )


def _weighted_l2(value, weights, mask=None):
    value = jnp.asarray(value, dtype=jnp.float64)
    if mask is None:
        mask = jnp.ones_like(value, dtype=bool)
    selected = jnp.where(mask, value, 0.0)
    w = jnp.where(mask, weights, 0.0)
    return float(jnp.sqrt(jnp.sum(w * selected**2) / jnp.maximum(jnp.sum(w), 1.0e-300)))


def _weighted_max(value, mask=None):
    value = jnp.abs(jnp.asarray(value, dtype=jnp.float64))
    if mask is not None:
        value = jnp.where(mask, value, 0.0)
    return float(jnp.max(value))


def _weighted_integral(value, weights):
    return float(jnp.sum(jnp.asarray(value, dtype=jnp.float64) * weights))


def _polar_fields(r, theta, z):
    """Smooth Cartesian polynomials, represented on the polar grid."""

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    wall = (1.0 - x * x - y * y) ** 2
    a = 0.17
    b = -0.13
    f = wall * (x + a * (x * x - y * y)) + 0.07 * z
    g = wall * (y + b * (2.0 * x * y)) - 0.11 * z
    h = wall * (0.4 * x + 0.23 * (x * x + y * y) * y)

    wx = -4.0 * x * (1.0 - x * x - y * y)
    wy = -4.0 * y * (1.0 - x * x - y * y)
    fx = wx * (x + a * (x * x - y * y)) + wall * (1.0 + 2.0 * a * x)
    fy = wy * (x + a * (x * x - y * y)) + wall * (-2.0 * a * y)
    gx = wx * (y + b * (2.0 * x * y)) + wall * (2.0 * b * y)
    gy = wy * (y + b * (2.0 * x * y)) + wall * (1.0 + 2.0 * b * x)
    exact = fx * gy - fy * gx
    return f, g, h, exact


def _cartesian_fields(x, y, z):
    """Smooth fields for the existing identity-metric Cartesianized fixture."""

    del z
    f = jnp.sin(x) + 0.2 * jnp.cos(y)
    g = jnp.cos(0.7 * x) + 0.3 * jnp.sin(2.0 * y)
    h = jnp.sin(0.6 * x + y)
    exact = 0.6 * jnp.cos(x) * jnp.cos(2.0 * y) - 0.14 * jnp.sin(y) * jnp.sin(0.7 * x)
    return f, g, h, exact


def _identity_geometry(shape):
    geometry = _build_local_geometry(shape, 1, global_shape=shape)

    def identity_metric(metric):
        ones = jnp.ones_like(metric.J_halo)
        zeros = jnp.zeros_like(metric.J_halo)
        return replace(
            metric,
            J_halo=ones,
            g11_halo=ones,
            g22_halo=ones,
            g33_halo=ones,
            g12_halo=zeros,
            g13_halo=zeros,
            g23_halo=zeros,
            g_11_halo=ones,
            g_22_halo=ones,
            g_33_halo=ones,
            g_12_halo=zeros,
            g_13_halo=zeros,
            g_23_halo=zeros,
        )

    def axial_bfield(field):
        b = jnp.zeros_like(field.B_contra_halo).at[..., 2].set(1.0)
        return replace(field, B_contra_halo=b, Bmag_halo=jnp.ones_like(field.Bmag_halo))

    return replace(
        geometry,
        cell_metric=identity_metric(geometry.cell_metric),
        face_metric=replace(
            geometry.face_metric,
            x=identity_metric(geometry.face_metric.x),
            y=identity_metric(geometry.face_metric.y),
            z=identity_metric(geometry.face_metric.z),
        ),
        cell_bfield=axial_bfield(geometry.cell_bfield),
        face_bfield=replace(
            geometry.face_bfield,
            x=axial_bfield(geometry.face_bfield.x),
            y=axial_bfield(geometry.face_bfield.y),
            z=axial_bfield(geometry.face_bfield.z),
        ),
    )


def _evaluate_case(name: str, shape: tuple[int, int, int], polar: bool):
    if polar:
        geometry, domain, context, coordinates, *_ = polar_fixture(shape=shape)
        r, theta, z = coordinates
        f, g, h, exact_halo = _polar_fields(r, theta, z)
        axis_regular_axes = (True, False, False)
    else:
        geometry = _identity_geometry(shape)
        domain = _build_domain(shape, 1)
        context = StencilBuilderContext(layout=geometry.layout, domain=domain)
        x = geometry.grid.x.centers_halo[:, None, None]
        y = geometry.grid.y.centers_halo[None, :, None]
        z = geometry.grid.z.centers_halo[None, None, :]
        x, y, z = jnp.broadcast_arrays(x, y, z)
        f, g, h, exact_halo = _cartesian_fields(x, y, z)
        axis_regular_axes = (False, False, False)

    fs = build_local_conservative_stencil_from_field(f, geometry, context)
    gs = build_local_conservative_stencil_from_field(g, geometry, context)
    hs = build_local_conservative_stencil_from_field(h, geometry, context)
    direct = local_poisson_bracket_op(
        build_local_stencil_from_field(f, geometry, context),
        build_local_stencil_from_field(g, geometry, context),
        geometry,
    )
    compatible = local_poisson_bracket_compatible_flux_op(
        fs, gs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
    )
    reverse = local_poisson_bracket_compatible_flux_op(
        gs, fs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
    )
    constant = jnp.ones_like(f)
    cs = build_local_conservative_stencil_from_field(constant, geometry, context)
    constant_result = local_poisson_bracket_compatible_flux_op(
        fs, cs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
    )
    constant_reverse = local_poisson_bracket_compatible_flux_op(
        cs, fs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
    )

    exact = _owned(exact_halo, geometry)
    weights = _weights(geometry)
    nr = shape[0]
    first = jnp.zeros_like(exact, dtype=bool).at[: min(3, nr)].set(True)
    outer = ~first
    all_mask = jnp.ones_like(exact, dtype=bool)
    # The outer-region mask is empty for very small radial runs; keep its
    # metrics numeric and make that fact explicit in the output.
    def region_metrics(error, mask):
        if int(jnp.sum(mask)) == 0:
            return {"l2": None, "max": None}
        return {"l2": _weighted_l2(error, weights, mask), "max": _weighted_max(error, mask)}

    direct_error = direct - exact
    compatible_error = compatible - exact
    # The public operator does not expose its completed U_s face flux, so the
    # compatible curvature/product-rule residual cannot be formed here.

    def weak_defect(op):
        fg = op
        gh = local_poisson_bracket_compatible_flux_op(
            gs, hs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
        )
        hf = local_poisson_bracket_compatible_flux_op(
            hs, fs, geometry, domain=domain, axis_regular_axes=axis_regular_axes
        )
        h_owned = h[geometry.layout.owned_slices_cell]
        f_owned = f[geometry.layout.owned_slices_cell]
        g_owned = g[geometry.layout.owned_slices_cell]
        return _weighted_integral(h_owned * fg + f_owned * gh + g_owned * hf, weights)

    result = {
        "kind": "case",
        "fixture": name,
        "shape": list(shape),
        "bracket_normalization": "B=1 in both supplied fixtures; values therefore represent bracket/B.",
        "direct_gradient_over_B": {
            "error_l2": _weighted_l2(direct_error, weights),
            "error_max": _weighted_max(direct_error),
            "error_first_3_rings": region_metrics(direct_error, first),
            "error_outer_region": region_metrics(direct_error, outer),
        },
        "compatible_flux": {
            "error_l2": _weighted_l2(compatible_error, weights),
            "error_max": _weighted_max(compatible_error),
            "error_first_3_rings": region_metrics(compatible_error, first),
            "error_outer_region": region_metrics(compatible_error, outer),
            "difference_from_direct_l2": _weighted_l2(compatible - direct, weights),
        },
        "argument_antisymmetry_max": _weighted_max(compatible + reverse),
        "constant_preservation_max": {
            "constant_second_argument": _weighted_max(constant_result),
            "constant_first_argument": _weighted_max(constant_reverse),
        },
        "volume_weighted_global_integral": {
            "direct": _weighted_integral(direct, weights),
            "compatible": _weighted_integral(compatible, weights),
            "exact": _weighted_integral(exact, weights),
        },
        "volume_weighted_weak_ibp_defect": {
            "compatible": weak_defect(compatible),
            "note": "Uses the smooth third field h; physical-boundary terms are included in this reported defect.",
        },
        "curvature_product_rule_residual": {
            "status": "omitted",
            "reason": "local_poisson_bracket_compatible_flux_op does not publicly expose its completed face flux.",
        },
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _refinement_records(results):
    by_fixture = {}
    for result in results:
        by_fixture.setdefault(result["fixture"], []).append(result)
    for fixture, cases in by_fixture.items():
        cases.sort(key=lambda item: tuple(item["shape"]))
        for coarse, fine in zip(cases[:-1], cases[1:]):
            c = coarse["compatible_flux"]["error_first_3_rings"]["l2"]
            f = fine["compatible_flux"]["error_first_3_rings"]["l2"]
            ratio = None if c is None or f is None or f == 0.0 else c / f
            order = None if ratio is None or ratio <= 0.0 else float(np.log2(ratio))
            print(json.dumps({
                "kind": "refinement",
                "fixture": fixture,
                "coarse_shape": coarse["shape"],
                "fine_shape": fine["shape"],
                "compatible_first_3_ring_l2_ratio": ratio,
                "observed_order_log2": order,
                "note": "Adjacent cases are not necessarily a pure one-axis refinement; interpret as an observed diagnostic only.",
            }, sort_keys=True))


def main():
    print("# compatible-flux Poisson bracket diagnostics")
    print("# JSON records below are machine-readable; no identity is asserted.")
    polar_shapes = ((8, 16, 4), (16, 16, 4), (16, 32, 4), (32, 32, 4))
    cartesian_shapes = ((8, 16, 4), (16, 16, 4), (16, 32, 4), (32, 32, 4))
    results = []
    for shape in polar_shapes:
        results.append(_evaluate_case("polar_axis", shape, True))
    for shape in cartesian_shapes:
        results.append(_evaluate_case("cartesian_identity_metric", shape, False))
    _refinement_records(results)


if __name__ == "__main__":
    main()
