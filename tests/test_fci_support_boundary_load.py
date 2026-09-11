"""Independent boundary-load regressions for support-paired FCI actions."""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fci_projected_fine_grid_control_volume import _apply, _setup
from drbx.native.fci_boundaries import (
    BC_DIRICHLET,
    BC_NEUMANN,
    BC_NOFLUX,
    BC_NORMALFLUX,
)
from drbx.native.fci_operators import LocalPerpLaplacianInverseSolver


def test_support_paired_action_does_not_route_through_conservative_operator():
    source = inspect.getsource(LocalPerpLaplacianInverseSolver._apply_A_support_paired)
    assert "_apply_A_conservative" not in source


def test_support_constant_dirichlet_trace_has_zero_action():
    data = _setup(shape=(3, 8, 4))
    _g, _d, _c, _e, _s, _l, cvbc, face, solver, active, _w = data
    value = 0.37
    bc = replace(face, kind_x=face.kind_x.at[-1].set(BC_DIRICHLET), value_x=face.value_x.at[-1].set(value))
    field = np.zeros(active.shape); field[active] = value
    out = _apply(replace(solver, operator_form="support-paired"), bc, cvbc, field)
    np.testing.assert_allclose(out[active], 0.0, atol=2e-9)


def test_support_constant_gradient_and_boundary_trace_are_exact():
    data = _setup(shape=(3, 8, 4))
    _g, _d, _c, _e, _s, _l, _cvbc, face, solver, active, _w = data
    value = 0.37
    field = np.zeros(active.shape)
    field[active] = value
    paired = replace(solver, operator_form="support-paired")
    dirichlet_face = replace(
        face,
        kind_x=face.kind_x.at[-1].set(BC_DIRICHLET),
        value_x=face.value_x.at[-1].set(value),
    )
    gradient = paired._support_paired_gradient(field, face_bc=dirichlet_face)
    np.testing.assert_allclose(gradient, 0.0, atol=2e-9)
    trace = paired._support_paired_boundary_trace(field, face_bc=dirichlet_face)
    x_size = int(np.prod(dirichlet_face.value_x.shape))
    x_trace = trace[:x_size].reshape(dirichlet_face.value_x.shape)
    outer_active = np.asarray(dirichlet_face.mask_x[-1], dtype=bool)
    np.testing.assert_allclose(x_trace[-1][outer_active], value, atol=2e-9)
    packed_mask = np.concatenate(
        tuple(
            np.asarray(mask, dtype=bool).ravel()
            for mask in (
                dirichlet_face.mask_x,
                dirichlet_face.mask_y,
                dirichlet_face.mask_z,
            )
        )
    )
    np.testing.assert_allclose(trace[~packed_mask], 0.0, atol=2e-9)


def test_support_dirichlet_affine_shift_and_linearity():
    data = _setup(shape=(3, 8, 4))
    _g, _d, _c, _e, _s, _l, cvbc, face, solver, active, _w = data
    chosen = replace(solver, operator_form="support-paired")
    bc0 = replace(face, kind_x=face.kind_x.at[-1].set(BC_DIRICHLET), value_x=face.value_x.at[-1].set(0.0))
    bc1 = replace(bc0, value_x=bc0.value_x.at[-1].set(0.25))
    u = np.zeros(active.shape); u[active] = np.arange(np.count_nonzero(active)).reshape(-1) / 100.0
    np.testing.assert_allclose(_apply(chosen, bc1, cvbc, u + 0.25), _apply(chosen, bc0, cvbc, u), atol=2e-9)
    v = np.zeros_like(u); v[active] = 0.13
    np.testing.assert_allclose(_apply(chosen, bc0, cvbc, u + v), _apply(chosen, bc0, cvbc, u) + _apply(chosen, bc0, cvbc, v), atol=2e-9)


def test_support_neumann_flux_balance():
    data = _setup(shape=(3, 8, 4))
    _g, _d, _c, _e, _s, _l, cvbc, face, solver, active, weights = data
    selected = replace(solver, operator_form="support-paired")
    bc = replace(face, kind_x=face.kind_x.at[-1].set(BC_NEUMANN),
                 value_x=face.value_x.at[-1].set(0.19))
    zero = np.zeros(active.shape)
    source = _apply(selected, bc, cvbc, zero)
    homogeneous = _apply(selected, face, cvbc, zero)
    integrated_source = float(np.sum((source - homogeneous)[active] * weights[active]))
    np.testing.assert_allclose(
        integrated_source,
        -0.19 * 4.0 * np.pi**2,
        atol=2e-9,
    )


def test_support_normalflux_and_noflux_boundary_loads():
    data = _setup(shape=(3, 8, 4))
    _g, _d, _c, _e, _s, _l, cvbc, face, solver, active, weights = data
    selected = replace(solver, operator_form="support-paired")
    zero = np.zeros(active.shape)
    for kind, expected in ((BC_NORMALFLUX, -0.19 * 4.0 * np.pi**2), (BC_NOFLUX, 0.0)):
        bc = replace(
            face,
            kind_x=face.kind_x.at[-1].set(kind),
            value_x=face.value_x.at[-1].set(0.19),
        )
        source = _apply(selected, bc, cvbc, zero)
        homogeneous = _apply(selected, face, cvbc, zero)
        integrated_source = float(
            np.sum((source - homogeneous)[active] * weights[active])
        )
        np.testing.assert_allclose(integrated_source, expected, atol=2e-9)
