"""Focused diagnostics for the opt-in ordinary-theta characteristic flux."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import (
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_curvature_face_coefficients,
)
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry
from drbx.native.fci_drb_EB_rhs import (
    LocalFciDrbEBRhs,
    background_curvature_characteristic_absolute_matrix,
    background_curvature_characteristic_metric,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.native.fci_operators import _poloidal_characteristic_owner_correction


def _setup(*, groups=(8, 1, 1), shape=(3, 8, 4)):
    geometry, domain, _context, _coords, exchange, scalar_filler, _vector, _flux = polar_fixture(
        shape=shape, halo_width=1
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    # Build the owner topology from the requested profile itself.  Replacing
    # only angular_group_size on a host built with another profile leaves the
    # owner map inconsistent with its ordinary/agglomerated row labels.
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    host = build_polar_angular_agglomeration_geometry(
        u,
        theta,
        eta,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=2,
        angular_group_size=groups,
    )
    cells = lower_polar_angular_agglomeration_geometry(host, geometry)
    coefficients = build_local_curvature_face_coefficients(geometry, domain)
    # The polar chart fixture's compatible y-curvature coefficient is exactly
    # zero (the curvature is radial in this axisymmetric map).  Replace only
    # that diagnostic coefficient by a positive frozen face field so the
    # characteristic-face algebra is exercised; geometry, volumes, B, and
    # owner topology remain the mapped production fixture.
    coefficients = replace(coefficients, y=jnp.ones_like(coefficients.y))
    bmag_face = jnp.asarray(geometry.face_bfield.y.Bmag_owned, dtype=jnp.float64)
    penalty = background_curvature_characteristic_absolute_matrix(bmag_face, 1.25)
    metric = np.asarray(background_curvature_characteristic_metric(jnp.asarray(1.4), 1.25))
    return geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, metric


def _stencils(field, geometry, domain, context, exchange, scalar_filler):
    """Build four coupled conservative stencils with periodic theta/eta halos."""
    fields = []
    for equation in range(4):
        halo = inject_owned_field_to_halo(field[..., equation], geometry.layout)
        halo = exchange(halo, domain)
        halo = scalar_filler(halo, domain)
        fields.append(build_local_conservative_stencil_from_field(halo, geometry, context))
    return tuple(fields)


def _apply(field, setup):
    geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, _metric = setup
    stencils = _stencils(field, geometry, domain, context, exchange, scalar_filler)
    return _poloidal_characteristic_owner_correction(
        coefficients.y, stencils, penalty, geometry, cells, domain, penalty=0.65
    )


def test_constant_state_is_exactly_null():
    setup = _setup()
    geometry = setup[0]
    constant = jnp.broadcast_to(
        jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float64), geometry.owned_shape + (4,)
    )
    residual = _apply(constant, setup)
    np.testing.assert_array_equal(np.asarray(residual), 0.0)
    np.testing.assert_array_equal(np.asarray(residual), np.zeros_like(np.asarray(constant)))


def test_rhs_default_and_explicit_radial_selector_are_exactly_equal():
    """The new selector must preserve the pre-selector radial correction."""
    setup = _setup()
    geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, _metric = setup
    rng = np.random.default_rng(41)
    field = jnp.asarray(rng.normal(size=geometry.owned_shape + (4,)), dtype=jnp.float64)
    stencils = _stencils(field, geometry, domain, context, exchange, scalar_filler)

    def make_rhs(axes):
        rhs = object.__new__(LocalFciDrbEBRhs)
        object.__setattr__(rhs, "geometry", geometry)
        object.__setattr__(rhs, "domain", domain)
        object.__setattr__(rhs, "control_volume_geometry", cells)
        object.__setattr__(rhs, "curvature_face_coefficients", coefficients)
        object.__setattr__(rhs, "curvature_rlp_face_scheme", "fine-glue-characteristic-bulk")
        object.__setattr__(rhs, "curvature_characteristic_axes", axes)
        object.__setattr__(rhs, "curvature_scale", 1.0)
        object.__setattr__(rhs, "curvature_rlp_fine_glue_penalty", 0.65)
        object.__setattr__(rhs, "curvature_rlp_fine_glue_transition_face", None)
        return rhs

    default = make_rhs("legacy")
    radial = make_rhs("radial")
    default_result = default._fine_glue_characteristic_curvature_correction(
        stencils, tau=1.25
    )
    radial_result = radial._fine_glue_characteristic_curvature_correction(
        stencils, tau=1.25
    )
    for lhs, rhs in zip(default_result, radial_result):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_poloidal_penalty_override_scales_only_poloidal_correction():
    setup = _setup()
    geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, _metric = setup
    rng = np.random.default_rng(43)
    field = jnp.asarray(rng.normal(size=geometry.owned_shape + (4,)), dtype=jnp.float64)
    stencils = _stencils(field, geometry, domain, context, exchange, scalar_filler)

    def make_rhs(poloidal_penalty=None):
        rhs = object.__new__(LocalFciDrbEBRhs)
        object.__setattr__(rhs, "geometry", geometry)
        object.__setattr__(rhs, "domain", domain)
        object.__setattr__(rhs, "control_volume_geometry", cells)
        object.__setattr__(rhs, "curvature_face_coefficients", coefficients)
        object.__setattr__(rhs, "curvature_characteristic_axes", "radial-poloidal")
        object.__setattr__(rhs, "curvature_rlp_face_scheme", "fine-glue-characteristic-bulk")
        object.__setattr__(rhs, "curvature_scale", 1.0)
        object.__setattr__(rhs, "curvature_rlp_fine_glue_penalty", 0.65)
        object.__setattr__(rhs, "poloidal_characteristic_penalty", poloidal_penalty)
        return rhs

    inherited = make_rhs()
    overridden = make_rhs(0.2)
    inherited_result = inherited._poloidal_characteristic_curvature_correction(
        stencils, tau=1.25, face_penalty=penalty
    )
    overridden_result = overridden._poloidal_characteristic_curvature_correction(
        stencils, tau=1.25, face_penalty=penalty
    )
    for inherited_value, overridden_value in zip(inherited_result, overridden_result):
        np.testing.assert_allclose(
            np.asarray(overridden_value),
            np.asarray(inherited_value) * (0.2 / 0.65),
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_shared_theta_faces_are_volume_weighted_conservative():
    setup = _setup()
    geometry, _domain, _context, _exchange, _scalar, cells, _coeff, _penalty, _metric = setup
    rng = np.random.default_rng(23)
    field = jnp.asarray(rng.normal(size=geometry.owned_shape + (4,)), dtype=jnp.float64)
    residual = np.asarray(_apply(field, setup))
    natural_volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    np.testing.assert_allclose(
        np.sum(natural_volume[..., None] * residual, axis=(0, 1, 2)),
        0.0,
        atol=3.0e-12,
        rtol=0.0,
    )


def test_theta_nyquist_mode_is_strictly_dissipative_in_frozen_H_metric():
    setup = _setup()
    geometry, _domain, _context, _exchange, _scalar, cells, _coeff, _penalty, metric = setup
    theta_mode = (-1.0) ** np.arange(geometry.owned_shape[1])
    amplitudes = np.asarray((1.0, -0.5, 0.25, 0.75))
    field = jnp.asarray(
        theta_mode[None, :, None, None] * amplitudes[None, None, None, :],
        dtype=jnp.float64,
    )
    field = jnp.broadcast_to(field, geometry.owned_shape + (4,))
    residual = np.asarray(_apply(field, setup))
    volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    power = np.sum(
        volume * np.einsum("...i,ij,...j->...", np.asarray(field), metric, residual)
    )
    assert power < -1.0e-11


def test_periodic_low_mode_is_less_dissipative_than_nyquist():
    """The periodic face symbol should target the terminal theta mode."""
    setup = _setup()
    geometry = setup[0]
    cells = setup[5]
    metric = setup[8]
    amplitudes = np.asarray((1.0, -0.5, 0.25, 0.75))
    volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )

    def mode_power(mode: int) -> float:
        phase = 2.0 * np.pi * mode * np.arange(geometry.owned_shape[1]) / geometry.owned_shape[1]
        field = jnp.asarray(
            np.cos(phase)[None, :, None, None]
            * amplitudes[None, None, None, :],
            dtype=jnp.float64,
        )
        field = jnp.broadcast_to(field, geometry.owned_shape + (4,))
        residual = np.asarray(_apply(field, setup))
        return float(
            np.sum(volume * np.einsum("...i,ij,...j->...", np.asarray(field), metric, residual))
        )

    low_mode_power = mode_power(1)
    nyquist_power = mode_power(geometry.owned_shape[1] // 2)
    assert nyquist_power < -1.0e-11
    # A quarter is deliberately loose relative to the measured periodic
    # stencil symbol, while still catching a correction that damps low modes
    # as strongly as the Nyquist mode.
    assert abs(low_mode_power) < 0.25 * abs(nyquist_power)


def test_agglomerated_radial_rows_are_exactly_masked():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    rng = np.random.default_rng(29)
    field = jnp.asarray(rng.normal(size=geometry.owned_shape + (4,)), dtype=jnp.float64)
    residual = np.asarray(_apply(field, setup))
    np.testing.assert_array_equal(residual[0], 0.0)
    np.testing.assert_array_equal(residual[1], 0.0)
    assert np.linalg.norm(residual[2]) > 0.0


def test_eta_only_variation_has_zero_theta_correction():
    setup = _setup()
    geometry = setup[0]
    eta = np.arange(geometry.owned_shape[2], dtype=np.float64)
    field = jnp.asarray(
        np.sin(2.0 * np.pi * eta / eta.size)[None, None, :, None]
        * np.asarray((1.0, -0.5, 0.25, 0.75))[None, None, None, :],
        dtype=jnp.float64,
    )
    field = jnp.broadcast_to(field, geometry.owned_shape + (4,))
    np.testing.assert_array_equal(np.asarray(_apply(field, setup)), 0.0)


def test_quadratic_theta_trace_jump_is_zero_on_unwrapped_interior_faces():
    """The ordinary interior trace transpose should preserve degree <=2 data."""
    setup = _setup()
    geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, _metric = setup
    theta = np.arange(geometry.owned_shape[1], dtype=np.float64) + 0.5
    amplitudes = np.asarray((1.0, -0.5, 0.25, 0.75))
    field = jnp.asarray(
        theta[None, :, None, None] ** 2 * amplitudes[None, None, None, :],
        dtype=jnp.float64,
    )
    field = jnp.broadcast_to(field, geometry.owned_shape + (4,))
    residual = np.asarray(_apply(field, setup))
    # The periodic seam is not an unwrapped polynomial face.  Centered slopes
    # contaminate the seam-adjacent face jumps f=1 and f=ny-1, and the exact
    # transpose scatters those jumps into theta rows 2 and ny-3 as well.
    # Check only the strictly unwrapped interior rows for this nonperiodic
    # polynomial probe.
    np.testing.assert_allclose(residual[:, 3:-3], 0.0, atol=4.0e-12, rtol=0.0)


def test_jit_and_four_eta_slices_match_the_unsliced_diagnostic():
    setup = _setup(shape=(3, 8, 8))
    geometry, domain, context, exchange, scalar_filler, cells, coefficients, penalty, _metric = setup
    rng = np.random.default_rng(31)
    field = jnp.asarray(rng.normal(size=geometry.owned_shape + (4,)), dtype=jnp.float64)
    stencils = _stencils(field, geometry, domain, context, exchange, scalar_filler)
    compiled = jax.jit(
        lambda values: _poloidal_characteristic_owner_correction(
            coefficients.y, values, penalty, geometry, cells, domain, penalty=0.65
        )
    )(stencils)
    direct = _poloidal_characteristic_owner_correction(
        coefficients.y, stencils, penalty, geometry, cells, domain, penalty=0.65
    )
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(direct), atol=3.0e-12, rtol=0.0)

    # Eta planes are independent for this local theta correction.  Splitting
    # the input into four eta slabs and summing their responses is the local
    # analogue of four-way eta sharding and catches accidental cross-eta
    # gathers without requiring four devices in the test process.
    pieces = []
    for start, stop in ((0, 2), (2, 4), (4, 6), (6, 8)):
        slab = jnp.zeros_like(field).at[:, :, start:stop, :].set(
            field[:, :, start:stop, :]
        )
        pieces.append(np.asarray(_apply(slab, setup)))
    np.testing.assert_allclose(
        np.sum(np.stack(pieces, axis=0), axis=0), np.asarray(direct),
        atol=3.0e-12, rtol=0.0,
    )
