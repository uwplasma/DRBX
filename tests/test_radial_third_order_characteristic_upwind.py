"""Focused radial third-order characteristic-upwind contract tests.

The tests in this module are deliberately small: no nonlinear run is started.
The scalar symbols are the Fourier analogue of the radial coupled flux, while
the matrix and owner tests exercise the production characteristic algebra.
"""

from __future__ import annotations

from dataclasses import MISSING, replace
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from axis_regular_operator_support import polar_fixture
from drbx.geometry import (
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_local_curvature_face_coefficients,
)
from drbx.native.fci_model import inject_owned_field_to_halo
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.native.fci_drb_EB_rhs import (
    FciDrbEBRhsParameters,
    LocalFciDrbEBRhs,
    background_curvature_characteristic_absolute_matrix,
    background_curvature_characteristic_decomposition,
)
from drbx.native.fci_boundaries import LocalControlVolumeBoundaryBC3D
from drbx.native.fci_operators import (
    _curvature_characteristic_absolute_matrix,
    _radial_characteristic_third_order_owner_correction,
)


def _third_order_flux_derivative(values: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """Third-order upwind derivative used by the radial face-flux analogue.

    For positive speed the face flux is
    ``-u[i-1]/6 + 5*u[i]/6 + u[i+1]/3``.  The negative-speed stencil is
    obtained by reflection.  The derivative is periodic so that the symbol
    can be checked without physical-boundary closures.
    """

    u = np.asarray(values, dtype=float)
    if speed >= 0.0:
        flux = (-np.roll(u, 1) / 6.0 + 5.0 * u / 6.0 + np.roll(u, -1) / 3.0)
    else:
        flux = (-np.roll(u, -1) / 6.0 + 5.0 * u / 6.0 + np.roll(u, 1) / 3.0)
    return speed * (flux - np.roll(flux, 1))


def _centered_derivative(values: np.ndarray) -> np.ndarray:
    u = np.asarray(values, dtype=float)
    return 0.5 * (np.roll(u, -1) - np.roll(u, 1))


def test_background_absolute_matrix_is_exact_for_multiple_b_tau_values():
    """The analytic absolute block must equal its spectral construction."""

    jax.config.update("jax_enable_x64", True)
    for bmag in (0.31, 0.8, 1.4, 2.7):
        for tau in (0.15, 0.7, 1.25, 3.0):
            speeds, projectors = background_curvature_characteristic_decomposition(
                jnp.asarray(bmag), tau
            )
            numerical = sum(
                abs(float(np.asarray(speed))) * np.asarray(projector)
                for speed, projector in zip(speeds, projectors)
                if abs(float(np.asarray(speed))) > 0.0
            )
            analytic = np.asarray(
                background_curvature_characteristic_absolute_matrix(
                    jnp.asarray(bmag), tau
                )
            )
            np.testing.assert_allclose(analytic, numerical, rtol=3e-11, atol=3e-11)


def test_background_decomposition_is_jittable_and_reconstructs_m():
    jax.config.update("jax_enable_x64", True)

    def reconstruct(bmag, tau):
        speeds, projectors = background_curvature_characteristic_decomposition(bmag, tau)
        return sum(speed[..., None, None] * projector for speed, projector in zip(speeds, projectors))

    values = jax.jit(reconstruct)(
        jnp.asarray((0.4, 1.1, 2.0)), jnp.asarray((0.2, 0.9, 2.3))
    )
    expected = np.stack(
        [
            [[2.0, 2.0, 0.0, 0.0], [4.0 / 3.0, 14.0 / 3.0, 0.0, 0.0],
             [4.0 / 3.0, 4.0 / 3.0, -10.0 * t / 3.0, 0.0],
             [2.0 * b * b * (1.0 + t), 2.0 * b * b, 2.0 * t * b * b, 0.0]]
            for b, t in zip((0.4, 1.1, 2.0), (0.2, 0.9, 2.3))
        ]
    )
    np.testing.assert_allclose(np.asarray(values), expected, rtol=3e-11, atol=3e-11)


def _live_curvature_matrix(state, bmag, tau):
    """Reference live M(U) used to audit the analytic absolute helper."""

    n, te, ti, _omega = np.moveaxis(np.asarray(state, dtype=float), -1, 0)
    n = max(float(n), 1.0e-30)
    b2 = float(bmag) ** 2
    return np.asarray(
        [
            [2.0 * te, 2.0 * n, 0.0, 0.0],
            [4.0 * te * te / (3.0 * n), 14.0 * te / 3.0, 0.0, 0.0],
            [4.0 * ti * te / (3.0 * n), 4.0 * ti / 3.0, -10.0 * tau * ti / 3.0, 0.0],
            [2.0 * b2 * (te + tau * ti) / n, 2.0 * b2, 2.0 * tau * b2, 0.0],
        ]
    )


def test_live_absolute_matrix_matches_background_at_unit_state_and_eigendecomposition():
    """Live n=Te=Ti=1 is the established background; random states match eig(|M|)."""

    jax.config.update("jax_enable_x64", True)
    for bmag, tau in ((0.35, 0.2), (0.9, 0.8), (1.4, 1.25), (2.2, 2.5)):
        state = jnp.asarray((1.0, 1.0, 1.0, 0.0), dtype=jnp.float64)
        speeds, projectors = background_curvature_characteristic_decomposition(
            jnp.asarray(bmag), tau
        )
        reconstructed_m = sum(
            float(np.asarray(speed)) * np.asarray(projector)
            for speed, projector in zip(speeds, projectors)
        )
        np.testing.assert_allclose(
            reconstructed_m, _live_curvature_matrix(state, bmag, tau),
            rtol=3e-11, atol=3e-11,
        )
        live = np.asarray(_curvature_characteristic_absolute_matrix(state, bmag, tau))
        background = np.asarray(background_curvature_characteristic_absolute_matrix(bmag, tau))
        np.testing.assert_allclose(live, background, rtol=4e-11, atol=4e-11)

    rng = np.random.default_rng(27)
    for _ in range(12):
        state = np.asarray((
            rng.uniform(0.2, 3.0),
            rng.uniform(0.2, 3.0),
            rng.uniform(0.2, 3.0),
            rng.normal(),
        ))
        bmag = rng.uniform(0.2, 2.5)
        tau = rng.uniform(0.1, 3.0)
        matrix = _live_curvature_matrix(state, bmag, tau)
        eigenvalues, right = np.linalg.eig(matrix)
        numerical = right @ np.diag(np.abs(eigenvalues)) @ np.linalg.inv(right)
        analytic = np.asarray(
            _curvature_characteristic_absolute_matrix(jnp.asarray(state), bmag, tau)
        )
        np.testing.assert_allclose(analytic, np.real_if_close(numerical), rtol=2e-9, atol=2e-9)


def test_third_order_upwind_derivative_has_expected_symbol_and_damping():
    """The radial analogue has the 4/3 Nyquist coefficient, versus 8/1 for JTW."""

    n = 64
    nyquist = (-1.0) ** np.arange(n)
    divergence = _third_order_flux_derivative(nyquist)
    # The curvature equation carries the physical RHS sign ``-Q M`` for
    # positive metric flux Q.  Thus the positive-divergence face operator has
    # a negative RHS eigenvalue on the Nyquist mode.
    rhs = -divergence
    np.testing.assert_allclose(rhs, -4.0 / 3.0 * nyquist, atol=1e-13, rtol=0.0)

    # The centered principal has no real Nyquist damping.  The characteristic
    # correction is therefore the canonical 4/3 coefficient, while the old
    # squared reconstruction has coefficient 8 in this normalized symbol.
    np.testing.assert_allclose(_centered_derivative(nyquist), 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.max(np.abs(rhs)) / np.max(np.abs(nyquist)), 4.0 / 3.0, atol=1e-13
    )
    assert (4.0 / 3.0) < 8.0

    dt = 0.5
    z = -dt * 4.0 / 3.0
    rk4 = 1.0 + z + z * z / 2.0 + z**3 / 6.0 + z**4 / 24.0
    assert abs(rk4) < 1.0


def test_third_order_upwind_derivative_is_exact_on_degree_two_polynomials():
    """The correction, target minus centered principal, vanishes for <=2."""

    # Periodic polynomial values are only used away from the wrap seam.  A
    # sufficiently long interior slice makes the local identity explicit.
    x = np.arange(32, dtype=float)
    for values in (np.ones_like(x), x, 2.0 * x * x - 0.3 * x + 4.0):
        target = _third_order_flux_derivative(values)
        centered = _centered_derivative(values)
        np.testing.assert_allclose(target[3:-3], centered[3:-3], atol=5e-12, rtol=0.0)


def test_third_order_upwind_derivative_is_third_order_on_smooth_periodic_data():
    errors = []
    for n in (32, 64, 128, 256):
        x = 2.0 * np.pi * np.arange(n) / n
        errors.append(np.max(np.abs(_third_order_flux_derivative(np.sin(x)) * n / (2.0 * np.pi) - np.cos(x))))
    observed = np.log2(errors[-2] / errors[-1])
    assert observed > 2.85, (errors, observed)


def _owner_fixture(groups=(8, 4, 4, 2), shape=(4, 8, 2)):
    geometry, domain, _context, _coords, exchange, scalar_filler, *_ = polar_fixture(
        shape=shape, halo_width=1
    )
    u = np.linspace(0.0, 1.0, shape[0] + 1)
    theta = np.linspace(-np.pi, np.pi, shape[1] + 1)
    eta = np.linspace(-np.pi, np.pi, shape[2] + 1)
    host = build_polar_angular_agglomeration_geometry(
        u, theta, eta,
        lambda points: np.maximum(np.asarray(points)[..., 0], 1.0e-14),
        quadrature_order=2,
        angular_group_size=groups,
    )
    cells = lower_polar_angular_agglomeration_geometry(host, geometry)
    coefficients = build_local_curvature_face_coefficients(geometry, domain)
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    return geometry, domain, cells, coefficients, context, exchange, scalar_filler


def _coupled_stencils(value, geometry, context, exchange, scalar_filler):
    result = []
    for field in range(4):
        halo = inject_owned_field_to_halo(value[..., field], geometry.layout)
        halo = exchange(halo, context.domain)
        halo = scalar_filler(halo, context.domain)
        result.append(build_local_conservative_stencil_from_field(halo, geometry, context))
    return tuple(result)


def test_owner_space_radial_correction_is_conservative_at_q_transition():
    geometry, _domain, cells, coefficients, context, exchange, scalar_filler = _owner_fixture()
    rng = np.random.default_rng(13)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    value = jnp.asarray(raw, dtype=jnp.float64)
    q_face = jnp.ones_like(coefficients.x)
    penalty = jnp.broadcast_to(
        jnp.eye(4, dtype=jnp.float64),
        (geometry.owned_shape[0] - 1,) + geometry.owned_shape[1:] + (4, 4),
    )
    stencils = _coupled_stencils(value, geometry, context, exchange, scalar_filler)
    correction = _radial_characteristic_third_order_owner_correction(
        q_face, stencils, geometry, cells, tau=1.25,
    )
    natural_volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    np.testing.assert_allclose(
        np.sum(natural_volume[..., None] * np.asarray(correction), axis=(0, 1, 2)),
        0.0, atol=3e-11, rtol=0.0,
    )

    # The transition is active, whereas a constant coupled state remains in
    # the null space of the incidence scatter.
    # Replace only the radial stencil payload.  This isolates the face-flux
    # identity from the polar fixture's physical radial ghost policy.
    constant = jnp.broadcast_to(jnp.asarray((1.0, 2.0, 3.0, 4.0)), value.shape)
    null_stencils = tuple(
        stencil.replace(
            x=replace(
                stencil.x,
                center=jnp.full_like(stencil.x.center, float(constant[0, 0, 0, field])),
                minus=jnp.full_like(stencil.x.minus, float(constant[0, 0, 0, field])),
                plus=jnp.full_like(stencil.x.plus, float(constant[0, 0, 0, field])),
            )
        )
        for field, stencil in enumerate(stencils)
    )
    null = _radial_characteristic_third_order_owner_correction(
        q_face, null_stencils, geometry, cells, tau=1.25,
    )
    np.testing.assert_allclose(np.asarray(null), 0.0, atol=0.0, rtol=0.0)


def test_owner_correction_jit_matches_eager_and_transition_parts_sum_to_bulk():
    geometry, _domain, cells, coefficients, context, exchange, scalar_filler = _owner_fixture()
    rng = np.random.default_rng(14)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    value = jnp.asarray(raw, dtype=jnp.float64)
    q_face = jnp.ones_like(coefficients.x)
    # Build stencils outside the jitted correction; this mirrors the RHS's
    # already-materialized coupled radial stencils and keeps the test cheap.
    stencils = _coupled_stencils(value, geometry, context, exchange, scalar_filler)
    eager = _radial_characteristic_third_order_owner_correction(
        q_face, stencils, geometry, cells, tau=1.25,
    )
    compiled = jax.jit(
        lambda current: _radial_characteristic_third_order_owner_correction(
            q_face, current, geometry, cells, tau=1.25,
        )
    )(stencils)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), atol=3e-11, rtol=0.0)

def test_legacy_characteristic_mode_is_zero_correction_contract():
    """The legacy selector remains an exact no-op for the new radial path."""

    from drbx.native.fci_drb_EB_rhs import LocalFciDrbEBRhs

    axes = LocalFciDrbEBRhs.__dataclass_fields__["curvature_characteristic_axes"]
    scheme = LocalFciDrbEBRhs.__dataclass_fields__["curvature_radial_characteristic_scheme"]
    assert axes.default_factory() == "legacy"
    assert scheme.default_factory() == "legacy"

    geometry, _domain, _cells, _coefficients, context, exchange, scalar_filler = _owner_fixture()
    rng = np.random.default_rng(15)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    stencils = _coupled_stencils(
        jnp.asarray(raw, dtype=jnp.float64), geometry, context, exchange, scalar_filler
    )
    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "geometry", geometry)
    object.__setattr__(rhs, "curvature_radial_characteristic_scheme", "legacy")
    correction = rhs._third_order_radial_characteristic_curvature_correction(
        stencils, tau=1.25
    )
    for value in correction:
        np.testing.assert_array_equal(np.asarray(value), 0.0)


def _assembly_probe(geometry, *, radial_scheme, rlp_scheme="projected-fine"):
    """Minimal disabled-curvature object that still reaches the common add branch."""

    rhs = object.__new__(LocalFciDrbEBRhs)
    object.__setattr__(rhs, "geometry", geometry)
    object.__setattr__(rhs, "curvature_scheme", "disabled")
    object.__setattr__(rhs, "curvature_split_scheme", "legacy")
    object.__setattr__(rhs, "curvature_component_diagnostic_scheme", "directional")
    object.__setattr__(rhs, "curvature_scale", 1.0)
    object.__setattr__(rhs, "curvature_equations", ("density", "Te", "Ti", "vorticity"))
    object.__setattr__(rhs, "ion_temperature_curvature_self_form", "product")
    object.__setattr__(rhs, "curvature_radial_characteristic_scheme", radial_scheme)
    object.__setattr__(rhs, "curvature_rlp_face_scheme", rlp_scheme)
    return rhs


def _assembly_call(rhs, shape):
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return rhs._curvature_rhs_contributions(
        state_halo=None,
        face_bc=None,
        context=None,
        density=zeros + 1.0,
        Te=zeros + 1.0,
        Ti=zeros + 1.0,
        bmag=zeros + 1.0,
        density_safe=zeros + 1.0,
        tau=1.25,
        Pe_face_bc=None,
        pressure_face_bc=None,
        operator_boundary=None,
        Pe_gradient=None,
        pressure_gradient=None,
        phi_gradient=None,
        Te_gradient=None,
        Ti_gradient=None,
        density_conservative_stencil=None,
        Pe_conservative_stencil=None,
        pressure_conservative_stencil=None,
        phi_conservative_stencil=None,
        Te_conservative_stencil=None,
        Ti_conservative_stencil=None,
        vorticity_conservative_stencil=None,
    )


def test_curvature_rhs_common_add_branch_uses_third_order_correction_and_legacy_is_unchanged():
    geometry, _domain, _cells, _coefficients, _context, _exchange, _scalar = _owner_fixture(
        shape=(3, 8, 2), groups=(8, 4, 2)
    )
    shape = geometry.owned_shape
    expected = tuple(
        jnp.full(shape, value, dtype=jnp.float64)
        for value in (1.25, -2.5, 3.75, -5.0)
    )
    third_order = _assembly_probe(geometry, radial_scheme="third-order-upwind")
    with patch.object(
        LocalFciDrbEBRhs,
        "_third_order_radial_characteristic_curvature_correction",
        return_value=expected,
    ) as mocked:
        assembled = _assembly_call(third_order, shape)
    mocked.assert_called_once()
    for actual, target in zip(assembled, expected):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(target))

    legacy = _assembly_probe(geometry, radial_scheme="legacy")
    legacy_assembled = _assembly_call(legacy, shape)
    for actual in legacy_assembled:
        np.testing.assert_array_equal(np.asarray(actual), 0.0)


def _validation_probe(**updates):
    """Populate defaults enough to exercise LocalFciDrbEBRhs.__post_init__."""

    fields = LocalFciDrbEBRhs.__dataclass_fields__
    probe = object.__new__(LocalFciDrbEBRhs)
    for name, field in fields.items():
        if field.default is not MISSING:
            value = field.default
        elif field.default_factory is not MISSING:
            value = field.default_factory()
        else:
            value = None
        object.__setattr__(probe, name, value)
    object.__setattr__(probe, "parameters", FciDrbEBRhsParameters())
    for name, value in updates.items():
        object.__setattr__(probe, name, value)
    return probe


def test_third_order_configuration_rejects_invalid_or_legacy_combinations():
    with pytest.raises(ValueError, match="curvature_radial_characteristic_scheme"):
        _validation_probe(curvature_radial_characteristic_scheme="bogus").__post_init__()
    with pytest.raises(ValueError, match="curvature_scheme='conservative'"):
        _validation_probe(
            curvature_radial_characteristic_scheme="third-order-upwind",
            curvature_scheme="direct",
        ).__post_init__()

    geometry, domain, cells, _coefficients, _context, _exchange, _scalar = _owner_fixture(
        shape=(3, 8, 2), groups=(8, 4, 2)
    )
    faces = cells.irregular_faces
    empty_boundary = LocalControlVolumeBoundaryBC3D.empty(
        max_rows=faces.max_rows, max_patches=faces.max_patches
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validation_probe(
            geometry=geometry,
            domain=domain,
            control_volume_geometry=cells,
            control_volume_boundary_bc=empty_boundary,
            curvature_scheme="conservative",
            poisson_bracket_scheme="compatible-flux",
            axis_regular_axes=(True, False, False),
            curvature_radial_characteristic_scheme="third-order-upwind",
            curvature_rlp_face_scheme="fine-glue-characteristic",
        ).__post_init__()
