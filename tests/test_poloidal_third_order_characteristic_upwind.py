"""Focused contract tests for third-order poloidal characteristic upwinding.

The scalar tests document the periodic third-order symbol.  The remaining
tests use the small polar owner topology, so they exercise the live coupled
owner-space implementation without starting a nonlinear run.
"""

from __future__ import annotations

from dataclasses import MISSING, replace
from pathlib import Path
from types import SimpleNamespace
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
from drbx.geometry.fci_control_volumes import build_polar_angular_agglomeration_geometry
from drbx.native.fci_angular_agglomeration import lower_polar_angular_agglomeration_geometry
from drbx.native.fci_drb_EB_rhs import FciDrbEBRhsParameters, LocalFciDrbEBRhs
from drbx.native.fci_boundaries import LocalControlVolumeBoundaryBC3D
from drbx.native.fci_model import inject_owned_field_to_halo
try:
    from drbx.native.fci_operators import _poloidal_characteristic_third_order_owner_correction
except ImportError:  # The production branch may land after this test module.
    _poloidal_characteristic_third_order_owner_correction = None


def _scalar_third_order_flux_derivative(values: np.ndarray, speed: float) -> np.ndarray:
    """Canonical third-order characteristic flux derivative on a periodic ring."""

    u = np.asarray(values, dtype=float)
    if speed >= 0.0:
        flux = -np.roll(u, 1) / 6.0 + 5.0 * u / 6.0 + np.roll(u, -1) / 3.0
    else:
        flux = -np.roll(u, -1) / 6.0 + 5.0 * u / 6.0 + np.roll(u, 1) / 3.0
    return speed * (flux - np.roll(flux, 1))


def _centered_derivative(values: np.ndarray) -> np.ndarray:
    return 0.5 * (np.roll(values, -1) - np.roll(values, 1))


def test_periodic_scalar_symbol_has_canonical_nyquist_strength_for_both_signs():
    n = 64
    nyquist = (-1.0) ** np.arange(n)
    for speed in (1.0, -1.0):
        rhs = -_scalar_third_order_flux_derivative(nyquist, speed)
        # The reflected negative-speed reconstruction may reverse the phase
        # convention, but its Nyquist damping strength is identical.
        np.testing.assert_allclose(np.abs(rhs), 4.0 / 3.0 * np.abs(nyquist), atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(_centered_derivative(nyquist), 0.0, atol=0.0)

        x = 2.0 * np.pi * np.arange(n) / n
        low = np.sin(x)
        low_rhs = -_scalar_third_order_flux_derivative(low, speed)
        assert np.linalg.norm(low_rhs) < np.linalg.norm(rhs)


def test_periodic_scalar_symbol_is_exact_on_quadratics_away_from_seam():
    x = np.arange(40, dtype=float)
    for values in (np.ones_like(x), x, 2.0 * x * x - 0.3 * x + 4.0):
        target = _scalar_third_order_flux_derivative(values, 1.0)
        centered = _centered_derivative(values)
        np.testing.assert_allclose(target[4:-4], centered[4:-4], atol=5e-12, rtol=0.0)


def test_periodic_scalar_symbol_is_third_order_on_smooth_data():
    errors = []
    for n in (32, 64, 128, 256):
        x = 2.0 * np.pi * np.arange(n) / n
        deriv = _scalar_third_order_flux_derivative(np.sin(x), 1.0) * n / (2.0 * np.pi)
        errors.append(np.max(np.abs(deriv - np.cos(x))))
    assert np.log2(errors[-2] / errors[-1]) > 2.85, (errors,)


def _setup(*, groups=(8, 2, 1), shape=(3, 8, 2)):
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
    coefficients = replace(
        build_local_curvature_face_coefficients(geometry, domain),
        y=jnp.ones((shape[0], shape[1] + 1, shape[2]), dtype=jnp.float64),
    )
    context = StencilBuilderContext(layout=geometry.layout, domain=domain)
    return geometry, domain, context, exchange, scalar_filler, cells, coefficients


def _stencils(field, setup):
    geometry, domain, context, exchange, scalar_filler, _cells, _coefficients = setup
    result = []
    for component in range(4):
        halo = inject_owned_field_to_halo(field[..., component], geometry.layout)
        halo = scalar_filler(exchange(halo, domain), domain)
        result.append(build_local_conservative_stencil_from_field(halo, geometry, context))
    return tuple(result)


def _require_production():
    if _poloidal_characteristic_third_order_owner_correction is None:
        pytest.skip("third-order poloidal production helper is not present yet")


def _apply(field, setup, *, q=None, cells=None, domain=None):
    _require_production()
    geometry, original_domain, _context, _exchange, _scalar, original_cells, coefficients = setup
    q = coefficients.y if q is None else q
    cells = original_cells if cells is None else cells
    domain = original_domain if domain is None else domain
    return _poloidal_characteristic_third_order_owner_correction(
        q, _stencils(field, setup), geometry, cells, domain, tau=1.25
    )


def test_constant_coupled_state_is_exactly_null():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    field = jnp.broadcast_to(
        jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float64),
        geometry.owned_shape + (4,),
    )
    np.testing.assert_array_equal(np.asarray(_apply(field, setup)), 0.0)


def test_live_coupled_matrix_changes_with_face_state():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    theta = np.arange(geometry.owned_shape[1], dtype=float)
    profile = np.sin(2.0 * np.pi * theta / theta.size)
    base = np.broadcast_to(np.asarray((1.0, 1.3, 0.8, 0.0)), geometry.owned_shape + (4,)).copy()
    base[..., 0] += profile[None, :, None]
    shifted = base.copy()
    shifted[..., 0] += 0.75
    first = np.asarray(_apply(jnp.asarray(base), setup))
    second = np.asarray(_apply(jnp.asarray(shifted), setup))
    assert np.linalg.norm(second - first) > 1.0e-9


def test_owner_profile_activates_q2_boundary_faces_but_qny_ring_is_null():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    # Nyquist on each owner lattice: q=8 has one owner and is identically
    # null; q=2 alternates every two fine cells; q=1 alternates every cell.
    theta_mode = np.stack(
        ((-1.0) ** (np.arange(geometry.owned_shape[1]) // 8),
         (-1.0) ** (np.arange(geometry.owned_shape[1]) // 2),
         (-1.0) ** np.arange(geometry.owned_shape[1])),
        axis=0,
    )
    amplitudes = np.asarray((1.0, -0.5, 0.25, 0.75))
    field = theta_mode[:, :, None, None] * amplitudes[None, None, None, :]
    field = jnp.broadcast_to(jnp.asarray(field), geometry.owned_shape + (4,))
    residual = np.asarray(_apply(field, setup))
    np.testing.assert_array_equal(residual[0], 0.0)
    assert np.linalg.norm(residual[1]) > 1.0e-10
    assert np.linalg.norm(residual[2]) > 1.0e-10


def test_owner_space_correction_is_naturally_volume_conservative():
    setup = _setup(groups=(8, 2, 1))
    geometry, _domain, _context, _exchange, _scalar, cells, _coeff = setup
    rng = np.random.default_rng(91)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    field = jnp.asarray(raw, dtype=jnp.float64)
    correction = np.asarray(_apply(field, setup))
    volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    np.testing.assert_allclose(
        np.sum(volume[..., None] * correction, axis=(0, 1, 2)), 0.0, atol=4.0e-11, rtol=0.0
    )


def test_nonuniform_face_fraction_and_closed_face_mask_remain_conservative():
    setup = _setup(groups=(8, 2, 1))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    rng = np.random.default_rng(94)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    field = jnp.asarray(raw, dtype=jnp.float64)
    q = np.array(coefficients.y, copy=True)
    q *= np.linspace(0.25, 1.75, q.shape[1])[None, :, None]
    q[:, 4, :] = 0.0  # a closed periodic face/fraction
    # Change both geometric factors, and close one canonical owner face.  The
    # regular-face object is frozen, so replace it rather than mutating it.
    regular = cells.regular_faces
    regular = replace(
        regular,
        y_area=1.25 * regular.y_area,
        y_area_fraction=0.8 * regular.y_area_fraction,
        y_open_mask=regular.y_open_mask.at[:, 2, :].set(False),
    )
    modified_cells = replace(cells, regular_faces=regular)
    correction = np.asarray(_apply(field, setup, q=jnp.asarray(q), cells=modified_cells))
    baseline = np.asarray(_apply(field, setup, q=jnp.asarray(q)))
    assert not np.allclose(correction, baseline)
    volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    assert np.all(np.isfinite(correction))
    np.testing.assert_allclose(
        np.sum(volume[..., None] * correction, axis=(0, 1, 2)), 0.0, atol=5.0e-11, rtol=0.0
    )


def test_jit_matches_eager_and_eta_slabs_add():
    setup = _setup(groups=(8, 2, 1), shape=(3, 8, 8))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    rng = np.random.default_rng(97)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    field = jnp.asarray(raw, dtype=jnp.float64)
    stencils = _stencils(field, setup)
    eager = _poloidal_characteristic_third_order_owner_correction(
        coefficients.y, stencils, geometry, cells, domain, tau=1.25
    )
    compiled = jax.jit(
        lambda values: _poloidal_characteristic_third_order_owner_correction(
            coefficients.y, values, geometry, cells, domain, tau=1.25
        )
    )(stencils)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), atol=5.0e-11, rtol=0.0)
    pieces = []
    for start in (0, 2, 4, 6):
        slab = jnp.zeros_like(field).at[:, :, start:start + 2, :].set(field[:, :, start:start + 2, :])
        pieces.append(np.asarray(_apply(slab, setup)))
    np.testing.assert_allclose(np.sum(pieces, axis=0), np.asarray(eager), atol=5.0e-11, rtol=0.0)


def test_theta_sharding_is_rejected():
    setup = _setup(groups=(8, 2, 1))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    fake_spec = replace(domain.shard_spec, shard_counts=(1, 2, 1))
    fake_domain = object.__new__(type(domain))
    for name in domain.__dataclass_fields__:
        object.__setattr__(fake_domain, name, getattr(domain, name))
    object.__setattr__(fake_domain, "shard_spec", fake_spec)
    field = jnp.ones(geometry.owned_shape + (4,), dtype=jnp.float64)
    with pytest.raises((NotImplementedError, ValueError), match="theta|shard"):
        _apply(field, setup, domain=fake_domain)


def test_qy_zero_is_exactly_poloidal_null():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    rng = np.random.default_rng(98)
    raw = rng.normal(size=geometry.owned_shape + (4,))
    raw[..., :3] = np.abs(raw[..., :3]) + 1.0
    field = jnp.asarray(raw, dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(_apply(field, setup, q=jnp.zeros_like(setup[-1].y))), 0.0)


def test_input_on_one_q_transition_ring_does_not_leak_radially():
    setup = _setup(groups=(8, 2, 1))
    geometry = setup[0]
    theta = np.arange(geometry.owned_shape[1], dtype=float)
    field = np.zeros(geometry.owned_shape + (4,), dtype=float)
    field[1, :, :, :] = np.sin(2.0 * np.pi * theta / theta.size)[:, None, None]
    field[..., :3] += 1.0
    residual = np.asarray(_apply(jnp.asarray(field), setup))
    np.testing.assert_array_equal(residual[0], 0.0)
    np.testing.assert_array_equal(residual[2], 0.0)
    assert np.linalg.norm(residual[1]) > 0.0


def test_owner_nyquist_has_negative_frozen_H_power_for_q1_and_q2():
    from drbx.native.fci_drb_EB_rhs import background_curvature_characteristic_metric

    setup = _setup(groups=(8, 2, 1))
    geometry, _domain, _context, _exchange, _scalar, cells, _coefficients = setup
    mode = np.stack(
        ((-1.0) ** (np.arange(8) // 8),
         (-1.0) ** (np.arange(8) // 2),
         (-1.0) ** np.arange(8)), axis=0
    )
    amplitudes = np.asarray((1.0, -0.5, 0.25, 0.75))
    field = jnp.asarray(
        np.broadcast_to(
            mode[:, :, None, None] * amplitudes[None, None, None, :],
            geometry.owned_shape + (4,),
        )
    )
    residual = np.asarray(_apply(field, setup))
    volume = np.asarray(cells.cells.raw_volume) / np.maximum(
        np.asarray(geometry.cell_bfield.Bmag_owned), 1.0e-30
    )
    metric = np.asarray(background_curvature_characteristic_metric(jnp.asarray(1.4), 1.25))
    for ring in (1, 2):
        power = np.sum(
            volume[ring] * np.einsum("...i,ij,...j->...", np.asarray(field)[ring], metric, residual[ring])
        )
        assert power < -1.0e-12, (ring, power)


def _assembly_probe(geometry, domain, cells, coefficients, *, radial, poloidal, curvature="disabled"):
    rhs = object.__new__(LocalFciDrbEBRhs)
    attrs = {
        "geometry": geometry,
        "domain": domain,
        "parameters": SimpleNamespace(tau=1.25),
        "control_volume_geometry": cells,
        "curvature_face_coefficients": coefficients,
        "curvature_scheme": curvature,
        "curvature_split_scheme": "legacy",
        "curvature_component_diagnostic_scheme": "directional",
        "curvature_scale": 1.0,
        "curvature_equations": ("density", "Te", "Ti", "vorticity"),
        "ion_temperature_curvature_self_form": "product",
        "curvature_radial_characteristic_scheme": radial,
        "curvature_poloidal_characteristic_scheme": poloidal,
        "curvature_rlp_face_scheme": "projected-fine",
        "curvature_inflow_closure": "none",
    }
    for name, value in attrs.items():
        object.__setattr__(rhs, name, value)
    return rhs


def _assembly_call(rhs, shape, *, directional=False):
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return rhs._curvature_rhs_contributions(
        state_halo=None, face_bc=None, context=None,
        density=zeros + 1.0, Te=zeros + 1.0, Ti=zeros + 1.0,
        bmag=zeros + 1.0, density_safe=zeros + 1.0, tau=1.25,
        Pe_face_bc=None, pressure_face_bc=None, operator_boundary=None,
        Pe_gradient=None, pressure_gradient=None, phi_gradient=None,
        Te_gradient=None, Ti_gradient=None,
        density_conservative_stencil=None, Pe_conservative_stencil=None,
        pressure_conservative_stencil=None, phi_conservative_stencil=None,
        Te_conservative_stencil=None, Ti_conservative_stencil=None,
        vorticity_conservative_stencil=None,
        return_directional_components=directional,
    )


def test_combined_rhs_adds_radial_and_poloidal_componentwise():
    setup = _setup(groups=(8, 2, 1))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    shape = geometry.owned_shape
    radial = tuple(jnp.full(shape, value, dtype=jnp.float64) for value in (1.0, 2.0, 3.0, 4.0))
    poloidal = tuple(jnp.full(shape, value, dtype=jnp.float64) for value in (10.0, 20.0, 30.0, 40.0))
    rhs = _assembly_probe(geometry, domain, cells, coefficients, radial="third-order-upwind", poloidal="third-order-upwind")
    with patch.object(LocalFciDrbEBRhs, "_third_order_radial_characteristic_curvature_correction", return_value=radial), \
         patch.object(LocalFciDrbEBRhs, "_third_order_poloidal_characteristic_curvature_correction", return_value=poloidal):
        result = _assembly_call(rhs, shape)
    for actual, expected in zip(result, (11.0, 22.0, 33.0, 44.0)):
        np.testing.assert_array_equal(np.asarray(actual), expected)


def test_combined_rhs_directional_diagnostic_uses_lane_zero_radial_lane_one_poloidal():
    setup = _setup(groups=(8, 2, 1))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    shape = geometry.owned_shape
    radial = tuple(jnp.full(shape, value, dtype=jnp.float64) for value in (1.0, 2.0, 3.0, 4.0))
    poloidal = tuple(jnp.full(shape, value, dtype=jnp.float64) for value in (10.0, 20.0, 30.0, 40.0))
    rhs = _assembly_probe(geometry, domain, cells, coefficients, radial="third-order-upwind", poloidal="third-order-upwind", curvature="conservative")
    shape2 = (2,) + shape
    directional_base = jnp.zeros(shape2, dtype=jnp.float64)
    # The primitive conservative path is bypassed only for this assembly
    # audit; its zero diagnostic baseline leaves the two correction lanes
    # directly observable.
    state = SimpleNamespace(
        density=jnp.zeros(shape), Te=jnp.zeros(shape), Ti=jnp.zeros(shape), phi=jnp.zeros(shape)
    )
    boundary = SimpleNamespace(
        Pe=None, pressure=None, phi=None, Te=None, Ti=None, Ti_squared=None,
    )
    with patch.object(LocalFciDrbEBRhs, "_primitive_curvature_cv_closures", return_value={}), \
         patch.object(LocalFciDrbEBRhs, "_conservative_curvature_components", return_value=directional_base), \
         patch.object(LocalFciDrbEBRhs, "_third_order_radial_characteristic_curvature_correction", return_value=radial), \
         patch.object(LocalFciDrbEBRhs, "_third_order_poloidal_characteristic_curvature_correction", return_value=poloidal):
        result = rhs._curvature_rhs_contributions(
            state_halo=state, face_bc=boundary, context=None,
            density=jnp.zeros(shape), Te=jnp.zeros(shape), Ti=jnp.zeros(shape),
            bmag=jnp.ones(shape), density_safe=jnp.ones(shape), tau=1.25,
            Pe_face_bc=None, pressure_face_bc=None, operator_boundary=boundary,
            Pe_gradient=None, pressure_gradient=None, phi_gradient=None,
            Te_gradient=None, Ti_gradient=None,
            density_conservative_stencil=None, Pe_conservative_stencil=None,
            pressure_conservative_stencil=None, phi_conservative_stencil=None,
            Te_conservative_stencil=None, Ti_conservative_stencil=None,
            vorticity_conservative_stencil=None,
            return_directional_components=True,
        )
    for actual, rvalue, pvalue in zip(result, radial, poloidal):
        np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(rvalue))
        np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(pvalue))


def _validation_probe(**updates):
    fields = LocalFciDrbEBRhs.__dataclass_fields__
    if "curvature_poloidal_characteristic_scheme" not in fields:
        pytest.skip("poloidal selector is not present yet")
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


def test_poloidal_selector_defaults_legacy_and_rejects_invalid_or_unpaired_mode():
    fields = LocalFciDrbEBRhs.__dataclass_fields__
    if "curvature_poloidal_characteristic_scheme" not in fields:
        pytest.skip("poloidal selector is not present yet")
    assert fields["curvature_poloidal_characteristic_scheme"].default_factory() == "legacy"
    with pytest.raises(ValueError, match="curvature_poloidal_characteristic_scheme"):
        _validation_probe(curvature_poloidal_characteristic_scheme="bogus").__post_init__()
    setup = _setup(groups=(8, 2, 1))
    faces = setup[5].irregular_faces
    empty_boundary = LocalControlVolumeBoundaryBC3D.empty(
        max_rows=faces.max_rows, max_patches=faces.max_patches
    )
    with pytest.raises(ValueError, match="radial"):
        _validation_probe(
            curvature_poloidal_characteristic_scheme="third-order-upwind",
            curvature_radial_characteristic_scheme="legacy",
            curvature_scheme="conservative",
            control_volume_geometry=setup[5],
            control_volume_boundary_bc=empty_boundary,
            geometry=setup[0],
            domain=setup[1],
            axis_regular_axes=(True, False, False),
            poisson_bracket_scheme="compatible-flux",
            curvature_rlp_face_scheme="projected-fine",
        ).__post_init__()


def test_remote_owner_is_rejected_when_metadata_is_present():
    setup = _setup(groups=(8, 2, 1))
    geometry, domain, _context, _exchange, _scalar, cells, coefficients = setup
    if not hasattr(cells.cells, "owner_is_remote"):
        pytest.skip("owner remote metadata is not present")
    # Keep the fixture valid enough to reach the production guard: one merged
    # source points to a remote owner and carries a legal halo address.
    bad_cells = object.__new__(type(cells))
    for name in cells.__dataclass_fields__:
        object.__setattr__(bad_cells, name, getattr(cells, name))
    source = (1, 0, 0)
    bad_cell = object.__new__(type(cells.cells))
    for name in cells.cells.__dataclass_fields__:
        object.__setattr__(bad_cell, name, getattr(cells.cells, name))
    object.__setattr__(bad_cell, "owner_is_remote", cells.cells.owner_is_remote.at[source].set(True))
    object.__setattr__(bad_cells, "cells", bad_cell)
    field = jnp.ones(geometry.owned_shape + (4,), dtype=jnp.float64)
    with pytest.raises((ValueError, NotImplementedError), match="remote|owner"):
        _apply(field, setup, cells=bad_cells)
