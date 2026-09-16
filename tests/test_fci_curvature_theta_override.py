"""Focused tests for the explicit periodic-theta curvature face override."""

from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drbx.geometry import (  # noqa: E402
    LocalCurvatureFaceCoefficients3D,
    StencilBuilderContext,
    build_local_conservative_stencil_from_field,
    build_shifted_torus_geometry,
)
from drbx.native.fci_halo import (  # noqa: E402
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
)
from drbx.native.fci_model import inject_owned_field_to_halo  # noqa: E402
from drbx.native.fci_operators import (  # noqa: E402
    local_curvature_production_path_op,
)
from drbx.native.fci_sharding import (  # noqa: E402
    assemble_single_device_local_fci_geometry,
    build_local_fci_geometries,
)


def _case():
    global_geometry = build_shifted_torus_geometry(
        (2, 8, 8), construct_fci_maps=False
    )
    sharded = build_local_fci_geometries(
        global_geometry, (1, 1, 1), halo_width=2
    )
    geometry = assemble_single_device_local_fci_geometry(sharded)
    domain = replace(sharded.domain, mesh_axis_names=(None, None, None))
    topology = TopologyHaloFiller3D(
        (LocalPeriodicTopologyRule3D((False, True, True)),)
    )
    i, j, k = np.indices(geometry.owned_shape)
    theta = 2.0 * np.pi * (j + 0.5) / geometry.owned_shape[1]
    eta = 2.0 * np.pi * (k + 0.5) / geometry.owned_shape[2]
    fields = tuple(
        jnp.asarray(
            2.0
            + 0.13 * np.cos(2.0 * theta + phase)
            + 0.09 * np.sin(eta - phase)
            + 0.01 * i
        )
        for phase in (0.0, 0.31, -0.22, 0.47)
    )
    exchange = HaloExchange3D()
    halos = tuple(
        topology(
            exchange(inject_owned_field_to_halo(field, geometry.layout), domain),
            domain,
        )
        for field in fields
    )
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    stencils = tuple(
        build_local_conservative_stencil_from_field(halo, geometry, context)
        for halo in halos
    )
    coefficients = LocalCurvatureFaceCoefficients3D(
        layout=domain.layout,
        x=jnp.ones(domain.layout.face_control_shape(0)),
        y=jnp.ones(domain.layout.face_control_shape(1)),
        z=jnp.ones(domain.layout.face_control_shape(2)),
    )
    return geometry, domain, stencils, coefficients


def _theta_override(geometry):
    nx, ntheta_faces, neta = geometry.layout.face_control_shape(1)
    face = np.arange(ntheta_faces, dtype=np.float64)[None, :, None]
    eta = np.arange(neta, dtype=np.float64)[None, None, :]
    base = 2.0 + 0.2 * np.cos(2.0 * np.pi * face / ntheta_faces)
    base = base + 0.03 * np.sin(2.0 * np.pi * eta / neta)
    values = np.broadcast_to(base[..., None], (nx, ntheta_faces, neta, 4)).copy()
    values *= np.asarray((1.0, 1.1, 0.9, 0.7))[None, None, None, :]
    values[:, -1] = values[:, 0]
    centered = jnp.asarray(values)
    left = centered + 0.015
    right = centered - 0.015
    left = left.at[:, -1].set(left[:, 0])
    right = right.at[:, -1].set(right[:, 0])
    return centered, left, right


_OMITTED = object()


def _run(case, *, theta_face_states=_OMITTED, return_diagnostics=True, domain=None):
    geometry, original_domain, stencils, coefficients = case
    active_domain = original_domain if domain is None else domain
    kwargs = dict(
        tau=0.7,
        domain=active_domain,
        equilibrium=jnp.asarray((2.0, 2.0, 2.0, 2.0)),
        return_diagnostics=return_diagnostics,
    )
    if theta_face_states is not _OMITTED:
        kwargs["theta_face_states"] = theta_face_states
    return local_curvature_production_path_op(
        stencils,
        geometry,
        coefficients,
        **kwargs,
    )


def test_theta_face_states_none_matches_omitted_baseline():
    case = _case()
    omitted = _run(case)
    explicit_none = _run(case, theta_face_states=None)
    for actual, reference in zip(omitted, explicit_none):
        if isinstance(actual, dict):
            np.testing.assert_allclose(
                np.asarray(actual["directional_residual"]),
                np.asarray(reference["directional_residual"]),
                rtol=0.0,
                atol=0.0,
            )
        else:
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(reference), rtol=0.0, atol=0.0
            )


def test_constant_states_and_coherent_duplicated_theta_seams_are_null():
    geometry, domain, _stencils, coefficients = _case()
    constant = tuple(
        jnp.full(geometry.owned_shape, value, dtype=jnp.float64)
        for value in (2.0, 3.0, 4.0, 0.0)
    )
    context = StencilBuilderContext(layout=domain.layout, domain=domain)
    constant_stencils = tuple(
        build_local_conservative_stencil_from_field(
            jnp.full(geometry.halo_shape, value[0, 0, 0], dtype=jnp.float64),
            geometry,
            context,
        )
        for value in constant
    )
    theta_states = _theta_override(geometry)
    for state in theta_states:
        np.testing.assert_allclose(np.asarray(state[:, 0]), np.asarray(state[:, -1]))
    result = local_curvature_production_path_op(
        constant_stencils,
        geometry,
        coefficients,
        tau=0.7,
        domain=domain,
        equilibrium=jnp.asarray((2.0, 3.0, 4.0, 0.0)),
        theta_face_states=tuple(
            jnp.broadcast_to(
                jnp.asarray((2.0, 3.0, 4.0, 0.0)), state.shape
            )
            for state in theta_states
        ),
    )
    np.testing.assert_allclose(np.asarray(result), 0.0, rtol=0.0, atol=2.0e-12)


def test_nonconstant_theta_override_changes_only_theta_diagnostic():
    case = _case()
    geometry, _domain, _stencils, _coefficients = case
    baseline, baseline_diag = _run(case)
    overridden, overridden_diag = _run(case, theta_face_states=_theta_override(geometry))
    baseline_directional = np.asarray(baseline_diag["directional_residual"])
    overridden_directional = np.asarray(overridden_diag["directional_residual"])
    assert np.max(np.abs(overridden_directional[1] - baseline_directional[1])) > 1.0e-8
    np.testing.assert_allclose(
        overridden_directional[[0, 2]], baseline_directional[[0, 2]],
        rtol=0.0, atol=0.0,
    )
    assert np.max(np.abs(np.asarray(overridden) - np.asarray(baseline))) > 1.0e-8

    constant_theta = tuple(
        jnp.broadcast_to(
            jnp.asarray((2.0, 3.0, 4.0, 0.0)), state.shape
        )
        for state in _theta_override(geometry)
    )
    _constant_result, constant_diag = _run(
        case, theta_face_states=constant_theta
    )
    constant_directional = np.asarray(constant_diag["directional_residual"])
    np.testing.assert_allclose(
        constant_directional[1], 0.0, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        constant_directional[[0, 2]], baseline_directional[[0, 2]],
        rtol=0.0, atol=0.0,
    )


def test_theta_face_states_validate_periodicity_and_native_shapes():
    case = _case()
    geometry, domain, _stencils, _coefficients = case
    valid = _theta_override(geometry)
    with pytest.raises(ValueError, match="native shape"):
        _run(case, theta_face_states=(valid[0][:-1], valid[1], valid[2]))
    nonperiodic = replace(
        domain,
        shard_spec=replace(
            domain.shard_spec, periodic_axes=(False, False, True)
        ),
    )
    with pytest.raises(ValueError, match="periodic theta axis"):
        _run(case, theta_face_states=valid, domain=nonperiodic)
    with pytest.raises(ValueError, match="centered, left, and right"):
        _run(case, theta_face_states=valid[:2])
