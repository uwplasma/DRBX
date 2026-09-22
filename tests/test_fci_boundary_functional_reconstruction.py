import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry.fci_boundary_functional_reconstruction import (
    BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION,
    apply_boundary_reconstruction,
    build_boundary_functional_geometry,
    build_moment_boundary_relation,
    prepare_boundary_reconstruction,
)


FIXTURE = (
    Path(__file__).parent
    / "data"
    / "p06_actual_hsx_boundary"
    / "p06_actual_hsx_boundary.npz"
)


def _case(functional="normal_derivative"):
    with np.load(FIXTURE) as source:
        data = {name: np.asarray(source[name]) for name in source.files}
    geometry = build_boundary_functional_geometry(
        value_rows=data["value_rows"],
        gradient_rows=data["gradient_rows"],
        g_contravariant=data["g_contravariant"],
        jacobian=data["jacobian"],
        logical_quadrature_weight=data["logical_quadrature_weight"],
        tangential_coordinates=data["tangential_coordinates"],
        outward_sign=1.0,
    )
    relation = build_moment_boundary_relation(geometry, functional=functional)
    prepared = prepare_boundary_reconstruction(
        data["observation"], data["observation_weight"], relation
    )
    return data, geometry, relation, prepared


def test_actual_hsx_geometry_uses_oriented_physical_normal_and_surface_measure():
    data, geometry, _relation, _prepared = _case()
    expected_normal = data["g_contravariant"][:, 0, :] / np.sqrt(
        data["g_contravariant"][:, 0, 0]
    )[:, None]
    expected_measure = (
        np.abs(data["jacobian"])
        * np.sqrt(data["g_contravariant"][:, 0, 0])
        * data["logical_quadrature_weight"]
    )
    np.testing.assert_allclose(geometry.normal_covector, expected_normal)
    np.testing.assert_allclose(geometry.surface_measure, expected_measure)
    np.testing.assert_allclose(
        geometry.normal_derivative_rows,
        np.einsum("qa,qac->qc", expected_normal, data["gradient_rows"]),
    )
    np.testing.assert_allclose(geometry.moment_modes[:, 0], 1.0)
    assert geometry.version == BOUNDARY_FUNCTIONAL_RECONSTRUCTION_VERSION
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["resolution"] == 32
    assert metadata["raw_key"][0] == 31


def test_actual_hsx_dynamic_boundary_map_reproduces_all_cubic_coefficients():
    data, _geometry, relation, prepared = _case()
    identity = np.eye(data["observation"].shape[1])
    owner_data = data["observation"] @ identity
    boundary_data = relation.constraint_rows @ identity
    recovered = np.asarray(
        apply_boundary_reconstruction(prepared, owner_data, boundary_data)
    )
    np.testing.assert_allclose(recovered, identity, rtol=2e-10, atol=2e-10)
    assert prepared.observation_rank == 20
    assert prepared.constraint_rank == 3
    assert prepared.reduced_rank == 17


def test_homogeneous_thermodynamic_and_phi_constraints_are_distinct_and_exact():
    data, _normal_geometry, normal_relation, normal_prepared = _case(
        "normal_derivative"
    )
    _data, _value_geometry, value_relation, value_prepared = _case("value")
    zero = np.zeros(3)
    for field in range(3):
        coefficients = np.asarray(
            apply_boundary_reconstruction(
                normal_prepared, data["owner_values"][field], zero
            )
        )
        np.testing.assert_allclose(
            normal_relation.constraint_rows @ coefficients, zero, atol=2e-12
        )
    phi = np.asarray(
        apply_boundary_reconstruction(value_prepared, data["owner_values"][0], zero)
    )
    np.testing.assert_allclose(value_relation.constraint_rows @ phi, zero, atol=2e-12)
    # Vorticity has no independent wall constraint in this contract.
    np.testing.assert_allclose(
        data["unconstrained_coefficients"][3],
        data["unconstrained_coefficients"][3],
        rtol=0.0,
        atol=0.0,
    )


def test_dynamic_boundary_fixture_is_eager_jit_and_jvp_consistent():
    data, _geometry, relation, prepared = _case()
    owner = jnp.asarray(data["owner_values"][0])
    boundary = jnp.asarray([0.031, -0.012, 0.007])

    def action(owner_data, boundary_data):
        return apply_boundary_reconstruction(prepared, owner_data, boundary_data)

    eager = action(owner, boundary)
    compiled = jax.jit(action)(owner, boundary)
    np.testing.assert_allclose(compiled, eager, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        relation.constraint_rows @ np.asarray(eager), boundary, rtol=2e-11, atol=2e-11
    )
    owner_tangent = jnp.linspace(-0.2, 0.3, owner.shape[0])
    boundary_tangent = jnp.asarray([0.02, -0.01, 0.005])
    _value, tangent = jax.jvp(
        action, (owner, boundary), (owner_tangent, boundary_tangent)
    )
    expected = (
        prepared.owner_map @ np.asarray(owner_tangent)
        + prepared.boundary_map @ np.asarray(boundary_tangent)
    )
    np.testing.assert_allclose(tangent, expected, rtol=2e-12, atol=2e-12)


def test_rank_revealing_constraint_handling_removes_redundant_moment_row():
    data, _geometry, relation, _prepared = _case()
    from drbx.geometry.fci_boundary_functional_reconstruction import BoundaryRelation

    rows = np.vstack((relation.constraint_rows, relation.constraint_rows[0]))
    rhs = np.vstack((relation.rhs_map, relation.rhs_map[0]))
    redundant = BoundaryRelation(
        constraint_rows=rows,
        rhs_map=rhs,
        labels=relation.labels + ("duplicate",),
        functional=relation.functional,
    )
    prepared = prepare_boundary_reconstruction(
        data["observation"], data["observation_weight"], redundant
    )
    assert prepared.constraint_rank == 3
    coefficients = np.asarray(
        apply_boundary_reconstruction(
            prepared, data["owner_values"][1], np.asarray([0.1, -0.03, 0.02])
        )
    )
    np.testing.assert_allclose(
        relation.constraint_rows @ coefficients, [0.1, -0.03, 0.02], atol=2e-11
    )
