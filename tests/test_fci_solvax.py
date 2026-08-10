"""Standalone shard-map tests for the SOLVAX-backed SPMD GMRES layer."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from drbx.native.fci_gmres import (
    SolvaxGmresConfig,
    SolvaxGmresInfo,
    _spmd_dot,
    _spmd_norm,
    _spmd_remove_weighted_mean,
    _spmd_weighted_mean,
    solvax_gmres_solve,
)
from drbx.native.fci_operators import (
    AxisCoreLineUPreconditioner3D,
    LocalPerpLaplacianInverseSolver,
    _factor_periodic_block_tridiagonal,
    _principal_perp_laplacian_bands,
    build_local_perp_laplacian_face_projectors,
)
from drbx.native.fci_halo import (
    HaloExchange3D,
    LocalPeriodicTopologyRule3D,
    TopologyHaloFiller3D,
)
from tests.test_fci_operators_domain_decomp import (
    RHO_MIN,
    _build_domain,
    _build_ghost_filler,
    _build_local_geometry,
    _build_physical_bc,
    _mms_parallel_field,
    make_mesh_for_shard_counts,
    put_scalar_field_on_mesh,
)


def _replicated_gmres_info_spec() -> SolvaxGmresInfo:
    return SolvaxGmresInfo(
        num_steps=P(),
        converged=P(),
        failed=P(),
        initial_residual_l2=P(),
        final_residual_l2=P(),
        final_residual_rel_l2=P(),
        rhs_l2=P(),
        projected_rhs_mean=P(),
        projected_rhs_l2=P(),
        phi_is_finite=P(),
        rhs_is_finite=P(),
        guess_is_finite=P(),
    )


def test_single_shard_spmd_scalar_algebra_uses_global_weighted_mean() -> None:
    shape = (4, 3, 2)
    shard_counts = (1, 1, 1)
    halo_width = 1
    domain = _build_domain(shape, halo_width, shard_counts)

    values = jnp.arange(math.prod(shape), dtype=jnp.float64).reshape(shape)
    other = 2.0 - 0.25 * values

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        values_sharded = put_scalar_field_on_mesh(values, mesh)
        other_sharded = put_scalar_field_on_mesh(other, mesh)

        def kernel(values_owned, other_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            mean = _spmd_weighted_mean(values_owned, geometry, domain)
            centered = _spmd_remove_weighted_mean(values_owned, geometry, domain)
            return (
                _spmd_dot(values_owned, other_owned, geometry, domain),
                _spmd_norm(values_owned, geometry, domain),
                mean,
                _spmd_weighted_mean(centered, geometry, domain),
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"), P("x", "y", "z")),
            out_specs=(P(), P(), P(), P()),
            check_rep=False,
        )
        dot, norm, mean, centered_mean = kernel(values_sharded, other_sharded)

    np.testing.assert_allclose(np.asarray(dot), np.asarray(jnp.sum(values * other)))
    np.testing.assert_allclose(np.asarray(norm), np.asarray(jnp.linalg.norm(values)))
    np.testing.assert_allclose(np.asarray(centered_mean), 0.0, atol=1.0e-12)
    assert np.isfinite(np.asarray(mean))


def test_single_shard_solvax_gmres_solves_identity_inside_shard_map() -> None:
    shape = (4, 3, 2)
    shard_counts = (1, 1, 1)
    halo_width = 1
    domain = _build_domain(shape, halo_width, shard_counts)
    rhs = jnp.arange(1, math.prod(shape) + 1, dtype=jnp.float64).reshape(shape)
    guess = jnp.zeros_like(rhs)
    config = SolvaxGmresConfig(tol=1.0e-12, atol=1.0e-12, maxiter=4, restart=4)

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        rhs_sharded = put_scalar_field_on_mesh(rhs, mesh)
        guess_sharded = put_scalar_field_on_mesh(guess, mesh)

        def kernel(rhs_owned, guess_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            return solvax_gmres_solve(
                lambda values: values,
                rhs_owned,
                guess_owned,
                geometry,
                domain,
                config,
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"), P("x", "y", "z")),
            out_specs=(P("x", "y", "z"), _replicated_gmres_info_spec()),
            check_rep=False,
        )
        solution, info = kernel(rhs_sharded, guess_sharded)

    np.testing.assert_allclose(np.asarray(solution), np.asarray(rhs), rtol=1.0e-10, atol=1.0e-10)
    assert bool(info.converged)
    assert not bool(info.failed)
    assert int(info.num_steps) <= 2
    assert float(info.final_residual_l2) < 1.0e-10


def test_z_sharded_solvax_gmres_supports_collective_matvec_and_inner_product() -> None:
    """FGMRES may communicate in both the operator and Arnoldi reductions."""

    shape = (4, 4, 64)
    shard_counts = (1, 1, 4)
    if len(jax.devices()) < 4:
        pytest.skip("requires four JAX devices")
    owned_shape = (shape[0], shape[1], shape[2] // shard_counts[2])
    halo_width = 1
    domain = _build_domain(shape, halo_width, shard_counts)
    z = jnp.arange(shape[2], dtype=jnp.float64)[None, None, :]
    exact = jnp.broadcast_to(
        0.7
        + jnp.sin(2.0 * jnp.pi * z / shape[2])
        + 0.25 * jnp.cos(8.0 * jnp.pi * z / shape[2]),
        shape,
    )
    rhs = 2.1 * exact - jnp.roll(exact, 1, axis=2) - jnp.roll(exact, -1, axis=2)
    forward = ((0, 1), (1, 2), (2, 3), (3, 0))
    backward = ((0, 3), (3, 2), (2, 1), (1, 0))
    config = SolvaxGmresConfig(
        tol=1.0e-11,
        atol=1.0e-12,
        maxiter=16,
        restart=16,
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        rhs_sharded = put_scalar_field_on_mesh(rhs, mesh)

        def kernel(rhs_owned):
            shard_index = tuple(
                lax.axis_index(name) for name in ("x", "y", "z")
            )
            geometry = _build_local_geometry(
                owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )

            def apply_A(values):
                left_ghost = lax.ppermute(values[:, :, -1], "z", forward)
                right_ghost = lax.ppermute(values[:, :, 0], "z", backward)
                left = jnp.concatenate(
                    (left_ghost[:, :, None], values[:, :, :-1]),
                    axis=2,
                )
                right = jnp.concatenate(
                    (values[:, :, 1:], right_ghost[:, :, None]),
                    axis=2,
                )
                return 2.1 * values - left - right

            return solvax_gmres_solve(
                apply_A,
                rhs_owned,
                jnp.zeros_like(rhs_owned),
                geometry,
                domain,
                config,
            )

        mapped = jax.jit(
            shard_map(
                kernel,
                mesh=mesh,
                in_specs=(P("x", "y", "z"),),
                out_specs=(
                    P("x", "y", "z"),
                    _replicated_gmres_info_spec(),
                ),
                check_rep=False,
            )
        )
        solved, info = mapped(rhs_sharded)

    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray(exact),
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    assert bool(info.converged)
    assert not bool(info.failed)
    assert int(info.num_steps) <= 8
    assert float(info.final_residual_rel_l2) < 1.0e-10


def test_z_sharded_axis_core_coarse_solve_matches_replicated_identity() -> None:
    """The small periodic coefficient solve may span eta shards."""

    shape = (4, 4, 8)
    shard_counts = (1, 1, 4)
    if len(jax.devices()) < 4:
        pytest.skip("requires four JAX devices")
    domain = _build_domain(shape, 1, shard_counts)
    degree = 1
    coefficient_count = (degree + 1) * (degree + 2) // 2
    diagonal = jnp.broadcast_to(
        jnp.eye(coefficient_count, dtype=jnp.float64),
        (shape[2], coefficient_count, coefficient_count),
    )
    off_diagonal = jnp.zeros_like(diagonal)
    payload = AxisCoreLineUPreconditioner3D(
        factors=_factor_periodic_block_tridiagonal(
            off_diagonal,
            diagonal,
            off_diagonal,
        ),
        global_shape=shape,
        polynomial_degree=degree,
        observation_ring_count=1,
    )
    coefficients = jnp.arange(
        coefficient_count * shape[2],
        dtype=jnp.float64,
    ).reshape(coefficient_count, shape[2])

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        sharding = NamedSharding(mesh, P(None, "z"))
        sharded = jax.device_put(coefficients, sharding)
        solve = jax.jit(
            shard_map(
                lambda local: payload.solve_coefficients(local, domain),
                mesh=mesh,
                in_specs=(P(None, "z"),),
                out_specs=P(None, "z"),
                check_rep=False,
            )
        )
        solved = solve(sharded)

    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray(coefficients),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_single_shard_local_phi_solvax_gmres_reconstructs_manufactured_phi() -> None:
    shape = (8, 8, 8)
    shard_counts = (1, 1, 1)
    halo_width = 2
    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    toroidal = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    phi_exact = _mms_parallel_field(rho, theta, toroidal)
    config = SolvaxGmresConfig(
        tol=1.0e-9,
        atol=1.0e-9,
        maxiter=100,
        restart=100,
        project_mean_zero=True,
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        phi_sharded = put_scalar_field_on_mesh(phi_exact, mesh)

        def kernel(phi_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _build_physical_bc(geometry)
            solver = LocalPerpLaplacianInverseSolver(
                geometry=geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                config=config,
            )
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                control_volume_boundary_bc=(
                    solver._default_control_volume_boundary_bc()
                ),
                project_mean_zero=True,
            )
            solved, info = solver(
                rhs,
                phi_guess_owned=jnp.zeros_like(phi_owned),
                return_diagnostics=True,
            )
            return (
                solved,
                _spmd_remove_weighted_mean(phi_owned, geometry, domain),
                info,
            )

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(
                P("x", "y", "z"),
                P("x", "y", "z"),
                _replicated_gmres_info_spec(),
            ),
            check_rep=False,
        )
        solved, expected, info = kernel(phi_sharded)

    assert bool(info.converged), (
        "GMRES did not converge: "
        f"steps={int(info.num_steps)}, "
        f"initial_l2={float(info.initial_residual_l2):.6e}, "
        f"final_l2={float(info.final_residual_l2):.6e}, "
        f"final_rel={float(info.final_residual_rel_l2):.6e}"
    )
    assert not bool(info.failed), (
        "GMRES reported failure: "
        f"steps={int(info.num_steps)}, "
        f"initial_l2={float(info.initial_residual_l2):.6e}, "
        f"final_l2={float(info.final_residual_l2):.6e}, "
        f"final_rel={float(info.final_residual_rel_l2):.6e}"
    )
    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray(expected),
        rtol=5.0e-7,
        atol=5.0e-7,
    )
    assert float(info.final_residual_rel_l2) < 1.0e-7


def test_solvax_perp_laplacian_bands_have_owned_finite_geometry_shapes() -> None:
    """The geometry-derived SOLVAX bands are valid local shard arrays."""

    shape = (6, 5, 4)
    shard_counts = (1, 1, 1)
    halo_width = 2
    domain = _build_domain(shape, halo_width, shard_counts)

    with make_mesh_for_shard_counts(shard_counts) as mesh:

        def kernel(_token):
            shard_index = tuple(
                lax.axis_index(name) for name in ("x", "y", "z")
            )
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _build_physical_bc(geometry)
            projectors = build_local_perp_laplacian_face_projectors(
                geometry,
                domain,
            )
            diagonal, lower, upper = _principal_perp_laplacian_bands(
                geometry,
                domain,
                projectors,
                face_bc,
            )
            return diagonal, lower, upper

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=(
                P("x", "y", "z"),
                (P("x", "y", "z"),) * 3,
                (P("x", "y", "z"),) * 3,
            ),
            check_rep=False,
        )
        diagonal, lower, upper = kernel(jnp.asarray(0, dtype=jnp.int32))

    owned_shape = tuple(int(size) for size in diagonal.shape)
    assert owned_shape == shape
    assert np.all(np.isfinite(np.asarray(diagonal)))
    assert np.all(np.asarray(diagonal) > 0.0)
    assert len(lower) == len(upper) == 3
    for lower_band, upper_band in zip(lower, upper):
        assert tuple(lower_band.shape) == owned_shape
        assert tuple(upper_band.shape) == owned_shape
        assert np.all(np.isfinite(np.asarray(lower_band)))
        assert np.all(np.isfinite(np.asarray(upper_band)))


@pytest.mark.parametrize(
    "preconditioner",
    ("none", "jacobi", "line-u", "line-v", "line-uv"),
)
def test_solvax_preconditioner_reconstructs_manufactured_phi_in_shard_map(
    preconditioner: str,
) -> None:
    """Each local SOLVAX preconditioner preserves the manufactured inversion."""

    shape = (6, 6, 6)
    shard_counts = (1, 1, 1)
    halo_width = 2
    domain = _build_domain(shape, halo_width, shard_counts)
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    toroidal = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    phi_exact = _mms_parallel_field(rho, theta, toroidal)
    config = SolvaxGmresConfig(
        tol=1.0e-6,
        atol=1.0e-6,
        maxiter=30,
        restart=30,
        project_mean_zero=True,
        preconditioner=preconditioner,
    )

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        phi_sharded = put_scalar_field_on_mesh(phi_exact, mesh)

        def kernel(phi_owned):
            shard_index = tuple(
                lax.axis_index(name) for name in ("x", "y", "z")
            )
            geometry = _build_local_geometry(
                shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _build_physical_bc(geometry)
            solver = LocalPerpLaplacianInverseSolver(
                geometry=geometry,
                domain=domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                config=config,
            )
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                control_volume_boundary_bc=solver._default_control_volume_boundary_bc(),
                project_mean_zero=True,
            )
            solved, info = solver(
                rhs,
                phi_guess_owned=jnp.zeros_like(phi_owned),
                return_diagnostics=True,
            )
            expected = _spmd_remove_weighted_mean(phi_owned, geometry, domain)
            return solved, expected, info

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(
                P("x", "y", "z"),
                P("x", "y", "z"),
                _replicated_gmres_info_spec(),
            ),
            check_rep=False,
        )
        solved, expected, info = kernel(phi_sharded)

    assert bool(info.converged), (
        f"{preconditioner} did not converge: "
        f"steps={int(info.num_steps)}, "
        f"final_rel={float(info.final_residual_rel_l2):.6e}"
    )
    assert not bool(info.failed)
    print(
        f"preconditioner={preconditioner} iterations={int(info.num_steps)} "
        f"relative_residual={float(info.final_residual_rel_l2):.6e}"
    )
    np.testing.assert_allclose(
        np.asarray(solved),
        np.asarray(expected),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert float(info.final_residual_rel_l2) < 1.0e-5


def main() -> None:
    tests = (
        #test_single_shard_spmd_scalar_algebra_uses_global_weighted_mean,
        #test_single_shard_solvax_gmres_solves_identity_inside_shard_map,
        #test_single_shard_local_phi_solvax_gmres_reconstructs_manufactured_phi,
    )
    print(f"Running {len(tests)} GMRES shard-map tests")
    for test in tests:
        print(f"[ RUN ] {test.__name__}")
        test()
        print(f"[ OK  ] {test.__name__}")


if __name__ == "__main__":
    main()
