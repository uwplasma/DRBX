"""Standalone shard-map tests for the native SPMD GMRES layer."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

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
from jax.sharding import PartitionSpec as P

from drbx.native.fci_gmres import (
    SpmdGmresConfig,
    SpmdGmresInfo,
    _spmd_dot,
    _spmd_norm,
    _spmd_remove_weighted_mean,
    _spmd_weighted_mean,
    spmd_gmres_solve,
)
from drbx.native.fci_operators import (
    LocalPerpLaplacianInverseSolver,
    _homogeneous_local_cut_wall_bc,
    _homogeneous_local_face_bc,
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


def _replicated_gmres_info_spec() -> SpmdGmresInfo:
    return SpmdGmresInfo(
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


def test_single_shard_spmd_gmres_solves_identity_inside_shard_map() -> None:
    shape = (4, 3, 2)
    shard_counts = (1, 1, 1)
    halo_width = 1
    domain = _build_domain(shape, halo_width, shard_counts)
    rhs = jnp.arange(1, math.prod(shape) + 1, dtype=jnp.float64).reshape(shape)
    guess = jnp.zeros_like(rhs)
    config = SpmdGmresConfig(tol=1.0e-12, atol=1.0e-12, maxiter=4, restart=4)

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
            return spmd_gmres_solve(
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


def test_single_shard_local_phi_spmd_gmres_reconstructs_manufactured_phi() -> None:
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
    config = SpmdGmresConfig(
        tol=1.0e-9,
        atol=1.0e-9,
        maxiter=100,
        restart=100,
        project_mean_zero=True,
        stagnation_iters=0,
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
                cut_wall_bc=solver._default_cut_wall_bc(),
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


def test_single_shard_local_phi_spmd_gmres_matches_lineax_on_same_local_operator() -> None:
    try:
        import lineax as lx
    except ImportError:
        print("[ SKIP ] lineax is not installed")
        return

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

    config = SpmdGmresConfig(
        tol=1.0e-6,
        atol=1.0e-6,
        maxiter=100,
        restart=100,
        project_mean_zero=True,
        stagnation_iters=0,
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

            cut_wall_bc = solver._default_cut_wall_bc()
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=config.project_mean_zero,
            )
            boundary_source = solver._apply_A(
                jnp.zeros_like(phi_owned),
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=config.project_mean_zero,
            )
            linear_rhs = rhs - boundary_source
            if config.project_mean_zero:
                linear_rhs = _spmd_remove_weighted_mean(linear_rhs, geometry, domain)

            homogeneous_face_bc = _homogeneous_local_face_bc(face_bc)
            homogeneous_cut_wall_bc = _homogeneous_local_cut_wall_bc(cut_wall_bc)

            def apply_A_homogeneous(values):
                return solver._apply_A(
                    values,
                    face_bc=homogeneous_face_bc,
                    cut_wall_bc=homogeneous_cut_wall_bc,
                    project_mean_zero=config.project_mean_zero,
                )

            operator = lx.FunctionLinearOperator(
                apply_A_homogeneous,
                jax.ShapeDtypeStruct(phi_owned.shape, phi_owned.dtype),
            )
            lineax_solver = lx.GMRES(
                rtol=config.tol,
                atol=config.atol,
                restart=config.restart,
                max_steps=config.maxiter,
                stagnation_iters=20,
            )
            lineax_solution = lx.linear_solve(
                operator,
                linear_rhs,
                lineax_solver,
                options={"y0": jnp.zeros_like(phi_owned)},
                throw=True,
            ).value
            if config.project_mean_zero:
                lineax_solution = _spmd_remove_weighted_mean(
                    lineax_solution,
                    geometry,
                    domain,
                )
            lineax_residual = _spmd_norm(
                linear_rhs - apply_A_homogeneous(lineax_solution),
                geometry,
                domain,
            )
            lineax_residual_rel = lineax_residual / jnp.maximum(
                _spmd_norm(linear_rhs, geometry, domain),
                1.0e-30,
            )

            local_solution, local_info = solver(
                rhs,
                phi_guess_owned=jnp.zeros_like(phi_owned),
                return_diagnostics=True,
            )
            return lineax_solution, local_solution, lineax_residual_rel, local_info

        kernel = shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(
                P("x", "y", "z"),
                P("x", "y", "z"),
                P(),
                _replicated_gmres_info_spec(),
            ),
            check_rep=False,
        )
        phi_lineax, phi_local, lineax_residual_rel, local_info = kernel(phi_sharded)

    assert float(lineax_residual_rel) < 1.0e-5
    assert bool(local_info.converged), (
        "Local GMRES did not converge against the shared local Lineax solve: "
        f"steps={int(local_info.num_steps)}, "
        f"final_rel={float(local_info.final_residual_rel_l2):.6e}"
    )
    assert not bool(local_info.failed)
    np.testing.assert_allclose(
        np.asarray(phi_local),
        np.asarray(phi_lineax),
        rtol=5.0e-5,
        atol=5.0e-5,
    )


def test_z_sharded_local_phi_spmd_gmres_matches_lineax_on_same_local_operator() -> None:
    try:
        import lineax as lx
    except ImportError:
        print("[ SKIP ] lineax is not installed")
        return

    shape = (8, 8, 8)
    shard_counts = (1, 1, 4)
    required_devices = math.prod(shard_counts)
    available_devices = len(jax.devices())
    if available_devices < required_devices:
        print(
            "[ SKIP ] "
            f"shard_counts={shard_counts} requires {required_devices} devices, "
            f"but only {available_devices} are available"
        )
        return

    halo_width = 2
    reference_shard_counts = (1, 1, 1)
    reference_domain = _build_domain(shape, halo_width, reference_shard_counts)
    sharded_domain = _build_domain(shape, halo_width, shard_counts)
    sharded_owned_shape = tuple(
        int(size) // int(count)
        for size, count in zip(shape, shard_counts)
    )
    ghost_filler = _build_ghost_filler(halo_width)

    nx, ny, nz = shape
    rho_faces = jnp.linspace(RHO_MIN, 1.0, nx + 1, dtype=jnp.float64)
    theta_faces = jnp.linspace(0.0, 2.0 * jnp.pi, ny + 1, dtype=jnp.float64)
    phi_faces = jnp.linspace(0.0, 2.0 * jnp.pi, nz + 1, dtype=jnp.float64)
    rho = (0.5 * (rho_faces[:-1] + rho_faces[1:]))[:, None, None]
    theta = (0.5 * (theta_faces[:-1] + theta_faces[1:]))[None, :, None]
    toroidal = (0.5 * (phi_faces[:-1] + phi_faces[1:]))[None, None, :]
    phi_exact = _mms_parallel_field(rho, theta, toroidal)

    config = SpmdGmresConfig(
        tol=1.0e-6,
        atol=1.0e-6,
        maxiter=100,
        restart=100,
        project_mean_zero=True,
        stagnation_iters=0,
    )

    with make_mesh_for_shard_counts(reference_shard_counts) as mesh:
        phi_reference_sharded = put_scalar_field_on_mesh(phi_exact, mesh)

        def reference_kernel(phi_owned):
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
                domain=reference_domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                config=config,
            )

            cut_wall_bc = solver._default_cut_wall_bc()
            rhs = solver._apply_A(
                phi_owned,
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=config.project_mean_zero,
            )
            boundary_source = solver._apply_A(
                jnp.zeros_like(phi_owned),
                face_bc=face_bc,
                cut_wall_bc=cut_wall_bc,
                project_mean_zero=config.project_mean_zero,
            )
            linear_rhs = rhs - boundary_source
            if config.project_mean_zero:
                linear_rhs = _spmd_remove_weighted_mean(
                    linear_rhs,
                    geometry,
                    reference_domain,
                )

            homogeneous_face_bc = _homogeneous_local_face_bc(face_bc)
            homogeneous_cut_wall_bc = _homogeneous_local_cut_wall_bc(cut_wall_bc)

            def apply_A_homogeneous(values):
                return solver._apply_A(
                    values,
                    face_bc=homogeneous_face_bc,
                    cut_wall_bc=homogeneous_cut_wall_bc,
                    project_mean_zero=config.project_mean_zero,
                )

            operator = lx.FunctionLinearOperator(
                apply_A_homogeneous,
                jax.ShapeDtypeStruct(phi_owned.shape, phi_owned.dtype),
            )
            lineax_solver = lx.GMRES(
                rtol=config.tol,
                atol=config.atol,
                restart=config.restart,
                max_steps=config.maxiter,
                stagnation_iters=20,
            )
            lineax_solution = lx.linear_solve(
                operator,
                linear_rhs,
                lineax_solver,
                options={"y0": jnp.zeros_like(phi_owned)},
                throw=True,
            ).value
            if config.project_mean_zero:
                lineax_solution = _spmd_remove_weighted_mean(
                    lineax_solution,
                    geometry,
                    reference_domain,
                )
            lineax_residual = _spmd_norm(
                linear_rhs - apply_A_homogeneous(lineax_solution),
                geometry,
                reference_domain,
            )
            lineax_residual_rel = lineax_residual / jnp.maximum(
                _spmd_norm(linear_rhs, geometry, reference_domain),
                1.0e-30,
            )
            return rhs, lineax_solution, lineax_residual_rel

        reference_kernel = shard_map(
            reference_kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(P("x", "y", "z"), P("x", "y", "z"), P()),
            check_rep=False,
        )
        rhs_reference, phi_lineax, lineax_residual_rel = reference_kernel(
            phi_reference_sharded,
        )

    assert float(lineax_residual_rel) < 1.0e-5

    with make_mesh_for_shard_counts(shard_counts) as mesh:
        rhs_sharded = put_scalar_field_on_mesh(rhs_reference, mesh)

        def sharded_kernel(rhs_owned):
            shard_index = tuple(lax.axis_index(name) for name in ("x", "y", "z"))
            geometry = _build_local_geometry(
                sharded_owned_shape,
                halo_width,
                global_shape=shape,
                shard_index=shard_index,
            )
            face_bc = _build_physical_bc(geometry)
            solver = LocalPerpLaplacianInverseSolver(
                geometry=geometry,
                domain=sharded_domain,
                halo_exchange=HaloExchange3D(),
                topology_filler=TopologyHaloFiller3D(
                    rules=(LocalPeriodicTopologyRule3D(),),
                ),
                physical_ghost_filler=ghost_filler,
                face_bc=face_bc,
                config=config,
            )
            return solver(
                rhs_owned,
                phi_guess_owned=jnp.zeros_like(rhs_owned),
                return_diagnostics=True,
            )

        sharded_kernel = shard_map(
            sharded_kernel,
            mesh=mesh,
            in_specs=(P("x", "y", "z"),),
            out_specs=(P("x", "y", "z"), _replicated_gmres_info_spec()),
            check_rep=False,
        )
        phi_sharded, sharded_info = sharded_kernel(rhs_sharded)

    assert bool(sharded_info.converged), (
        "Sharded local GMRES did not converge against the one-shard local "
        "Lineax reference: "
        f"steps={int(sharded_info.num_steps)}, "
        f"final_rel={float(sharded_info.final_residual_rel_l2):.6e}"
    )
    assert not bool(sharded_info.failed)
    np.testing.assert_allclose(
        np.asarray(phi_sharded),
        np.asarray(phi_lineax),
        rtol=5.0e-5,
        atol=5.0e-5,
    )


def main() -> None:
    tests = (
        #test_single_shard_spmd_scalar_algebra_uses_global_weighted_mean,
        #test_single_shard_spmd_gmres_solves_identity_inside_shard_map,
        #test_single_shard_local_phi_spmd_gmres_reconstructs_manufactured_phi,
        #test_single_shard_local_phi_spmd_gmres_matches_lineax_on_same_local_operator,
        test_z_sharded_local_phi_spmd_gmres_matches_lineax_on_same_local_operator,
    )
    print(f"Running {len(tests)} GMRES shard-map tests")
    for test in tests:
        print(f"[ RUN ] {test.__name__}")
        test()
        print(f"[ OK  ] {test.__name__}")


if __name__ == "__main__":
    main()
