"""End-to-end sharded regression for the EB ARK additive split."""

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import build_local_curvature_coefficients
from drbx.native import (
    HaloExchange3D,
    LocalFciDrbEBRhs,
    LocalPeriodicTopologyRule3D,
    SolvaxGmresConfig,
    TopologyHaloFiller3D,
    assemble_local_fci_geometry,
    build_local_fci_geometries,
    build_local_perp_laplacian_face_projectors,
    make_shard_mesh,
)
from drbx.native.fci_drb_EB_rhs import FciDrbEBState
from jax.sharding import NamedSharding, PartitionSpec as P

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from shifted_torus_eb_mms_data import _mms_exact_state, build_shifted_torus_eb_mms_context  # noqa: E402
from test_mms_shifted_torus_EB_sharded import (  # noqa: E402
    AXIS_REGULAR_AXES,
    HALO_WIDTH,
    PERIODIC_AXES,
    _build_face_bcs,
    _ghost_filler,
)


def test_sharded_imex_split_sums_to_the_production_rhs() -> None:
    """The actual local/sharded split is exact for a supplied stage phi."""

    shape = (4, 6, 4)
    context = build_shifted_torus_eb_mms_context(shape)
    state = _mms_exact_state(context, 0.013)
    mesh = make_shard_mesh((1, 1, 1))
    local = build_local_fci_geometries(
        context.geometry, (1, 1, 1), halo_width=HALO_WIDTH, periodic_axes=PERIODIC_AXES
    )
    partition = P("x", "y", "z")
    sharding = NamedSharding(mesh, partition)
    fields = tuple(
        jax.device_put(jnp.asarray(getattr(state, name), dtype=jnp.float64), sharding)
        for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    )
    cell_fields = jax.device_put(local.cell_fields, sharding)
    ghost = _ghost_filler(HALO_WIDTH)

    def kernel(density, phi, Te, Ti, Vi, Ve, vorticity, local_cell_fields):
        geometry = assemble_local_fci_geometry(local, local_cell_fields)
        domain = local.domain
        rhs = LocalFciDrbEBRhs(
            geometry=geometry,
            domain=domain,
            halo_exchange=HaloExchange3D(),
            topology_filler=TopologyHaloFiller3D(rules=(LocalPeriodicTopologyRule3D(),)),
            physical_ghost_filler=ghost,
            parameters=context.parameters,
            curvature_coefficients_owned=build_local_curvature_coefficients(
                geometry, domain, periodic_axes=PERIODIC_AXES, axis_regular_axes=AXIS_REGULAR_AXES
            ),
            face_projectors=build_local_perp_laplacian_face_projectors(
                geometry, domain, axis_regular_axes=AXIS_REGULAR_AXES
            ),
            gmres_config=SolvaxGmresConfig(
                tol=1.0e-8, atol=1.0e-8, maxiter=20, restart=20,
                acceptance_tol=1.0e-6, acceptance_atol=1.0e-6,
                project_mean_zero=False,
                regularization_epsilon=context.parameters.phi_inversion_regularization,
            ),
            face_bc_builder=_build_face_bcs,
            curvature_scheme="direct",
            curvature_inflow_closure="central",
        )
        stage = FciDrbEBState(density, phi, Te, Ti, Vi, Ve, vorticity)
        full = rhs.evaluate_stage(stage, phi_owned=phi)
        implicit = rhs.evaluate_implicit_rhs(stage, phi_owned=phi)
        explicit = rhs.evaluate_explicit_rhs(stage, phi_owned=phi)
        rhs_without_curvature = replace(rhs, curvature_scale=0.0)
        full_without_curvature = rhs_without_curvature.evaluate_stage(
            stage, phi_owned=phi
        )
        implicit_without_curvature = rhs_without_curvature.evaluate_implicit_rhs(
            stage, phi_owned=phi
        )
        explicit_without_curvature = rhs_without_curvature.evaluate_explicit_rhs(
            stage, phi_owned=phi
        )
        full_curvature = full.axpy(full_without_curvature, scale=-1.0)
        implicit_curvature = implicit.axpy(
            implicit_without_curvature, scale=-1.0
        )
        explicit_curvature = explicit.axpy(
            explicit_without_curvature, scale=-1.0
        )

        # Behavioral placement check: a nonconstant potential perturbation
        # exercises the direct-stencil Poisson brackets.  For density, Te,
        # Ve, and vorticity, the production full-RHS response must now be in
        # F_I, leaving the exact complement independent of phi.
        phi_perturbed = phi + 0.013 * jnp.sin(
            jnp.arange(phi.size, dtype=jnp.float64).reshape(phi.shape)
        )
        perturbed_phi_state = FciDrbEBState(
            density, phi_perturbed, Te, Ti, Vi, Ve, vorticity
        )
        full_phi_perturbed = rhs.evaluate_stage(
            perturbed_phi_state, phi_owned=phi_perturbed
        )
        implicit_phi_perturbed = rhs.evaluate_implicit_rhs(
            perturbed_phi_state, phi_owned=phi_perturbed
        )
        explicit_phi_perturbed = rhs.evaluate_explicit_rhs(
            perturbed_phi_state, phi_owned=phi_perturbed
        )

        # A Ve perturbation exercises the electron parallel self-advection
        # together with the other already-implicit electron couplings.  No
        # Ve-dependent term should remain in the explicit complement for
        # these equations.
        Ve_perturbed = Ve + 0.017 * jnp.cos(
            jnp.arange(Ve.size, dtype=jnp.float64).reshape(Ve.shape)
        )
        perturbed_Ve_state = FciDrbEBState(
            density, phi, Te, Ti, Vi, Ve_perturbed, vorticity
        )
        full_Ve_perturbed = rhs.evaluate_stage(
            perturbed_Ve_state, phi_owned=phi
        )
        implicit_Ve_perturbed = rhs.evaluate_implicit_rhs(
            perturbed_Ve_state, phi_owned=phi
        )
        explicit_Ve_perturbed = rhs.evaluate_explicit_rhs(
            perturbed_Ve_state, phi_owned=phi
        )
        Ti_perturbed = Ti + 0.019 * jnp.sin(
            jnp.arange(Ti.size, dtype=jnp.float64).reshape(Ti.shape)
        )
        perturbed_Ti_state = FciDrbEBState(
            density, phi, Te, Ti_perturbed, Vi, Ve, vorticity
        )
        full_Ti_perturbed = rhs.evaluate_stage(
            perturbed_Ti_state, phi_owned=phi
        )
        implicit_Ti_perturbed = rhs.evaluate_implicit_rhs(
            perturbed_Ti_state, phi_owned=phi
        )
        explicit_Ti_perturbed = rhs.evaluate_explicit_rhs(
            perturbed_Ti_state, phi_owned=phi
        )

        placement_errors = []
        placement_effects = []
        for name in ("density", "Te", "Ve", "vorticity"):
            full_delta = getattr(full_phi_perturbed, name) - getattr(full, name)
            implicit_delta = getattr(implicit_phi_perturbed, name) - getattr(implicit, name)
            explicit_delta = getattr(explicit_phi_perturbed, name) - getattr(explicit, name)
            placement_errors.extend((
                jnp.max(jnp.abs(full_delta - implicit_delta)),
                jnp.max(jnp.abs(explicit_delta)),
            ))
            placement_effects.append(jnp.max(jnp.abs(implicit_delta)))
        # The moved Ti transport/curvature/current response is entirely in
        # F_I; the Ti leaf of the explicit complement is unchanged by Ti.
        ti_full_delta = full_Ti_perturbed.Ti - full.Ti
        ti_implicit_delta = implicit_Ti_perturbed.Ti - implicit.Ti
        ti_explicit_delta = explicit_Ti_perturbed.Ti - explicit.Ti
        placement_errors.extend((
            jnp.max(jnp.abs(ti_full_delta - ti_implicit_delta)),
            jnp.max(jnp.abs(ti_explicit_delta)),
        ))
        placement_effects.append(jnp.max(jnp.abs(ti_implicit_delta)))
        for name in ("density", "Te", "Ve", "vorticity"):
            full_delta = getattr(full_Ve_perturbed, name) - getattr(full, name)
            implicit_delta = getattr(implicit_Ve_perturbed, name) - getattr(implicit, name)
            explicit_delta = getattr(explicit_Ve_perturbed, name) - getattr(explicit, name)
            placement_errors.extend((
                jnp.max(jnp.abs(full_delta - implicit_delta)),
                jnp.max(jnp.abs(explicit_delta)),
            ))
            placement_effects.append(jnp.max(jnp.abs(implicit_delta)))
        errors = jnp.stack((
            jnp.max(jnp.abs(full.density - explicit.density - implicit.density)),
            jnp.max(jnp.abs(full.Te - explicit.Te - implicit.Te)),
            jnp.max(jnp.abs(full.Ti - explicit.Ti - implicit.Ti)),
            jnp.max(jnp.abs(full.Vi - explicit.Vi)),
            jnp.max(jnp.abs(full.Ve - explicit.Ve - implicit.Ve)),
            jnp.max(jnp.abs(full.vorticity - explicit.vorticity - implicit.vorticity)),
            # Production curvature for all implicit differential fields,
            # including Ti, is now in F_I; Vi remains explicit.
            jnp.max(jnp.abs(full_curvature.density - implicit_curvature.density)),
            jnp.max(jnp.abs(full_curvature.Te - implicit_curvature.Te)),
            jnp.max(jnp.abs(full_curvature.vorticity - implicit_curvature.vorticity)),
            jnp.max(jnp.abs(full_curvature.Ti - implicit_curvature.Ti)),
            jnp.max(jnp.abs(explicit_curvature.density)),
            jnp.max(jnp.abs(explicit_curvature.Te)),
            jnp.max(jnp.abs(explicit_curvature.vorticity)),
            jnp.max(jnp.abs(explicit_curvature.Ti)),
            jnp.max(jnp.abs(explicit_curvature.Ve)),
        ))
        return jnp.concatenate((errors, jnp.stack(placement_errors), jnp.stack(placement_effects)))

    compiled = jax.jit(jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(partition,) * 8,
        out_specs=P(),
        check_vma=False,
    ))
    errors = np.asarray(jax.block_until_ready(compiled(*fields, cell_fields)))
    # 15 exact split/curvature identities, followed by 18 placement errors
    # (phi, Ti, and Ve perturbations), then nine nonzero response effects.
    np.testing.assert_allclose(errors[:33], 0.0, atol=2.0e-11, rtol=0.0)
    np.testing.assert_array_less(2.0e-12, errors[33:])
