"""Gate for stellarator turbulence on closed and open field lines.

A multi-mode seeded four-field run on the rotating ellipse must stay finite and
generate interchange vorticity from the pure-density seed (closed field lines),
and with a toroidal limiter opening the outer flux surfaces the Bohm sheath
sink must drain particles through the open endpoints faster than the closed
run loses them (open field lines). Same seed, same geometry resolution, so the
comparison isolates the open-field-line channel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drbx.geometry import build_rotating_ellipse_geometry
from drbx.native.stellarator_turbulence import run_stellarator_turbulence

jax.config.update("jax_enable_x64", True)

SHAPE = (12, 16, 8)
STEPS = 6
DT = 2.0e-3
LIMITER_RADIUS = 0.6


def test_limiter_opens_only_the_scrape_off_layer() -> None:
    geometry = build_rotating_ellipse_geometry(SHAPE, limiter_radius=LIMITER_RADIUS)
    forward = np.asarray(geometry.maps.forward_boundary)
    backward = np.asarray(geometry.maps.backward_boundary)
    x = np.asarray(geometry.grid.x.centers)
    sol = x > LIMITER_RADIUS
    # Open endpoints exactly on the SOL flux surfaces at the limiter planes.
    assert np.array_equal(forward[:, :, -1], np.broadcast_to(sol[:, None], forward[:, :, -1].shape))
    assert np.array_equal(backward[:, :, 0], np.broadcast_to(sol[:, None], backward[:, :, 0].shape))
    assert not forward[:, :, :-1].any()
    assert not backward[:, :, 1:].any()
    # The core stays closed.
    assert not forward[~sol].any() and not backward[~sol].any()


def test_turbulence_grows_closed_and_drains_open() -> None:
    closed_geometry = build_rotating_ellipse_geometry(SHAPE)
    open_geometry = build_rotating_ellipse_geometry(SHAPE, limiter_radius=LIMITER_RADIUS)

    closed = run_stellarator_turbulence(closed_geometry, steps=STEPS, dt=DT, seed=1)
    open_run = run_stellarator_turbulence(open_geometry, steps=STEPS, dt=DT, seed=1, sheath_sink=True)

    for run in (closed, open_run):
        assert np.all(np.isfinite(run.density_frames))
        assert np.all(np.isfinite(run.omega_frames))
        assert float(run.density_frames.min()) > 0.0

    # Closed: the curvature drive generates interchange vorticity from the
    # pure-density multi-mode seed.
    assert float(np.abs(closed.omega_frames[0]).max()) == 0.0
    assert float(np.abs(closed.omega_frames[-1]).max()) > 0.1

    # Open: the Bohm sheath sink acts on the limiter endpoints (positive target
    # flux) and drains particle content faster than the closed run.
    assert open_run.target_flux[-1] > 0.0
    closed_loss = closed.particle_content[0] - closed.particle_content[-1]
    open_loss = open_run.particle_content[0] - open_run.particle_content[-1]
    assert open_loss > 2.0 * max(closed_loss, 0.0)


def test_plane_block_preconditioner_cuts_gmres_iterations() -> None:
    from solvax import gmres

    from drbx.geometry import ConservativeStencilBuilder, build_conservative_stencil_from_field
    from drbx.native.fci_4_field_rhs import Fci4FieldBlobParameters
    from drbx.native.fci_operators import build_plane_block_preconditioner
    from drbx.native.stellarator_turbulence import build_four_field_phi_solver, radial_dirichlet_face_bc
    from drbx.native.fci_boundaries import CutWallBC3D

    geometry = build_rotating_ellipse_geometry((8, 12, 6))
    solver = build_four_field_phi_solver(
        geometry, Fci4FieldBlobParameters(phi_inversion_tol=1e-8),
        conservative_stencil_builder=ConservativeStencilBuilder(build_conservative_stencil_from_field.build_fn),
        face_projectors=None)
    face_bc = radial_dirichlet_face_bc(geometry)
    apply = lambda v: jnp.ravel(solver._apply_A(jnp.reshape(v, geometry.shape), face_bc, CutWallBC3D.empty(), False))  # noqa: E731
    rhs = jnp.ravel(jnp.asarray(np.random.default_rng(0).standard_normal(geometry.shape)))
    plain = gmres(apply, rhs, restart=64, rtol=1e-8, atol=0.0, max_restarts=4)
    preconditioned = gmres(apply, rhs, precond=solver.plane_preconditioner, restart=64, rtol=1e-8, atol=0.0, max_restarts=4)
    assert int(preconditioned.iterations) <= 8 < int(plain.iterations)
    assert np.allclose(preconditioned.x, plain.x, rtol=0.0, atol=1e-6 * float(jnp.max(jnp.abs(plain.x))))
    assert build_plane_block_preconditioner(lambda v: v, (64, 64, 32)) is None


def test_jit_with_array_arguments_matches_jit_without_constants() -> None:
    """Closed-over arrays become arguments of the compiled step; results match jax.jit."""

    from drbx.native.stellarator_turbulence import jit_with_array_arguments

    table = jax.numpy.asarray(np.linspace(0.0, 1.0, 4096))
    fn = lambda x, y: (jax.numpy.sin(table) * x + table * y, table * y)  # noqa: E731
    x, y = jax.numpy.ones(4096), jax.numpy.full(4096, 2.0)
    hoisted = jit_with_array_arguments(fn, x, y)
    for got, want in zip(hoisted(x, y), jax.jit(fn)(x, y)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    plain = jax.jit(fn).lower(x, y).as_text()
    lowered = hoisted.func.lower(*hoisted.args, x, y).as_text()
    assert len(lowered) < len(plain) / 10
