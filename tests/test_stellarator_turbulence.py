"""Gate for stellarator turbulence on closed and open field lines.

A multi-mode seeded four-field run on the rotating ellipse must stay finite and
generate interchange vorticity from the pure-density seed (closed field lines),
and with a toroidal limiter opening the outer flux surfaces the Bohm sheath
sink must drain particles through the open endpoints faster than the closed
run loses them (open field lines). Same seed, same geometry resolution, so the
comparison isolates the open-field-line channel.
"""

from __future__ import annotations

import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_drb.geometry import build_rotating_ellipse_geometry  # noqa: E402

from stellarator_turbulence_case import run_stellarator_turbulence  # noqa: E402

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
