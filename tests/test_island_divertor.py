"""B8 gate: the analytic island-divertor field and its emergent open SOL.

The sheared-iota rotating-ellipse field with three resonant perturbations forms
island chains at the rational surfaces and a stochastic edge. The gates pin the
magnetic topology from multi-transit field-line tracing — closed core, finite
connection lengths in the stochastic edge — and that the four-field turbulence
drains through the *emergent* open-endpoint masks (no hand-placed limiter).
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_drb.geometry import (  # noqa: E402
    IslandDivertorField,
    build_island_divertor_geometry,
    island_divertor_connection_length,
)

from stellarator_turbulence_case import run_stellarator_turbulence  # noqa: E402

jax.config.update("jax_enable_x64", True)

FIELD = IslandDivertorField()


def test_field_line_topology_core_closed_edge_open() -> None:
    theta0 = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)

    def scan(x0):
        transits, is_open = island_divertor_connection_length(
            FIELD, jnp.full_like(theta0, x0), theta0, jnp.zeros_like(theta0), max_transits=40
        )
        return np.asarray(transits), np.asarray(is_open)

    # Core (below the first island chain at iota = 2/3, x ~ 0.55): every field
    # line survives 40 toroidal transits -- closed surfaces.
    _, core_open = scan(0.35)
    assert not core_open.any()
    _, mid_open = scan(0.45)
    assert not mid_open.any()

    # Stochastic edge: most lines leave through the outer wall with a finite
    # connection length -- the island-divertor scrape-off layer.
    edge_transits, edge_open = scan(0.95)
    assert edge_open.sum() >= 6
    assert float(edge_transits[edge_open].max()) < 40.0
    # Deeper into the stochastic layer the connection lengths are longer on
    # average than at the wall-adjacent edge.
    layer_transits, layer_open = scan(0.85)
    assert layer_open.any()
    assert float(np.median(layer_transits[layer_open])) >= float(np.median(edge_transits[edge_open]))


def test_geometry_masks_emerge_from_the_field() -> None:
    geometry = build_island_divertor_geometry((16, 16, 8), open_field_line_masks=True, mask_max_transits=20)
    x = np.asarray(geometry.grid.x.centers)
    open_any = np.asarray(geometry.maps.forward_boundary) | np.asarray(geometry.maps.backward_boundary)
    fraction = open_any.mean(axis=(1, 2))

    # Closed core, open stochastic edge, transitional in between.
    assert fraction[x < 0.5].max() == 0.0
    assert fraction[-1] > 0.9
    assert 0.0 < fraction[(x > 0.6) & (x < 0.9)].mean() < 1.0

    # Metric from the shared embedding stays a consistent inverse pair.
    identity = np.einsum("...ik,...kj->...ij",
                         np.asarray(geometry.cell_metric.g_contra),
                         np.asarray(geometry.cell_metric.g_cov))
    assert float(np.abs(identity - np.eye(3)).max()) < 1e-10


def test_turbulence_drains_through_the_island_divertor() -> None:
    shape = (12, 16, 8)
    closed_geometry = build_island_divertor_geometry(shape)  # no masks -> closed
    open_geometry = build_island_divertor_geometry(shape, open_field_line_masks=True, mask_max_transits=20)

    closed = run_stellarator_turbulence(closed_geometry, steps=6, dt=2.0e-3, seed=2)
    open_run = run_stellarator_turbulence(open_geometry, steps=6, dt=2.0e-3, seed=2, sheath_sink=True)

    for run in (closed, open_run):
        assert np.all(np.isfinite(run.density_frames))
        assert float(run.density_frames.min()) > 0.0

    # Interchange vorticity grows from the pure-density seed on the island field.
    assert float(np.abs(closed.omega_frames[0]).max()) == 0.0
    assert float(np.abs(closed.omega_frames[-1]).max()) > 0.05

    # The emergent divertor drains: positive sheath flux on the traced masks and
    # faster particle loss than the closed run with the same seed.
    assert open_run.target_flux[-1] > 0.0
    closed_loss = closed.particle_content[0] - closed.particle_content[-1]
    open_loss = open_run.particle_content[0] - open_run.particle_content[-1]
    assert open_loss > 2.0 * max(closed_loss, 0.0)
