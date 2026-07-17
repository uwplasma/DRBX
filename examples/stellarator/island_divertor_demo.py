"""B8 island divertor: magnetic topology and turbulence draining through it.

Three views of the analytic island-divertor field (sheared iota crossing the
2/3, 3/4, 4/5 rational surfaces with resonant perturbations):

1. a Poincare section -- closed core surfaces, island chains, and the
   stochastic edge, with open field lines (those that reach the wall) in red;
2. the connection-length map at the outboard midplane -- the classic
   island-divertor footprint: infinite (closed) core, finite and structured in
   the stochastic scrape-off layer; and
3. four-field turbulence on this geometry: the multi-mode seed drains through
   the *emergent* open-endpoint masks (no hand-placed limiter), shown by the
   divertor sheath flux and the particle content vs a closed reference.

Run:

    PYTHONPATH=src python examples/stellarator/island_divertor_demo.py

writes ``output/island_divertor/island_divertor.png`` (release-hosted).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from jax_drb.geometry import (  # noqa: E402
    IslandDivertorField,
    build_island_divertor_geometry,
    island_divertor_connection_length,
    island_divertor_field_line_rhs,
)

from stellarator_turbulence_case import run_stellarator_turbulence  # noqa: E402

FIELD = IslandDivertorField()
SHAPE = (16, 24, 12)
STEPS = 30
DT = 2.0e-3
OUTPUT_DIR = Path("output/island_divertor")


def poincare_section(n_lines=36, transits=200, steps=64):
    """Trace lines and collect (x, theta) at every zeta = 0 crossing."""

    rng = np.random.default_rng(0)
    starts = [(x0, rng.uniform(0, 2 * np.pi)) for x0 in np.linspace(0.3, 0.98, n_lines)]
    dz = 2.0 * np.pi / steps
    sections = []
    for x0, th0 in starts:
        x, th = x0, th0
        points = [(x, th % (2 * np.pi))]
        escaped = False
        for _ in range(transits):
            for k in range(steps):
                ze = k * dz

                def f(xv, tv, zv):
                    dx, dt = island_divertor_field_line_rhs(FIELD, xv, tv, zv)
                    return float(dx), float(dt)

                k1 = f(x, th, ze)
                k2 = f(x + dz / 2 * k1[0], th + dz / 2 * k1[1], ze + dz / 2)
                k3 = f(x + dz / 2 * k2[0], th + dz / 2 * k2[1], ze + dz / 2)
                k4 = f(x + dz * k3[0], th + dz * k3[1], ze + dz)
                x += dz / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                th += dz / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                x = max(x, FIELD.x_min)
                if x >= FIELD.x_max:
                    escaped = True
                    break
            if escaped:
                break
            points.append((x, th % (2 * np.pi)))
        sections.append((np.asarray(points), escaped))
    return sections


def connection_length_map(nx=90, ntheta=96, max_transits=40):
    x = jnp.linspace(FIELD.x_min + 0.01, FIELD.x_max - 0.01, nx)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta)
    xx, tt = jnp.meshgrid(x, theta, indexing="ij")
    transits, is_open = island_divertor_connection_length(
        FIELD, xx, tt, jnp.zeros_like(xx), max_transits=max_transits
    )
    return np.asarray(x), np.asarray(theta), np.asarray(transits), np.asarray(is_open)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("tracing Poincare section...")
    sections = poincare_section()
    print("computing connection-length map...")
    x_map, theta_map, transits, is_open = connection_length_map()

    print("running turbulence (closed reference + island divertor)...")
    closed_geometry = build_island_divertor_geometry(SHAPE)
    open_geometry = build_island_divertor_geometry(SHAPE, open_field_line_masks=True, mask_max_transits=25)
    closed = run_stellarator_turbulence(closed_geometry, steps=STEPS, dt=DT, seed=2)
    open_run = run_stellarator_turbulence(open_geometry, steps=STEPS, dt=DT, seed=2, sheath_sink=True)

    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))

    ax = axes[0]
    for points, escaped in sections:
        color = "#d62728" if escaped else "#1f77b4"
        size = 1.2 if escaped else 0.6
        ax.scatter(points[:, 1], points[:, 0], s=size, color=color, linewidths=0)
    ax.set_xlabel("poloidal angle theta"), ax.set_ylabel("radius x")
    ax.set_title("Poincare section: closed core (blue), open lines (red)")
    ax.set_xlim(0, 2 * np.pi), ax.set_ylim(FIELD.x_min, FIELD.x_max)

    ax = axes[1]
    shown = np.where(is_open, transits, np.nan)  # closed cells blank
    mesh = ax.pcolormesh(theta_map, x_map, shown, cmap="magma_r",
                         norm=colors.LogNorm(vmin=1.0, vmax=40.0), shading="auto")
    ax.set_facecolor("#dce6f2")
    fig.colorbar(mesh, ax=ax, label="connection length (transits)")
    ax.set_xlabel("poloidal angle theta"), ax.set_ylabel("radius x")
    ax.set_title("Connection length (closed region shaded blue)")

    ax = axes[2]
    ax.plot(closed.times, closed.particle_content / closed.particle_content[0],
            color="#1f77b4", label="closed reference")
    ax.plot(open_run.times, open_run.particle_content / open_run.particle_content[0],
            color="#d62728", label="island divertor")
    ax.set_xlabel("time"), ax.set_ylabel("particle content (norm.)")
    ax.set_title("Turbulence drains through the emergent divertor")
    ax.legend(fontsize=8), ax.grid(True, ls=":", alpha=0.4)
    twin = ax.twinx()
    twin.plot(open_run.times, open_run.target_flux, color="#d62728", ls=":", alpha=0.7)
    twin.set_ylabel("divertor sheath flux", color="#d62728")

    fig.suptitle("B8 island divertor: island chains, stochastic edge, and turbulence draining to the wall")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "island_divertor.png", dpi=165)
    plt.close(fig)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "open_fraction_edge": float(is_open[x_map > 0.9].mean()),
        "open_fraction_core": float(is_open[x_map < 0.5].mean()),
        "median_edge_connection_transits": float(np.median(transits[(x_map[:, None] > 0.9) & is_open])),
        "closed_content_change": float(closed.particle_content[-1] - closed.particle_content[0]),
        "open_content_change": float(open_run.particle_content[-1] - open_run.particle_content[0]),
        "final_divertor_flux": float(open_run.target_flux[-1]),
    }, indent=2))
    print(f"wrote {OUTPUT_DIR / 'island_divertor.png'} and summary.json")


if __name__ == "__main__":
    main()
