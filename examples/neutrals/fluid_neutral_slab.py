"""Fluid neutral penetration into a uniform plasma slab.

Atoms enter a frozen, uniform plasma slab through its wall. In the diffusive limit
the fluid neutral model (n_n, u_n, T_n; advanced fluid closure with a flux limiter)
relaxes to the steady profile exp(-x / lambda_pen) with
lambda_pen = lambda_mfp / sqrt(nu_iz / nu), where nu = nu_cx + nu_iz. The script
starts on that profile, advances the full fluid step (transport, viscosity,
conduction, AMJUEL-2020 reactions) to steady state, fits the decay length and
checks the particle balance (ionization sink = wall influx).

Run:

    PYTHONPATH=src python examples/neutrals/fluid_neutral_slab.py

prints the fitted and analytic penetration lengths and writes
``output/fluid_neutral_slab/penetration.png``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.fci_braginskii.native import fci_neutral_fluid as FL  # noqa: E402
from drbx.fci_braginskii.native import fci_neutrals as F  # noqa: E402

# PARAMETERS
NX, NY, NZ = 96, 4, 4  # cells (the slab is uniform in y and z)
REFERENCE_DENSITY_M3 = 5.0e18
RHO_STAR = 5.0e-4
TE, TI, TN = 0.12, 0.15, 0.15  # normalized temperatures; plasma density is 1
WALL_DENSITY = 0.3  # neutral density at the wall, units of n0
LENGTHS_IN_PENETRATION = 8.0  # slab length in penetration lengths
NU_DT = 0.03  # nu * dt (the explicit friction split error is about nu dt)
STEPS = 121
OUTPUT = Path("output/fluid_neutral_slab")


def grid1d(n, length):
    faces = np.linspace(0.0, length, n + 1)
    return SimpleNamespace(centers=0.5 * (faces[1:] + faces[:-1]), widths=np.diff(faces), faces=faces)


def metric(u, v, w):
    """Identity metric (g_ij = g^ij = delta_ij, J = 1) at the points u x v x w."""

    ones = np.ones((len(u), len(v), len(w)))
    names = {f"g{pre}{i + 1}{j + 1}": float(i == j) * ones for pre in ("", "_") for i in range(3) for j in range(i, 3)}
    return SimpleNamespace(J=ones, **names)


def field(u, v, w):
    """Unit field direction (0.3, 0.5, 0.8): a line crossing every face family."""

    b = np.array([0.3, 0.5, 0.8])
    return SimpleNamespace(b_contra=np.broadcast_to(b / np.linalg.norm(b), (len(u), len(v), len(w), 3)))


def slab(shape, lengths):
    """Duck-typed FCI geometry of a Cartesian box."""

    gx, gy, gz = (grid1d(n, length) for n, length in zip(shape, lengths))
    c, f = (gx.centers, gy.centers, gz.centers), (gx.faces, gy.faces, gz.faces)
    return SimpleNamespace(
        shape=shape, grid=SimpleNamespace(x=gx, y=gy, z=gz),
        cell_metric=metric(*c),
        face_metric=SimpleNamespace(x=metric(f[0], c[1], c[2]), y=metric(c[0], f[1], c[2]), z=metric(c[0], c[1], f[2])),
        cell_bfield=field(*c),
        face_bfield=SimpleNamespace(x=field(f[0], c[1], c[2]), y=field(c[0], f[1], c[2]), z=field(c[0], c[1], f[2])),
    )


params = F.NeutralParameters(reference_density_m3=REFERENCE_DENSITY_M3, reference_temperature_ev=F.reference_temperature_ev(RHO_STAR))
shape = (NX, NY, NZ)
n, Te, Ti, Tn0 = (jnp.full(shape, v) for v in (1.0, TE, TI, TN))
nu_cx, nu_iz = (float(x[0, 0, 0]) for x in F.collision_frequencies(n, Te, Ti, params, Tn0))
lam = float(np.sqrt(TN / (nu_iz * (nu_cx + nu_iz))))  # analytic penetration length
length = LENGTHS_IN_PENETRATION * lam
geometry = slab(shape, (length, 1.0, 1.0))
grid, x = F.build_neutral_grid(geometry), np.asarray(geometry.grid.x.centers)
dt = NU_DT / (nu_cx + nu_iz)
flux = WALL_DENSITY * nu_iz * lam  # steady wall influx
source = jnp.zeros(shape).at[-1].set(flux / (length / NX))
profile = WALL_DENSITY * np.exp(-(length - x) / lam)[:, None, None] * np.ones(shape)

step = jax.jit(lambda nn, un, tn: FL.neutral_fluid_step(n, jnp.zeros(shape), Te, Ti, nn, un, tn, dt, grid, params, volume_source=source))
nn, un, tn = jnp.asarray(profile), jnp.zeros(shape), Tn0
for k in range(STEPS):
    previous = nn
    nn, un, tn = step(nn, un, tn)[4:7]
    if (k + 1) % 20 == 0:
        print(f"step {k + 1}/{STEPS}  max |dN/N| per step = {float(jnp.max(jnp.abs(nn - previous) / previous)):.2e}", flush=True)

window = (x > length - 4 * lam) & (x < length - 0.5 * lam)
line = np.asarray(nn[:, 0, 0])
fitted = 1.0 / np.polyfit(x[window], np.log(line[window]), 1)[0]
balance = nu_iz * float(jnp.sum(nn[:, 0, 0])) * length / NX / flux
print(f"nu_cx = {nu_cx:.3f}, nu_iz = {nu_iz:.3f} (normalized); analytic penetration length = {lam:.4f}")
print(f"fitted / analytic penetration length = {fitted / lam:.4f}; ionization sink / wall influx = {balance:.4f}")

OUTPUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.semilogy((length - x) / lam, line, "o", ms=3, label="fluid neutrals")
ax.semilogy((length - x) / lam, WALL_DENSITY * np.exp(-(length - x) / lam), "-", label=r"$e^{-x/\lambda_{pen}}$")
ax.set(xlabel=r"distance from the wall / $\lambda_{pen}$", ylabel="$n_n / n_0$")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "penetration.png", dpi=150)
