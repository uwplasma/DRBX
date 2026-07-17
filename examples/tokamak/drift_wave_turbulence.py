"""Tokamak closed-field-line flagship: Hasegawa-Wakatani drift-wave turbulence.

Physics. The two-field Hasegawa-Wakatani model (Hasegawa & Wakatani, Phys. Rev.
Lett. 50, 682 (1983)) evolves vorticity ``zeta = lap phi`` and density ``n`` in
the plane perpendicular to the magnetic field:

    d/dt zeta = -{phi, zeta} + alpha (phi - n) - nu lap^2 zeta - mu zeta
    d/dt n    = -{phi, n} - kappa d(phi)/dy + alpha (phi - n) - nu lap^2 n - mu n

A background density gradient ``kappa`` drives the resistive drift-wave
instability; the adiabaticity ``alpha`` is the parallel electron response that
couples ``n`` and ``phi``; hyperviscosity ``nu`` absorbs the forward enstrophy
cascade at the grid scale and friction ``mu`` (sheath/neutral drag) absorbs the
2-D inverse cascade at the box scale, so a fixed-step run reaches a
statistically steady turbulent state. This script starts from small random
noise and follows the full life cycle: linear growth (verified against
``drbx.linear``, benchmark B2, in ``tests/test_hasegawa_wakatani.py``),
nonlinear saturation, and steady outward E x B particle transport.

It prints (step, time, fluctuation energy, particle flux) every block and
writes ``output/drift_wave_turbulence/`` with a three-panel PNG (saturated
vorticity field, energy growth curve, flux time trace), an animated vorticity
GIF, and a JSON time series. Run it with:

    PYTHONPATH=src python examples/tokamak/drift_wave_turbulence.py

Edit the PARAMETERS block below to change resolution, drive, or run length.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    particle_flux,
    potential_from_vorticity,
)

# ----------------------------- PARAMETERS ----------------------------------
N = 96                       # grid points per side (raise for finer turbulence)
LENGTH = 2.0 * np.pi * 8.0   # box side in units of rho_s (k_min = 2 pi / LENGTH)
ADIABATICITY = 1.0           # alpha: lower it (0.1) for streamers, raise (2+) for weak transport
GRADIENT = 1.0               # kappa: density-gradient drive of the instability
HYPERVISCOSITY = 1.0e-2      # nu: grid-scale lap^2 damping (keeps the cascade resolved)
FRICTION = 3.0e-2            # mu: large-scale drag absorbing the inverse cascade
DT = 5.0e-3                  # time step (limited by the stiff alpha/k_min^2 response)
STEPS_PER_BLOCK = 800        # steps between diagnostics/movie frames (4 time units)
BLOCKS = 40                  # number of blocks: total time = BLOCKS * STEPS_PER_BLOCK * DT
SEED = 0                     # RNG seed of the initial noise (reproducibility)
SEED_AMPLITUDE = 5.0e-2      # rms of the initial real-space noise field
SAVE_MOVIE = True            # write the animated vorticity GIF
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "drift_wave_turbulence"

# ----------------------------- SIMULATION SETUP -----------------------------
# Periodic pseudo-spectral grid with wavenumbers, 1/k^2, and the 2/3 dealias mask.
grid = hw_grid(N, LENGTH)
params = HasegawaWakataniParameters(
    adiabaticity=ADIABATICITY,
    gradient=GRADIENT,
    hyperviscosity=HYPERVISCOSITY,
    friction=FRICTION,
)

# Initial condition: band-limited random noise. Building the spectrum as the
# FFT of a REAL field keeps it Hermitian, so vorticity and density stay real
# under the pseudo-spectral evolution (a non-Hermitian seed would silently
# evolve an unphysical complexified system).
rng = np.random.default_rng(SEED)
noise_hat = np.fft.fft2(rng.standard_normal((N, N)))
noise_hat[0, 0] = 0.0                                  # no mean component
noise_hat *= np.exp(-np.asarray(grid.k2))              # weight toward large scales
noise_hat *= np.asarray(grid.dealias)                  # keep the seed band-limited
noise_hat *= SEED_AMPLITUDE / np.sqrt(np.mean(np.real(np.fft.ifft2(noise_hat)) ** 2))
zeta = jnp.array(noise_hat)      # spectral vorticity zeta_k
density = jnp.array(noise_hat)   # spectral density n_k (same noise)

# ----------------------------- RUN LOOP -------------------------------------
# hw_run advances STEPS_PER_BLOCK RK4 steps inside a jitted lax.scan; the Python
# loop only handles diagnostics, so the whole run is a handful of XLA calls.
times, energies, fluxes, frames = [], [], [], []
for block in range(BLOCKS):
    zeta, density = hw_run(zeta, density, grid, params, dt=DT, steps=STEPS_PER_BLOCK)
    phi = potential_from_vorticity(zeta, grid)
    energy = float(jnp.sum(grid.k2 * jnp.abs(phi) ** 2 + jnp.abs(density) ** 2)) / N**4
    flux = float(particle_flux(zeta, density, grid))
    times.append(DT * STEPS_PER_BLOCK * (block + 1))
    energies.append(energy)
    fluxes.append(flux)
    frames.append(np.real(np.asarray(jnp.fft.ifft2(zeta))))
    print(
        f"step={STEPS_PER_BLOCK * (block + 1):6d}  t={times[-1]:7.1f}  "
        f"energy={energy:.4e}  particle_flux={flux:+.4e}"
    )

vorticity = frames[-1]

# ----------------------------- SAVE AND PLOT --------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "drift_wave_turbulence.json").write_text(
    json.dumps({"time": times, "energy": energies, "particle_flux": fluxes}, indent=2)
)
print(f"wrote {OUTPUT_DIR / 'drift_wave_turbulence.json'}")


def _write_vorticity_gif(frame_list, path, *, upscale=4, fps=12):
    """Encode the vorticity frames as a compact RdBu_r animated GIF."""

    from PIL import Image

    scale = max(abs(float(np.min(frame_list[-1]))), abs(float(np.max(frame_list[-1]))), 1e-30)
    images = []
    for frame in frame_list:
        norm = np.clip((frame / scale + 1.0) / 2.0, 0.0, 1.0)
        rgba = (plt.get_cmap("RdBu_r")(norm) * 255).astype(np.uint8)
        image = Image.fromarray(rgba, mode="RGBA").convert("P", palette=Image.ADAPTIVE)
        image = image.resize((image.width * upscale, image.height * upscale), Image.NEAREST)
        images.append(image)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0
    )


if SAVE_MOVIE:
    _write_vorticity_gif(frames, OUTPUT_DIR / "drift_wave_turbulence.gif")
    print(f"wrote {OUTPUT_DIR / 'drift_wave_turbulence.gif'}")

fig, (ax_field, ax_energy, ax_flux) = plt.subplots(1, 3, figsize=(15.0, 4.4))
im = ax_field.imshow(vorticity, cmap="RdBu_r", origin="lower")
ax_field.set_title(f"vorticity at t={times[-1]:.0f}")
ax_field.set_xticks([])
ax_field.set_yticks([])
fig.colorbar(im, ax=ax_field, fraction=0.046)
ax_energy.semilogy(times, energies, "-")
ax_energy.set_xlabel("time")
ax_energy.set_ylabel("fluctuation energy")
ax_energy.set_title("growth and saturation")
ax_energy.grid(alpha=0.3)
ax_flux.plot(times, fluxes, "-")
ax_flux.set_xlabel("time")
ax_flux.set_ylabel(r"particle flux $\langle n\, v_x\rangle$")
ax_flux.set_title("outward transport")
ax_flux.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "drift_wave_turbulence.png", dpi=180)
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'drift_wave_turbulence.png'}")
