"""Optimizing parallel electron conductivity for a target turbulent transport level.

Physics. In the Hasegawa-Wakatani model the adiabaticity parameter ``alpha``
(the normalized parallel electron response, proportional to the parallel
electron conductivity) controls the character of drift-wave turbulence and the
turbulent particle transport it drives. Small ``alpha`` gives the hydrodynamic
regime -- weak parallel coupling, streamer-like convection, high radial
particle flux; large ``alpha`` gives the adiabatic regime -- Boltzmann-like
electrons, wave-like drift turbulence, low flux. This
hydrodynamic-to-adiabatic transition is the classic result of Camargo, Biskamp
& Scott, Phys. Plasmas 2, 48 (1995), for the model of Hasegawa & Wakatani,
Phys. Rev. Lett. 50, 682 (1983).

This example asks the inverse-engineering question: how much parallel electron
conductivity does the plasma need to reach a specified confinement level?
Concretely, starting from saturated turbulence in the hydrodynamic regime at
``ALPHA_START``, find the adiabaticity whose saturated particle flux is
``FLUX_REDUCTION`` times lower. Because the whole pseudo-spectral rollout is
JAX, the objective gradient d ln(flux) / d ln(alpha) flows THROUGH the
turbulence: a safeguarded damped-Newton iteration on ln(alpha) converges in a
handful of steps.

Method notes (each choice is commented in place below):
- the flux is time-averaged over a saturated window, and every evaluation
  restarts from the same "anchor" turbulent state so the objective is a
  deterministic, smooth function of alpha (identical seed = fair comparison);
- the differentiated horizon is kept short (a few eddy turnovers); gradients
  of longer chaotic rollouts blow up exponentially (Lyapunov growth) and stop
  matching the macroscopic flux response;
- the anchor state is refreshed whenever Newton converges on the current
  anchor, so at the final fixed point the window average equals the true
  saturated flux at the optimized alpha (no relaxation bias);
- with a single scalar parameter, forward-mode AD (jax.jacfwd / jax.jvp) is
  the efficient choice: one tangent pass, no storage of the whole rollout as
  reverse mode would need.

It prints every Newton iteration (alpha, windowed flux, target, residual,
gradient) and a long independent verification at the initial and optimized
alpha, and writes ``output/hasegawa_wakatani_optimization/`` with a
two-column PNG (left: initial alpha, right: optimized alpha; top: saturated
density snapshot, bottom: flux time trace) plus a JSON summary. Run it with:

    PYTHONPATH=src python examples/tokamak/hasegawa_wakatani_optimization.py

Total runtime is a few minutes on a laptop CPU. Edit the PARAMETERS block to
change the regime, the target, or the optimizer safeguards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.native.hasegawa_wakatani import (  # noqa: E402
    HasegawaWakataniParameters,
    hw_grid,
    hw_run,
    hw_run_flux_history,
)

# ----------------------------- PARAMETERS ----------------------------------
N = 64                       # grid points per side (64 is plenty for the transition)
LENGTH = 2.0 * np.pi * 4.0   # box side in rho_s units (k_min = 0.25)
GRADIENT = 1.0               # kappa: background density-gradient drive
HYPERVISCOSITY = 1.0e-2      # nu: grid-scale lap^2 damping (resolves the cascade)
FRICTION = 3.0e-2            # mu: large-scale drag absorbing the inverse cascade
DT = 5.0e-3                  # RK4 step (stable for alpha up to ~5 on this grid)
SEED = 7                     # RNG seed of the turbulence initial condition
SEED_AMPLITUDE = 0.5         # rms of the initial noise (large: skip slow linear phase)

ALPHA_START = 0.3            # initial adiabaticity: hydrodynamic, high-transport regime
FLUX_REDUCTION = 4.0         # target: saturated flux ALPHA_START-flux / FLUX_REDUCTION

T_SPINUP = 60.0              # time to reach saturated turbulence at ALPHA_START
T_SETTLE = 10.0              # window head discarded after switching alpha
T_AVERAGE = 20.0             # flux-averaging window (a few eddy turnovers)
FLUX_SAMPLE_EVERY = 20       # steps between flux samples inside the window
NEWTON_TOL = 0.05            # convergence: | ln(flux) - ln(target) | < 5 %
NEWTON_MAX_ITER = 15         # iteration cap (typically converges in ~6)
NEWTON_MAX_STEP = 0.5        # clip on the ln(alpha) update (trust region)
SLOPE_CEILING = -0.3         # slope safeguard: flux must decrease with alpha
T_VERIFY = 150.0             # length of the final independent verification runs
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "output" / "hasegawa_wakatani_optimization"
)

# ----------------------------- SIMULATION SETUP -----------------------------
time_start = time.time()
grid = hw_grid(N, LENGTH)


def make_params(alpha):
    """Model parameters with everything fixed except the adiabaticity."""

    return HasegawaWakataniParameters(
        adiabaticity=alpha,
        gradient=GRADIENT,
        hyperviscosity=HYPERVISCOSITY,
        friction=FRICTION,
    )


# Initial condition: Hermitian (real-field) band-limited noise, identical for
# every alpha so the optimization compares like with like.
rng = np.random.default_rng(SEED)
noise_hat = np.fft.fft2(rng.standard_normal((N, N)))
noise_hat[0, 0] = 0.0
noise_hat *= np.exp(-np.asarray(grid.k2)) * np.asarray(grid.dealias)
noise_hat *= SEED_AMPLITUDE / np.sqrt(np.mean(np.real(np.fft.ifft2(noise_hat)) ** 2))
zeta_seed = jnp.array(noise_hat)
density_seed = jnp.array(noise_hat * 0.9)

# Spin up once (not differentiated) to saturated hydrodynamic turbulence: this
# is the anchor state every objective evaluation restarts from.
print(f"spin-up: alpha={ALPHA_START} for t={T_SPINUP:.0f} ...", flush=True)
zeta_anchor, density_anchor = hw_run(
    zeta_seed, density_seed, grid, make_params(ALPHA_START), dt=DT, steps=int(T_SPINUP / DT)
)

# ----------------------------- OBJECTIVE ------------------------------------
WINDOW_STEPS = int((T_SETTLE + T_AVERAGE) / DT)
N_TAIL = int(T_AVERAGE / DT) // FLUX_SAMPLE_EVERY  # flux samples actually averaged


def windowed_ln_flux(ln_alpha, zeta0, density0):
    """ln(time-averaged saturated flux) after a short rollout at exp(ln_alpha).

    Runs T_SETTLE + T_AVERAGE time units from the anchor state, discards the
    settle head, and averages the flux over the tail. Also returns the final
    state, which becomes the next anchor once Newton converges. Optimizing
    ln(alpha) keeps alpha positive and makes the Newton step scale-free.
    """

    _, _, fluxes = hw_run_flux_history(
        zeta0,
        density0,
        grid,
        make_params(jnp.exp(ln_alpha)),
        dt=DT,
        steps=WINDOW_STEPS,
        sample_every=FLUX_SAMPLE_EVERY,
    )
    return jnp.log(jnp.mean(fluxes[-N_TAIL:]))


def window_final_state(ln_alpha, zeta0, density0):
    """Final state of the same window rollout (used to refresh the anchor)."""

    zeta_f, density_f, _ = hw_run_flux_history(
        zeta0,
        density0,
        grid,
        make_params(jnp.exp(ln_alpha)),
        dt=DT,
        steps=WINDOW_STEPS,
        sample_every=FLUX_SAMPLE_EVERY,
    )
    return zeta_f, density_f


# One forward (jvp) pass gives the objective AND its derivative with respect to
# ln_alpha -- forward mode is the efficient choice for a single scalar
# parameter (jax.jacfwd(windowed_ln_flux) gives the same derivative; jvp just
# avoids evaluating the primal twice).
@jax.jit
def value_and_slope(ln_alpha, zeta0, density0):
    return jax.jvp(
        lambda la: windowed_ln_flux(la, zeta0, density0), (ln_alpha,), (1.0,)
    )


window_state = jax.jit(window_final_state)

# ----------------------------- NEWTON OPTIMIZATION --------------------------
ln_alpha = float(np.log(ALPHA_START))
ln_flux0, _ = value_and_slope(ln_alpha, zeta_anchor, density_anchor)
flux_initial = float(np.exp(float(ln_flux0)))
flux_target = flux_initial / FLUX_REDUCTION
print(
    f"initial saturated flux at alpha={ALPHA_START}: {flux_initial:.4f}  "
    f"-> target {flux_target:.4f} ({FLUX_REDUCTION:.0f}x reduction)"
)

history = []
converged_on_anchor = False
for iteration in range(NEWTON_MAX_ITER):
    ln_flux, slope = value_and_slope(ln_alpha, zeta_anchor, density_anchor)
    residual = float(ln_flux) - np.log(flux_target)
    history.append(
        {
            "iteration": iteration,
            "alpha": float(np.exp(ln_alpha)),
            "flux": float(np.exp(float(ln_flux))),
            "target": flux_target,
            "residual": residual,
            "dlnflux_dlnalpha": float(slope),
        }
    )
    print(
        f"iter={iteration:2d}  alpha={np.exp(ln_alpha):.4f}  "
        f"flux={np.exp(float(ln_flux)):.4f}  target={flux_target:.4f}  "
        f"residual={residual:+.4f}  dlnflux/dlnalpha={float(slope):+.3f}"
    )
    if abs(residual) < NEWTON_TOL:
        if converged_on_anchor:
            print("converged: residual below tolerance on a refreshed anchor")
            break
        # Converged on this anchor: refresh the anchor to the saturated state
        # at the current alpha and confirm, so the answer carries no memory of
        # the ALPHA_START state (removes the settle-window bias).
        print("  -> anchor refresh at current alpha, confirming ...")
        zeta_anchor, density_anchor = window_state(
            ln_alpha, zeta_anchor, density_anchor
        )
        converged_on_anchor = True
        continue
    converged_on_anchor = False
    # Damped Newton on g(ln alpha) = ln flux - ln target. The slope safeguard
    # keeps a (physically known) negative flux response even if the windowed
    # gradient is noisy; the step clip is a scalar trust region.
    safe_slope = min(float(slope), SLOPE_CEILING)
    step = float(np.clip(-residual / safe_slope, -NEWTON_MAX_STEP, NEWTON_MAX_STEP))
    ln_alpha += step

alpha_optimized = float(np.exp(ln_alpha))
print(
    f"optimized adiabaticity alpha={alpha_optimized:.4f} "
    f"(started {ALPHA_START}); optimization wall time {time.time() - time_start:.0f}s"
)

# ----------------------------- VERIFICATION RUNS ----------------------------
# Long, independent runs from the ORIGINAL seed at the initial and optimized
# alpha: saturated density snapshots and full flux traces for the figure.
VERIFY_SAMPLE = 50


@jax.jit
def verification_run(ln_alpha):
    return hw_run_flux_history(
        zeta_seed,
        density_seed,
        grid,
        make_params(jnp.exp(ln_alpha)),
        dt=DT,
        steps=int(T_VERIFY / DT),
        sample_every=VERIFY_SAMPLE,
    )


verify = {}
for label, alpha in [("initial", ALPHA_START), ("optimized", alpha_optimized)]:
    zeta_f, density_f, flux_trace = verification_run(float(np.log(alpha)))
    flux_trace = np.asarray(flux_trace)
    saturated = flux_trace[len(flux_trace) // 2 :]  # discard the spin-up half
    verify[label] = {
        "alpha": alpha,
        "density": np.real(np.asarray(jnp.fft.ifft2(density_f))),
        "flux_time": DT * VERIFY_SAMPLE * (1 + np.arange(len(flux_trace))),
        "flux_trace": flux_trace,
        "flux_mean": float(saturated.mean()),
        "flux_std": float(saturated.std()),
    }
    print(
        f"verification alpha={alpha:.4f}: saturated flux "
        f"= {verify[label]['flux_mean']:.4f} +/- {verify[label]['flux_std']:.4f} "
        f"(t > {T_VERIFY / 2:.0f})"
    )
achieved = verify["initial"]["flux_mean"] / verify["optimized"]["flux_mean"]
print(
    f"achieved flux reduction: {achieved:.2f}x (target {FLUX_REDUCTION:.0f}x); "
    f"total wall time {time.time() - time_start:.0f}s"
)

# ----------------------------- SAVE AND PLOT --------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "hasegawa_wakatani_optimization.json").write_text(
    json.dumps(
        {
            "alpha_start": ALPHA_START,
            "alpha_optimized": alpha_optimized,
            "flux_initial": flux_initial,
            "flux_target": flux_target,
            "flux_verified_initial": verify["initial"]["flux_mean"],
            "flux_verified_optimized": verify["optimized"]["flux_mean"],
            "achieved_reduction": achieved,
            "history": history,
        },
        indent=2,
    )
)
print(f"wrote {OUTPUT_DIR / 'hasegawa_wakatani_optimization.json'}")

fig, axes = plt.subplots(
    2, 2, figsize=(10.5, 8.6), gridspec_kw={"height_ratios": [1.35, 1.0]}
)
extent = [0.0, LENGTH, 0.0, LENGTH]
dens_scale = max(
    np.max(np.abs(verify["initial"]["density"])),
    np.max(np.abs(verify["optimized"]["density"])),
)
flux_max = 1.1 * max(
    verify["initial"]["flux_trace"].max(), verify["optimized"]["flux_trace"].max()
)
for column, label in enumerate(["initial", "optimized"]):
    run = verify[label]
    ax_dens, ax_flux = axes[0, column], axes[1, column]
    im = ax_dens.imshow(
        run["density"].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-dens_scale,
        vmax=dens_scale,
        extent=extent,
    )
    ax_dens.set_title(
        rf"{label}: $\alpha={run['alpha']:.2f}$"
        + "\n"
        + rf"$\langle\Gamma\rangle = {run['flux_mean']:.3f}$"
    )
    ax_dens.set_xlabel(r"$x/\rho_s$")
    ax_dens.set_ylabel(r"$y/\rho_s$" if column == 0 else "")
    ax_flux.plot(run["flux_time"], run["flux_trace"], lw=1.0, color="C0")
    ax_flux.axhline(
        run["flux_mean"], color="C1", lw=1.2, ls="-",
        label=rf"saturated mean {run['flux_mean']:.3f}",
    )
    if label == "optimized":
        ax_flux.axhline(
            flux_target, color="k", lw=1.0, ls="--",
            label=f"target {flux_target:.3f}",
        )
    ax_flux.set_ylim(0.0, flux_max)
    ax_flux.set_xlabel("time")
    ax_flux.set_ylabel(r"particle flux $\langle n\, v_x\rangle$" if column == 0 else "")
    ax_flux.legend(loc="upper left", fontsize=9)
    ax_flux.grid(alpha=0.3)
fig.colorbar(
    im, ax=axes[0, :].tolist(), fraction=0.03, pad=0.02, label=r"density fluctuation $n$"
)
fig.suptitle(
    "Tuning parallel electron conductivity for a "
    f"{FLUX_REDUCTION:.0f}x turbulent-transport reduction",
    y=0.98,
)
fig.savefig(OUTPUT_DIR / "hasegawa_wakatani_optimization.png", dpi=180,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUTPUT_DIR / 'hasegawa_wakatani_optimization.png'}")
