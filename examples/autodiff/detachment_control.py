"""Gradient-based control of the target temperature in the SD1D-matched model.

Find the upstream density ``n_up`` that places the steady target temperature
of the SD1D ``13.6eV`` configuration at a requested value, with Newton steps
on ``ln T_t`` whose derivative ``d T_t / d n_up`` comes from the
implicit-function theorem at the converged steady state
(:func:`detachment_target_outputs`, forward mode through
``lax.custom_linear_solve`` with the transposed block-tridiagonal solve) --
no differentiation through the pseudo-time iterations. The derivative is also
checked once against a central finite difference.

Run:

    PYTHONPATH=src python examples/autodiff/detachment_control.py

writes ``output/detachment_control/`` with a PNG and a JSON summary (about a
minute on a laptop CPU at the default 200 cells).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drbx.native.neutrals import (  # noqa: E402
    DetachmentSolParameters,
    detachment_sol_run,
    detachment_target_outputs,
)

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
NY = 200                  # cells (SD1D grid shape; 800 reproduces the paper values)
POWER_FLUX = 5.0e7        # W/m^2 into the first 10 m
TARGET_EV = 10.0          # requested steady target temperature
INITIAL_DENSITY = 2.0e19  # controller start [m^-3]
ITERATIONS = 8
TOLERANCE_EV = 1e-3
OUTPUT_DIR = Path("output/detachment_control")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
params = DetachmentSolParameters(ny=NY, upstream_density=INITIAL_DENSITY, power_flux=POWER_FLUX)
guess = detachment_sol_run(params)
print(f"== start: n_up = {INITIAL_DENSITY:.3e}, residual {guess.residual:.1e}")

density, history = INITIAL_DENSITY, []
for iteration in range(ITERATIONS):
    theta = jnp.asarray([density, POWER_FLUX])

    def outputs(th, guess=guess):
        return detachment_target_outputs(th, params, guess)

    value = outputs(theta)
    jac = jax.jacfwd(outputs)(theta)
    t_t, dt_dn = float(value[0]), float(jac[0, 0])
    history.append(dict(iteration=iteration, n_up=density, T_t=t_t, dT_dn=dt_dn))
    print(f"   iter {iteration}: n_up = {density:.5e}  T_t = {t_t:8.4f} eV  dT_t/dn_up = {dt_dn:.3e} eV m^3")
    if iteration == 0:
        h = 1e-3 * density
        fd = (float(outputs(theta + jnp.asarray([h, 0.0]))[0]) - float(outputs(theta - jnp.asarray([h, 0.0]))[0])) / (2 * h)
        print(f"   check: central difference {fd:.6e} vs implicit derivative {dt_dn:.6e} "
              f"(rel. diff {abs(fd / dt_dn - 1):.1e})")
    if abs(t_t - TARGET_EV) < TOLERANCE_EV:
        break
    step = -(np.log(t_t) - np.log(TARGET_EV)) / (dt_dn / t_t)
    density = float(np.clip(density + step, 0.5 * density, 2.0 * density))
    guess = detachment_sol_run(DetachmentSolParameters(ny=NY, upstream_density=density, power_flux=POWER_FLUX),
                               guess.state, source_scale=guess.source_scale)

print(f"== result: n_up = {history[-1]['n_up']:.5e} gives T_t = {history[-1]['T_t']:.4f} eV")
(OUTPUT_DIR / "summary.json").write_text(json.dumps(history, indent=2))

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.semilogy([h["iteration"] for h in history], [abs(h["T_t"] - TARGET_EV) + 1e-12 for h in history], "o-")
ax.set_xlabel("Newton iteration")
ax.set_ylabel(f"|T_t - {TARGET_EV:g} eV|")
ax.set_title("Target-temperature control with implicit derivatives")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "detachment_control.png", dpi=150)
print(f"wrote {OUTPUT_DIR / 'detachment_control.png'}")
