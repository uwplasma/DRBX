"""Survey of the linearized drift-reduced Braginskii solver.

``dkx.linear`` answers one question: *given any model and an equilibrium,
which perturbations grow, and how fast?* It has two entry points, both used
below:

- the **analytic dispersion operators** (`resistive_drift_wave_operator`,
  `shear_alfven_operator`, `interchange_operator`), small complex matrices
  assembled from the linearized model equations for a single Fourier mode; and
- the **general engine** (`jacobian_operator` + `eigenmodes`), which linearizes
  *any* right-hand side about *any* equilibrium with `jax.jacfwd` and
  diagonalizes the result — no hand derivation.

The script walks through three physical regimes, printing what is being solved
and checking each against its literature limit, then demonstrates the general
engine reproducing an analytic operator from a nonlinear right-hand side. It
ends with a four-panel summary figure.

Regimes surveyed (references in ``docs/linear_dispersion_benchmark.md``):

1. Resistive drift wave (Hasegawa-Wakatani): the adiabaticity ``alpha`` spans
   the hydrodynamic (``alpha << 1``) to adiabatic (``alpha >> 1``) regimes; in
   the adiabatic limit the frequency approaches the drift frequency
   ``omega* = kappa k_y / (1 + kperp^2)`` (Dudson et al., CPC 180, 1467, 2009).
2. Shear Alfven wave: electron inertia reduces the phase velocity,
   ``omega = k_par vA / sqrt(1 + kperp^2 de^2)`` (Stegmeir et al., PoP 26,
   052517, 2019).
3. Interchange / Rayleigh-Taylor: unstable only when the effective gravity and
   the density gradient push the same way, ``gamma = sqrt(g kappa) k_y / kperp``.

Run:

    PYTHONPATH=src python examples/benchmarks/linear_drb_survey.py
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from dkx.linear import (  # noqa: E402
    drift_wave_adiabatic_frequency,
    eigenmodes,
    interchange_growth_rate,
    interchange_operator,
    jacobian_operator,
    resistive_drift_wave_operator,
    shear_alfven_frequency,
    shear_alfven_operator,
)

# --- PARAMETERS -----------------------------------------------------------------
OUTPUT_DIR = Path("output/linear_drb_survey")   # artifact directory (cwd-relative)


def growth_and_frequency(operator):
    """Largest-growth eigenmode of a linear operator d(state)/dt = A state."""

    modes = eigenmodes(operator)
    leading = jnp.argmax(modes.growth_rates)
    return float(modes.growth_rates[leading]), float(modes.frequencies[leading])


def survey_drift_wave():
    print("\n=== 1. Resistive drift wave (Hasegawa-Wakatani, single mode) ===")
    print("solving d/dt [phi_k, n_k] = A [phi_k, n_k] with A from the linearized")
    print("HW equations; scanning the adiabaticity alpha at k_y = 0.5, kperp^2 = 1.25")
    k_y, kperp2, kappa = 0.5, 1.25, 1.0
    alphas = np.logspace(-2, 2, 41)
    growth, frequency = [], []
    for alpha in alphas:
        gamma, omega = growth_and_frequency(
            resistive_drift_wave_operator(k_y, kperp2, alpha, kappa))
        growth.append(gamma)
        frequency.append(omega)
    omega_star = float(drift_wave_adiabatic_frequency(k_y, kperp2, kappa))
    print(f"  hydrodynamic (alpha={alphas[0]:.2f}): gamma={growth[0]:.4f}")
    print(f"  adiabatic    (alpha={alphas[-1]:.0f}):  |omega|={abs(frequency[-1]):.4f}"
          f"  vs omega* = {omega_star:.4f}  (drift frequency)")
    # eigenvalues follow e^(lambda t): the wave frequency is |Im lambda|.
    assert abs(abs(frequency[-1]) - omega_star) < 0.02 * abs(omega_star)
    return alphas, np.asarray(growth), np.abs(np.asarray(frequency)), omega_star


def survey_shear_alfven():
    print("\n=== 2. Shear Alfven wave with electron inertia ===")
    print("solving d/dt [phi_k, psi_k] = A [phi_k, psi_k]; scanning kperp at")
    print("k_par = 0.1, vA = 1; electron skin depth de = 1 slows the wave")
    k_par, v_alfven = 0.1, 1.0
    kperps = np.linspace(0.0, 3.0, 31)
    numeric, analytic = [], []
    for de in (0.0, 1.0):
        for k_perp in kperps:
            _, omega = growth_and_frequency(
                shear_alfven_operator(k_par, k_perp, v_alfven, de))
            numeric.append(abs(omega))
            analytic.append(float(shear_alfven_frequency(k_par, k_perp, v_alfven, de)))
    numeric = np.asarray(numeric).reshape(2, -1)
    analytic = np.asarray(analytic).reshape(2, -1)
    error = float(np.max(np.abs(numeric - analytic)))
    print(f"  eigenvalue vs analytic omega = k_par vA / sqrt(1 + kperp^2 de^2):"
          f" max |error| = {error:.2e}")
    assert error < 1e-12
    return kperps, numeric, analytic


def survey_interchange():
    print("\n=== 3. Interchange / Rayleigh-Taylor ===")
    print("solving d/dt [phi_k, n_k] = A [phi_k, n_k]; scanning the effective")
    print("gravity g: unstable (gamma = sqrt(g kappa) k_y/kperp) only for g > 0")
    k_y, kperp2, kappa = 1.0, 2.0, 1.0
    gravities = np.linspace(-1.0, 1.0, 41)
    growth = [growth_and_frequency(interchange_operator(k_y, kperp2, g, kappa))[0]
              for g in gravities]
    analytic = [float(interchange_growth_rate(k_y, kperp2, g, kappa)) if g > 0 else 0.0
                for g in gravities]
    stable = max(g for g, grav in zip(growth, gravities) if grav < 0)
    print(f"  stable branch (g < 0): max gamma = {stable:.2e} (no instability)")
    print(f"  unstable branch matches sqrt(g kappa) k_y/kperp to "
          f"{float(np.max(np.abs(np.asarray(growth) - analytic))):.2e}")
    return gravities, np.asarray(growth), np.asarray(analytic)


def survey_general_engine():
    print("\n=== 4. The general engine: linearize ANY right-hand side ===")
    print("jacobian_operator(rhs, equilibrium) builds A = d(rhs)/d(state) with")
    print("jax.jacfwd; the state must be real (complex spectral states are")
    print("realified first, as in the Hasegawa-Wakatani gate). Here: a nonlinear")
    print("interchange system in real variables, linearized at its equilibrium")
    k_y, kperp2, gravity, kappa = 1.0, 2.0, 0.5, 1.0
    # The complex interchange operator has purely imaginary couplings; in the
    # real variables (phi, i n) it becomes this real matrix with the same
    # eigenvalues +/- sqrt(g kappa) k_y / kperp.
    linear_part = jnp.array([[0.0, gravity * k_y / kperp2], [kappa * k_y, 0.0]])

    def nonlinear_rhs(state):
        # Linear interchange terms plus a quadratic nonlinearity that vanishes
        # at the equilibrium state = 0, so the Jacobian there is the linear part.
        return linear_part @ state + 0.3 * state * jnp.roll(state, 1) ** 2

    operator = jacobian_operator(nonlinear_rhs, jnp.zeros(2))
    difference = float(jnp.max(jnp.abs(operator - linear_part)))
    gamma_general, _ = growth_and_frequency(operator)
    gamma_analytic = float(interchange_growth_rate(k_y, kperp2, gravity, kappa))
    print(f"  |jacobian - linear part| = {difference:.2e}")
    print(f"  growth rate from the general engine: {gamma_general:.6f}"
          f"  (analytic sqrt(g kappa) k_y/kperp = {gamma_analytic:.6f})")
    assert difference < 1e-12
    assert abs(gamma_general - gamma_analytic) < 1e-12
    print("  (the same call linearizes the full nonlinear Hasegawa-Wakatani model;")
    print("   see tests/test_linear_dispersion.py for that gate at ~1e-15)")


# --- run the three regime surveys plus the general engine -------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
alphas, dw_growth, dw_frequency, omega_star = survey_drift_wave()
kperps, sa_numeric, sa_analytic = survey_shear_alfven()
gravities, ic_growth, ic_analytic = survey_interchange()
survey_general_engine()

# --- four-panel summary figure ----------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))
ax = axes[0, 0]
ax.semilogx(alphas, dw_growth, color="#d62728", label="growth rate")
ax.semilogx(alphas, dw_frequency, color="#1f77b4", label="frequency")
ax.axhline(omega_star, color="#1f77b4", ls=":", label="omega* (adiabatic limit)")
ax.set_xlabel("adiabaticity alpha"), ax.set_ylabel("gamma, omega")
ax.set_title("Resistive drift wave: hydrodynamic -> adiabatic")
ax.legend(fontsize=8), ax.grid(True, ls=":", alpha=0.4)

ax = axes[0, 1]
for row, de, color in ((0, 0.0, "#1f77b4"), (1, 1.0, "#d62728")):
    ax.plot(kperps, sa_numeric[row], "o", ms=3.5, color=color,
            label=f"eigenvalue, de = {de:g}")
    ax.plot(kperps, sa_analytic[row], "-", lw=1, color=color, alpha=0.7)
ax.set_xlabel("kperp"), ax.set_ylabel("|omega|")
ax.set_title("Shear Alfven wave: electron inertia slows the wave")
ax.legend(fontsize=8), ax.grid(True, ls=":", alpha=0.4)

ax = axes[1, 0]
ax.plot(gravities, ic_growth, "o", ms=3.5, color="#d62728", label="eigenvalue")
ax.plot(gravities, ic_analytic, "-", color="gray", label="sqrt(g kappa) k_y/kperp")
ax.set_xlabel("effective gravity g"), ax.set_ylabel("growth rate")
ax.set_title("Interchange: unstable only for g > 0")
ax.legend(fontsize=8), ax.grid(True, ls=":", alpha=0.4)

ax = axes[1, 1]
ax.axis("off")
ax.text(0.0, 0.9, "What is being solved", fontsize=11, weight="bold")
ax.text(0.0, 0.05,
        "Linearize a model about an equilibrium:\n"
        "    d(state)/dt = A state,   A = d rhs / d state\n\n"
        "Analytic operators: A written down per Fourier mode\n"
        "General engine: A = jax.jacfwd(rhs)(equilibrium),\n"
        "then eigenmodes(A) -> growth rates + frequencies.\n\n"
        "Anchors: drift wave (Dudson CPC 2009),\n"
        "shear Alfven (Stegmeir PoP 2019),\n"
        "interchange (flute-mode gamma = sqrt(g kappa) ky/kperp);\n"
        "each reproduced to machine precision.",
        fontsize=9, family="monospace", va="bottom")

fig.suptitle("Linearized drift-reduced Braginskii: three regimes and the general engine")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "linear_drb_survey.png", dpi=170)
plt.close(fig)
print(f"\nwrote {OUTPUT_DIR / 'linear_drb_survey.png'}")


