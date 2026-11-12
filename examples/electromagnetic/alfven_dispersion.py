"""Alfven-wave dispersion of the electromagnetic four-field model against theory.

Left: shear-Alfven wave with electron inertia, omega / (kz vA) = 1 / sqrt(1 + k_perp^2 d_e^2). Waves of the production
RK4 step are seeded with the analytic eigenvector and the phase of phi is read (a 12 x 3 x 16 slab).
Right: drift-Alfven wave in a density gradient (Ln = 10, beta = 0.03); the flutter of the background gradient
b1.grad n0 lowers the Alfven branch from its electrostatic-limit value (12 x 16 x 16 slab).

Run:  PYTHONPATH=src python examples/electromagnetic/alfven_dispersion.py   (about 1.5 minutes)
Writes OUTPUT_DIR/alfven_dispersion.png.
"""

from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drbx.validation.electromagnetic_waves import dispersion_matrix, measured_frequency

jax.config.update("jax_enable_x64", True)

# ----------------------------- PARAMETERS -----------------------------------
OUTPUT_DIR = Path("output/electromagnetic")
MU = 100.0                                  # mi / me
KX, KZ = 1.0, 0.25                          # shear-Alfven wave numbers (k_perp = KX)
BETAS = [0.5, 0.15, 0.05, 0.015, 0.005]     # k_perp d_e = KX / sqrt(beta MU) from 0.14 to 1.4
SHEAR_SHAPE = (12, 3, 16)
DRIFT = dict(kx=1.0, ky=1.0, beta=0.03, te=1.0, ln=10.0)   # drift-Alfven case
DRIFT_KZ = [0.05, 0.1, 0.2]
DRIFT_SHAPE = (12, 16, 16)
# ----------------------------------------------------------------------------

shear = []
for beta in BETAS:
    k_de = KX / np.sqrt(beta * MU)
    _, measured = measured_frequency(KX, 0.0, KZ, beta, MU, 0.0, None, SHEAR_SHAPE)
    shear.append((k_de, measured / (KZ / np.sqrt(beta))))
    print(f"shear-Alfven k_perp d_e = {k_de:.2f}: measured / (kz vA) = {shear[-1][1]:.4f}, theory {1 / np.sqrt(1 + k_de**2):.4f}")

drift = []
for kz in DRIFT_KZ:
    args = (DRIFT["kx"], DRIFT["ky"], kz, DRIFT["beta"], MU, DRIFT["te"], DRIFT["ln"])
    analytic, measured = measured_frequency(*args, DRIFT_SHAPE)
    electrostatic = (1j * np.linalg.eigvals(dispersion_matrix(*args[:3], 1e-8, *args[4:]))).real.max()
    drift.append((kz, analytic, measured, electrostatic))
    print(f"drift-Alfven kz = {kz}: analytic {analytic:.4f}, measured {measured:.4f}, electrostatic limit {electrostatic:.4f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
x = np.linspace(0.05, 1.6, 200)
left.plot(x, 1 / np.sqrt(1 + x**2), "k-", label=r"$1/\sqrt{1+k_\perp^2 d_e^2}$")
left.axhline(1.0, color="0.6", ls="--", label="no electron inertia")
left.plot(*zip(*shear), "o", color="C3", label="RHS of the model")
left.set(xlabel=r"$k_\perp d_e$", ylabel=r"$\omega / (k_\parallel v_A)$", title="shear-Alfven wave")
left.legend(frameon=False)
kz_line = np.linspace(0.03, 0.22, 100)
roots = lambda beta, k: max((1j * np.linalg.eigvals(dispersion_matrix(DRIFT["kx"], DRIFT["ky"], k, beta, MU, DRIFT["te"], DRIFT["ln"]))).real.max(), 0)  # noqa: E731
right.plot(kz_line, [roots(DRIFT["beta"], k) for k in kz_line], "k-", label=r"drift-Alfven, $\beta = 0.03$")
right.plot(kz_line, [roots(1e-8, k) for k in kz_line], "k--", label=r"$\beta \to 0$ (electrostatic)")
right.plot([d[0] for d in drift], [d[2] for d in drift], "o", color="C3", label="RHS of the model")
right.set(xlabel=r"$k_\parallel$", ylabel=r"$\omega$", title=r"drift-Alfven wave, $L_n = 10$")
right.legend(frameon=False)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_DIR / "alfven_dispersion.png", dpi=110)
print(f"wrote {OUTPUT_DIR / 'alfven_dispersion.png'}")
