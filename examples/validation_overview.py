"""Validation overview figure: operators, models and simulations, each against its reference.

(a) FCI parallel gradient, parallel Laplacian and perpendicular Laplacian: manufactured-solution error against resolution.
(b) Shear-Alfven and drift-Alfven waves of the electromagnetic four-field model against the analytic dispersion relation.
(c) 1D plasma-neutral leg (AMJUEL rates) against the SD1D 13.6 eV upstream-density scan.

Run from the repository root with ``PYTHONPATH=src:. python examples/validation_overview.py``.
Writes output/validation_overview.png.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

jax.config.update("jax_enable_x64", True)

from drbx.native.neutrals import DetachmentSolParameters, detachment_diagnostics, detachment_sol_run
from drbx.validation.electromagnetic_waves import dispersion_matrix, measured_frequency
from tests import test_fci_operators as ops

# ----------------------------- PARAMETERS -----------------------------------
OUT = Path("output/validation_overview.png")
RESOLUTIONS = (16, 32, 64)
MU, KX, KZ, BETAS, SHEAR_SHAPE = 100.0, 1.0, 0.25, (0.5, 0.15, 0.05, 0.015, 0.005), (12, 3, 16)
DRIFT, DRIFT_KZ, DRIFT_SHAPE = dict(kx=1.0, ky=1.0, beta=0.03, te=1.0, ln=10.0), (0.05, 0.1, 0.2), (12, 16, 16)
SD1D = ((2.000355e19, 21.30), (2.501205e19, 13.50), (2.999828e19, 9.58), (3.992420e19, 6.22),   # Dudson et al., PPCF 61, 065008
        (5.000658e19, 4.80), (5.999908e19, 4.07))                                              # (n_up, T_target), 13.6 eV scan
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)
plt.rcParams["axes.titlesize"] = 10
(a, b), (c, d) = axes
summary = {}

# (a) operators
quiet = io.StringIO()
curves = {}
with redirect_stdout(quiet):
    for label, test in (("parallel gradient (FCI traced)", ops.mms_test_grad_parallel_fci_dirichlet),
                        ("parallel Laplacian", ops.mms_test_parallel_laplacian_direct_op),
                        ("perpendicular Laplacian", ops.mms_test_perp_laplacian_local_op)):
        curves[label] = [float(test(ops.build_test_fci_geometry((n, n, n), construct_fci_maps=True))[0]) for n in RESOLUTIONS]
for label, err in curves.items():
    err = np.asarray(err)
    order = -np.polyfit(np.log(RESOLUTIONS), np.log(err), 1)[0]
    a.loglog(RESOLUTIONS, err / err[0], "o-", label=f"{label}: {order:.2f}")
    summary[label] = order
a.loglog(RESOLUTIONS, (np.array(RESOLUTIONS) / RESOLUTIONS[0]) ** -2.0, "k--", lw=1, label="slope -2")
a.set(xlabel="cells per direction", ylabel="rms error, relative to 16$^3$", title="(a) FCI operators, MMS")
a.legend(fontsize=7)

# (b) electromagnetic waves
shear = []
for beta in BETAS:
    k_de = KX / np.sqrt(beta * MU)
    shear.append((k_de, measured_frequency(KX, 0.0, KZ, beta, MU, 0.0, None, SHEAR_SHAPE)[1] / (KZ / np.sqrt(beta))))
x = np.linspace(0.05, 1.6, 100)
b.plot(x, 1 / np.sqrt(1 + x**2), "k-", label=r"$1/\sqrt{1+k_\perp^2 d_e^2}$ (analytic)")
b.plot(*zip(*shear), "o", color="C3", label="model RHS, RK4")
b.set(xlabel=r"$k_\perp d_e$", ylabel=r"$\omega/(k_\parallel v_A)$", title="(b) Shear-Alfven wave")
b.legend(fontsize=8)
summary["shear_err"] = max(abs(y * np.sqrt(1 + k**2) - 1) for k, y in shear)
drift = []
for kz in DRIFT_KZ:
    args = (DRIFT["kx"], DRIFT["ky"], kz, DRIFT["beta"], MU, DRIFT["te"], DRIFT["ln"])
    analytic, measured = measured_frequency(*args, DRIFT_SHAPE)
    drift.append((kz, analytic, measured))
kz_line = np.linspace(0.03, 0.22, 60)
roots = lambda k: max((1j * np.linalg.eigvals(dispersion_matrix(DRIFT["kx"], DRIFT["ky"], k, DRIFT["beta"], MU, DRIFT["te"], DRIFT["ln"]))).real.max(), 0)  # noqa: E731
c.plot(kz_line, [roots(k) for k in kz_line], "k-", label="analytic")
c.plot([r[0] for r in drift], [r[2] for r in drift], "o", color="C3", label="model RHS, RK4")
c.set(xlabel=r"$k_\parallel$", ylabel=r"$\omega$", title=r"(c) Drift-Alfven wave")
c.legend(fontsize=8)
summary["drift_err"] = max(abs(r[2] / r[1] - 1) for r in drift)

# (d) SD1D
rows, previous = [], None
for n_up, t_ref in SD1D:
    params = DetachmentSolParameters(ny=800, upstream_density=n_up, ionisation_energy=13.6)
    result = detachment_sol_run(params, previous.state if previous else None, source_scale=previous.source_scale if previous else 0.1)
    previous = result
    rows.append((n_up, t_ref, float(detachment_diagnostics(result.state, params, result.source_scale).target_temperature_ev)))
n_up, t_ref, t_run = (np.array(r) for r in zip(*rows))
d.semilogy(n_up / 1e19, t_ref, "ks", mfc="none", label="SD1D")
d.semilogy(n_up / 1e19, t_run, "o", color="C3", label="DRBX 1D leg")
d.set(xlabel=r"upstream density ($10^{19}$ m$^{-3}$)", ylabel="target temperature (eV)", title="(d) 1D plasma-neutral leg, SD1D")
d.legend(fontsize=8)
summary["sd1d_err"] = np.max(np.abs(t_run / t_ref - 1))

print(summary)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=100)
im = Image.open(OUT).convert("RGB").quantize(colors=128, dither=Image.Dither.NONE)
im.save(OUT, optimize=True)
print(OUT, OUT.stat().st_size // 1024, "KB")
