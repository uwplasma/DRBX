"""Validation overview figure: operators, models and simulations, each against its reference.

(a) FCI parallel gradient, parallel Laplacian and perpendicular Laplacian: manufactured-solution error against resolution.
(b) Shear-Alfven and drift-Alfven waves of the electromagnetic four-field model against the analytic dispersion relation.
(c) 1D plasma-neutral leg (AMJUEL rates) against the SD1D 13.6 eV upstream-density scan.
(d) HSX 16^3 run with neutrals and sheath: wall-face particle and energy ledger against the integrated wall fluxes.
(e) Reverse-mode gradient through the SOLVAX FGMRES solve against central differences.

Run from the repository root:  PYTHONPATH=src:. python examples/validation_overview.py   (about 6 min on a laptop CPU)
Writes docs/media/validation_overview.png.
"""

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

jax.config.update("jax_enable_x64", True)

from drbx.cli import main as drbx
from drbx.fci_braginskii.native.fci_gmres import SolvaxGmresConfig, solvax_gmres_solve
from drbx.native.neutrals import DetachmentSolParameters, detachment_diagnostics, detachment_sol_run
from drbx.validation.electromagnetic_waves import dispersion_matrix, measured_frequency
from tests import test_fci_operators as ops
from tests.fci_braginskii.test_fci_gmres_budget import _convection_dominated_operator, _single_device_slab
from tests.fci_braginskii.test_fci_parallel_adjoint import _dense, _pair

# ----------------------------- PARAMETERS -----------------------------------
OUT = Path("docs/media/validation_overview.png")
RESOLUTIONS = (16, 32, 64)
MU, KX, KZ, BETAS, SHEAR_SHAPE = 100.0, 1.0, 0.25, (0.5, 0.15, 0.05, 0.015, 0.005), (12, 3, 16)
DRIFT, DRIFT_KZ, DRIFT_SHAPE = dict(kx=1.0, ky=1.0, beta=0.03, te=1.0, ln=10.0), (0.05, 0.1, 0.2), (12, 16, 16)
SD1D = ((2.000355e19, 21.30), (2.501205e19, 13.50), (2.999828e19, 9.58), (3.992420e19, 6.22),   # Dudson et al., PPCF 61, 065008
        (5.000658e19, 4.80), (5.999908e19, 4.07))                                              # (n_up, T_target), 13.6 eV scan
HSX_DECK, HSX_STEPS, HSX_TIME, HSX_DT = "examples/inputs/hsx_fci_neutrals.toml", 800, 0.6, 7.5e-4
DIRECTIONS = 8
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
plt.rcParams["axes.titlesize"] = 10
(a, b, c), (d, e, f) = axes
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
gradient, divergence, mass, _, _ = _pair()
g, dv, v = _dense(gradient, mass.shape), _dense(divergence, mass.shape), np.asarray(mass).reshape(-1)
adjoint = np.linalg.norm(dv + (g.T * v[None, :]) / v[:, None]) / np.linalg.norm(dv)
summary["adjoint"] = adjoint
a.set(xlabel="cells per direction", ylabel="rms error, relative to 16$^3$", title="(a) FCI operators, MMS")
a.legend(fontsize=7, title=f"order; (G, D) adjoint residual {adjoint:.0e}", title_fontsize=7)

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

# (c) SD1D
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

# (d) HSX wall ledger and step time
buffer = io.StringIO()
with redirect_stdout(buffer):
    drbx(["run", HSX_DECK, "--set", f"fci_braginskii.final_time={HSX_TIME}", "--set", f"fci_braginskii.num_steps={HSX_STEPS}",
          "--set", "fci_braginskii.save_every=100", "--set", "fci_braginskii.output=../../output/validation_overview/hsx"])
lines = buffer.getvalue().splitlines()
num = lambda line, key: float(re.search(rf"\b{key}=([-+0-9.e]+)", line).group(1))  # noqa: E731
wall = np.array([[num(l, k) for k in ("t", "q_e", "q_i", "K_i")] for l in lines if l.startswith("[wall] ")])
run = np.array([[num(l, "t"), num(l, "closure"), *map(float, re.search(r"W_e=([-+0-9.e]+) W_i=([-+0-9.e]+)", l).groups())]
                for l in lines if l.startswith("[wall-run]")])
rate = wall[:, 1] + wall[:, 2]
loss = np.interp(run[:, 0], wall[:, 0], np.concatenate([[0], np.cumsum(0.5 * np.diff(wall[:, 0]) * (rate[1:] + rate[:-1]))]))
energy = np.abs(run[:, 2] + run[:, 3] + np.interp(run[:, 0], wall[:, 0], wall[:, 3])) / loss   # |W_e + W_i + K_i - W(0) - face flux| / wall loss
late = run[:, 0] > 0.05
e.semilogy(run[late, 0], run[late, 1], color="C0", label="particles")
e.semilogy(run[late, 0], energy[late], color="C3", label="energy")
e.set(xlabel=r"time ($L_{ref}/c_{s0}$)", ylabel="closure error / wall loss")
e.legend(fontsize=8)
summary["hsx_particle"], summary["hsx_energy"] = run[-1, 1], energy[-1]
e.set_title("(e) HSX 16$^3$ wall-face ledger")

# (e) gradient vs finite differences
geometry, domain = _single_device_slab()
rhs = jnp.asarray(np.random.default_rng(1).standard_normal(geometry.owned_shape))
config = SolvaxGmresConfig(tol=1.0e-13, atol=1.0e-14, restart=50, maxiter=200)
apply_A, weights = _convection_dominated_operator(geometry.owned_shape, diffusion=0.05), jnp.cos(jnp.arange(rhs.size).reshape(rhs.shape))
loss = lambda r: jnp.sum(weights * solvax_gmres_solve(apply_A, r, jnp.zeros_like(r), geometry, domain, config, preconditioner=lambda u: u / 1.3)[0])  # noqa: E731
grad = jax.jit(jax.grad(loss))(rhs)
directions = np.random.default_rng(2).standard_normal((DIRECTIONS,) + rhs.shape)
reverse = np.array([float(jnp.vdot(grad, u)) for u in directions])
fd = np.array([float(loss(rhs + 1e-6 * u) - loss(rhs - 1e-6 * u)) / 2e-6 for u in directions])
summary["grad_err"] = np.max(np.abs(reverse - fd)) / np.max(np.abs(fd))
f.plot(fd, reverse, "o", color="C3")
f.plot([fd.min(), fd.max()], [fd.min(), fd.max()], "k-", lw=1)
f.set(xlabel="central difference", ylabel="reverse mode", title="(f) Reverse mode vs finite difference")

print(summary)
fig.savefig(OUT, dpi=100)
im = Image.open(OUT).convert("RGB").quantize(colors=128, dither=Image.Dither.NONE)
im.save(OUT, optimize=True)
print(OUT, OUT.stat().st_size // 1024, "KB")
