"""Verification figure of the electromagnetic four-field model: beta -> 0 limit, energy conservation, manufactured solution.

Left: max |RHS_em - RHS_es| / max |RHS_es| against beta (A_par and the flutter are O(beta)).
Middle: energies over one Alfven-wave period; the sum E x B + magnetic + electron kinetic is constant.
Right: error of the solved A_par and of dw/dt against grid size for a manufactured solution (slope -2 expected).

Run:  PYTHONPATH=src python examples/electromagnetic/electromagnetic_verification.py   (about 2 minutes)
Writes OUTPUT_DIR/electromagnetic_verification.png.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drbx.geometry import build_open_slab_geometry
from drbx.native.electromagnetic import em_energies
from drbx.native.fci_4_field_rhs import Fci4FieldState
from drbx.validation.electromagnetic_waves import laplacian, rhs, rk4, slab

jax.config.update("jax_enable_x64", True)

# ----------------------------- PARAMETERS -----------------------------------
OUTPUT_DIR = Path("output/electromagnetic")
MU = 100.0                       # mi / me
BETAS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
LIMIT_SHAPE, LIMIT_LENGTHS = (8, 6, 8), (np.pi, 2 * np.pi, 8.0)
WAVE_SHAPE, WAVE_BETA, WAVE_KX, WAVE_KY, WAVE_KZ, STEPS = (12, 8, 16), 0.05, 1.0, 1.0, 0.25, 40
MMS_BETA, MMS_MU, MMS_SIZES = 0.3, 50.0, [8, 12, 16, 24]
# ----------------------------------------------------------------------------
FIELDS = ("density", "omega", "v_ion_parallel", "v_electron_parallel")


def peak(state):
    return max(float(jnp.max(jnp.abs(getattr(state, f)))) for f in FIELDS)


# beta -> 0
g = build_open_slab_geometry(LIMIT_SHAPE, radial_extent=LIMIT_LENGTHS[0], binormal_extent=LIMIT_LENGTHS[1], parallel_length=LIMIT_LENGTHS[2])
x, y, z = (np.asarray(a.centers) for a in (g.grid.x, g.grid.y, g.grid.z))
sx = np.sin(np.pi * x / LIMIT_LENGTHS[0])[:, None, None]
ky, kz = 2 * np.pi / LIMIT_LENGTHS[1], 2 * np.pi / LIMIT_LENGTHS[2]
phi = 1e-2 * sx * np.cos(ky * y[None, :, None] + kz * z[None, None, :])
state = Fci4FieldState(density=jnp.asarray(1 + 1e-2 * sx * np.cos(ky * y[None, :, None] - kz * z[None, None, :]) + 0 * z),
                       omega=jnp.asarray(-(np.pi / LIMIT_LENGTHS[0]) ** 2 * phi - ky**2 * phi),
                       v_ion_parallel=jnp.asarray(1e-2 * sx * np.sin(ky * y[None, :, None] + 2 * kz * z[None, None, :])),
                       v_electron_parallel=jnp.asarray(0.5e-2 * sx * np.cos(2 * kz * z[None, None, :] - ky * y[None, :, None])))
_, es_kw, *_ = slab(LIMIT_SHAPE, LIMIT_LENGTHS, beta=1.0, mu=MU, te=1.0, em=False)
reference = rhs(state, es_kw)[0]
differences = []
for beta in BETAS:
    result = rhs(state, slab(LIMIT_SHAPE, LIMIT_LENGTHS, beta=beta, mu=MU, te=1.0)[1])[0]
    differences.append(max(float(jnp.max(jnp.abs(getattr(result, f) - getattr(reference, f)))) for f in FIELDS) / peak(reference))
    print(f"beta = {beta:g}: relative RHS difference {differences[-1]:.2e}")

# energy over one period
lengths = (np.pi / WAVE_KX, 2 * np.pi / WAVE_KY, 2 * np.pi / WAVE_KZ)
g, kw, phi_solver, bcs = slab(WAVE_SHAPE, lengths, beta=WAVE_BETA, mu=MU)
closure = kw["electromagnetic"]
x, y, z = (np.asarray(a.centers) for a in (g.grid.x, g.grid.y, g.grid.z))
wave = np.sin(WAVE_KX * x)[:, None, None] * np.cos(WAVE_KY * y[None, :, None] + WAVE_KZ * z[None, None, :])
period = 2 * np.pi * np.sqrt(1 + (WAVE_KX**2 + WAVE_KY**2) / (WAVE_BETA * MU)) * np.sqrt(WAVE_BETA) / WAVE_KZ
zero = jnp.zeros(WAVE_SHAPE)
state = Fci4FieldState(density=jnp.ones(WAVE_SHAPE), omega=laplacian(1e-3 * jnp.asarray(wave), g, kw, bcs), v_ion_parallel=zero, v_electron_parallel=zero)
step = jax.jit(lambda s: rk4(s, kw, period / STEPS))
history = []
for _ in range(STEPS + 1):
    potential = phi_solver(-state.omega, face_bc=bcs.phi_face_bc)
    apar = closure.solve_apar(state.density, state.v_ion_parallel, state.v_electron_parallel, face_bc=bcs.v_electron_parallel_face_bc)
    history.append([float(e) for e in em_energies(closure, state, potential, apar, g.cell_metric.J)[:3]])
    state = step(state)
history = np.array(history)
print(f"energy drift over one period: {np.ptp(history.sum(1)) / history[0].sum():.2e}")

# manufactured solution
a_f = lambda x, y, z: 0.1 * jnp.sin(x) * jnp.cos(z)  # noqa: E731
phi_f = lambda x, y, z: 0.1 * jnp.sin(x) * jnp.cos(z + 1.0)  # noqa: E731
n_f = lambda x, y, z: 1 + 0.2 * jnp.sin(x) * jnp.cos(y)  # noqa: E731
vi_f = lambda x, y, z: 0.1 * jnp.sin(2 * x) * jnp.cos(y)  # noqa: E731
w_f = lambda x, y, z: vi_f(x, y, z) - (-jax.grad(jax.grad(a_f, 0), 0)(x, y, z) + MMS_BETA * MMS_MU * n_f(x, y, z) * a_f(x, y, z)) / (MMS_BETA * n_f(x, y, z))  # noqa: E731
ve_f = lambda x, y, z: w_f(x, y, z) + MMS_MU * a_f(x, y, z)  # noqa: E731
d = lambda f, i: (lambda x, y, z: jax.grad(f, i)(x, y, z))  # noqa: E731
bracket = lambda f, h: (lambda x, y, z: d(f, 0)(x, y, z) * d(h, 1)(x, y, z) - d(f, 1)(x, y, z) * d(h, 0)(x, y, z))  # noqa: E731
gpar = lambda f: (lambda x, y, z: d(f, 2)(x, y, z) + bracket(a_f, f)(x, y, z))  # noqa: E731
dw_f = lambda x, y, z: -bracket(phi_f, ve_f)(x, y, z) + MMS_MU * (gpar(phi_f)(x, y, z) - gpar(n_f)(x, y, z) / n_f(x, y, z))  # noqa: E731
errors = []
for n in MMS_SIZES:
    g, kw, _, bcs = slab((n, n, n), (np.pi, 2 * np.pi, 2 * np.pi), beta=MMS_BETA, mu=MMS_MU, te=1.0, iterations=12)
    cx, cy, cz = (jnp.asarray(a.centers) for a in (g.grid.x, g.grid.y, g.grid.z))
    field = lambda f: jax.vmap(jax.vmap(jax.vmap(f, (None, None, 0)), (None, 0, None)), (0, None, None))(cx, cy, cz)  # noqa: E731
    state = Fci4FieldState(density=field(n_f), omega=field(lambda x, y, z: jax.grad(jax.grad(phi_f, 0), 0)(x, y, z)),
                           v_ion_parallel=field(vi_f), v_electron_parallel=field(w_f))
    a_num = kw["electromagnetic"].solve_apar(state.density, state.v_ion_parallel, state.v_electron_parallel, face_bc=bcs.v_electron_parallel_face_bc)
    errors.append((float(jnp.max(jnp.abs(a_num - field(a_f)))), float(jnp.max(jnp.abs(rhs(state, kw)[0].v_electron_parallel - field(dw_f))))))
    print(f"N = {n}: error of A_par {errors[-1][0]:.2e}, of dw/dt {errors[-1][1]:.2e}")
errors = np.array(errors)

fig, (left, middle, right) = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
left.loglog(BETAS, differences, "o-", label="measured")
left.loglog(BETAS, np.array(BETAS) * differences[-1] / BETAS[-1], "k--", label=r"$\propto \beta$")
left.set(xlabel=r"$\beta$", ylabel="relative RHS difference", title=r"$\beta \to 0$: electrostatic limit")
left.legend(frameon=False)
t = np.linspace(0, 1, STEPS + 1)
for k, label in enumerate(("E x B", "magnetic", "electron kinetic")):
    middle.plot(t, history[:, k] / history[0].sum(), label=label)
middle.plot(t, history.sum(1) / history[0].sum(), "k", label="sum")
middle.set(xlabel="t / period", ylabel="energy / initial energy", title="energy over one Alfven period")
middle.legend(frameon=False, fontsize=7)
right.loglog(MMS_SIZES, errors[:, 0], "o-", label=r"$A_\parallel$")
right.loglog(MMS_SIZES, errors[:, 1], "s-", label=r"$dw/dt$")
right.loglog(MMS_SIZES, errors[0, 1] * (MMS_SIZES[0] / np.array(MMS_SIZES)) ** 2, "k--", label=r"$h^2$")
right.set(xlabel="N (cells per direction)", ylabel="max error", title="manufactured solution")
right.legend(frameon=False)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_DIR / "electromagnetic_verification.png", dpi=100)
print(f"wrote {OUTPUT_DIR / 'electromagnetic_verification.png'}")
