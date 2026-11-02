"""Phase A diagnostics of docs/plan.md for the HSX FCI backend (diagnosis only, no model change).

Edit the constants and run ``python scripts/fci_diagnostics.py``.  The script runs the deck up to the
point where the sharded model can be built, captures that model and then reports

A1  adjointness of the parallel operator pairs (G, D) as dense matrices, spectra of the linearised Ve-Te
    and Ve-phi-omega blocks with and without the sheath rows (advection and heat-blob tests: see plan.md);
A2  host synchronisations and callbacks of one step;
A3  electron energy rates of the state, term by term;
A4 (sheath heat transmission) is tests/test_sheath_heat_transmission.py.

Snapshot ``STATE`` is a restart file written by ``snapshot_times`` of a default run (step 100 of 200).
"""

import dataclasses
import sys
import time
from pathlib import Path

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from jax.sharding import PartitionSpec as P

from drbx import deck
from drbx.fci_braginskii.native.fci_operators import (
    local_grad_parallel_op_fci_compatible_from_q,
    local_parallel_div_b_fci_from_q_op,
    local_parallel_q_flux_div_fci_op,
)
from drbx.fci_braginskii.native.fci_parallel_conduction import conductivity
from drbx.fci_braginskii.geometry.fci_geometry import FCI_DEP_PHYSICAL_BOUNDARY, build_local_fci_stencil_from_field

jax.config.update("jax_enable_x64", True)

DECK = "examples/inputs/hsx_fci_blob.toml"  # [geometry] table: shipped coils and vessel, built once and cached
GEOMETRY_CACHE = "output/hsx_geometry"
STATE = "output/phaseA/hsx_snapshot_0.075.npz"  # state to linearise around (None: the initial blob)
DT = 7.5e-4  # default step of the 200-step run
EIGS = 6  # eigenvalues kept per block
SHEATH_CHANGES = {"with sheath rows": {"wall_closure": "bohm"}, "without sheath rows": {"wall_closure": "zero-flow"}}


class Captured(Exception):
    pass


def capture(**changes):
    """Run the deck until the first shard_map of ``run_full_eb`` and keep that frame's locals."""
    name, cfg = deck.load_deck(DECK)
    cfg["geometry"] = dataclasses.replace(cfg["geometry"], cache_dir=Path(GEOMETRY_CACHE))
    cfg["fci_braginskii"] = dataclasses.replace(cfg["fci_braginskii"], restart_from=STATE, final_time=0.15, num_steps=200, **changes)
    real = jax.shard_map

    def stop(fun, **kwargs):
        raise Captured(sys._getframe(1).f_locals)

    jax.shard_map = stop
    try:
        deck.BACKENDS[name].run(cfg, print)
    except Captured as caught:
        return caught.args[0]
    finally:
        jax.shard_map = real
    raise RuntimeError("run_full_eb did not reach its first shard_map")


def on(cap, fn):
    """``fn(model, state, *args)`` inside the one-device shard_map of the captured run (jitted)."""
    spatial, geo = cap["spatial_spec"], cap["geometry_spec"]

    def compiled(state, *args):
        body = lambda st, cf, mf, cv, *a: fn(cap["build_local_model"](cf, mf, cv), st, *a)
        specs = (cap["state_spec"], geo, geo, geo) + (spatial,) * len(args)
        mapped = jax.shard_map(body, mesh=cap["mesh"], in_specs=specs, out_specs=P(), check_vma=False)
        return mapped(state, cap["cell_fields"], cap["map_fields"], cap["control_volume_fields"], *args)

    return jax.jit(compiled)


def dense(op, shape):
    """Matrix of a linear map on ``shape`` arrays (column j = op(e_j))."""
    n = int(np.prod(shape))
    eye = jnp.eye(n).reshape((n,) + shape)
    return jax.lax.map(lambda e: op(e).reshape(-1), eye, batch_size=64).T


def homogeneous_halo_ops(model, face_bc):
    """Legacy mapped q-flux divergence and compatible gradient with homogeneous wall data (linear parts)."""
    ctx = model._stencil_builder_context()
    phi_bc = face_bc.phi
    zero_bc = dataclasses.replace(phi_bc, value_x=0 * phi_bc.value_x, value_y=0 * phi_bc.value_y, value_z=0 * phi_bc.value_z)
    bmag = jnp.maximum(model.geometry.cell_bfield.Bmag_owned, 1e-30)
    inv_b = model._fci_prepare_inverse_b(face_bc, ctx)
    div_b = local_parallel_div_b_fci_from_q_op(
        inv_b[0], model.geometry, context=ctx, forward_remote_q_values=inv_b[1], backward_remote_q_values=inv_b[2]
    )

    def prepared(values):
        halo = model._prepare_fine_storage_halo(values / bmag, zero_bc)
        return (halo,) + tuple(model._fci_remote_values(halo, ctx))

    def divergence(values):
        halo, fwd, bwd = prepared(values)
        return local_parallel_q_flux_div_fci_op(
            halo, model.geometry, context=ctx, forward_remote_q_values=fwd, backward_remote_q_values=bwd
        )

    def gradient(values):
        halo, fwd, bwd = prepared(values)
        return local_grad_parallel_op_fci_compatible_from_q(
            halo, model.geometry, context=ctx, field_owned=values, div_b=div_b,
            forward_remote_q_values=fwd, backward_remote_q_values=bwd,
        )

    return divergence, gradient


def forward_values(model, face_bc, values):
    """``(center, plus, minus, dx_plus)`` of the mapped field-line stencil of a field (homogeneous wall data)."""
    ctx = model._stencil_builder_context()
    zero_bc = dataclasses.replace(face_bc.phi, value_x=0 * face_bc.phi.value_x, value_y=0 * face_bc.phi.value_y,
                                  value_z=0 * face_bc.phi.value_z)
    halo = model._prepare_fine_storage_halo(values, zero_bc)
    fwd, bwd = model._fci_remote_values(halo, ctx)
    st = build_local_fci_stencil_from_field(halo, model.geometry, ctx, forward_remote_values=fwd, backward_remote_values=bwd)
    return st.center, st.plus, st.minus, st.dx_plus


def operator_matrices(model, state):
    """Dense G and D of every pair in use, plus masks, volumes and the temperature conductivity."""
    shape = model.geometry.owned_shape
    face_bc, ctx = model._face_bcs(state), model._stencil_builder_context()
    zero = jnp.zeros(shape)
    gs, ds, core = model._fci_support_core_pair(face_bc=face_bc, context=ctx)
    gp, dp, ptarget = model._fci_current_phi_boundary_pair(
        face_bc=face_bc, context=ctx, wall_endpoint_current_values=(zero, zero)
    )
    dl, gl = homogeneous_halo_ops(model, face_bc)
    cvg = model.control_volume_geometry
    active = model.geometry.active_cell_mask_owned
    if cvg is not None:
        active = active & cvg.cells.is_active_owner
    return {
        "Gs": dense(gs, shape), "Ds": dense(ds, shape), "Gp": dense(gp, shape), "Dp": dense(dp, shape),
        "Gl": dense(gl, shape), "Dl": dense(dl, shape),
        "core": core, "ptarget": ptarget, "active": active, "volume": model._fci_pair_cell_mass(),
        "wall": (model.geometry.maps.backward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY)
        | (model.geometry.maps.forward.endpoint_kind == FCI_DEP_PHYSICAL_BOUNDARY),
    }


def adjoint_report(name, g, d, v):
    """||D + V^-1 G^T V||/||D||, sum V D(f) for random f, max|G(1)|/||G||_inf, and G applied to a constant."""
    rng = np.random.default_rng(0)
    gt = (g.T * v[None, :]) / v[:, None]
    f = rng.standard_normal((len(v), 8))
    flux = v @ (d @ f)
    scale = np.abs(v)[:, None] * np.abs(d @ f)
    print(
        f"  {name:<34s} |D+V^-1G^TV|/|D| = {np.linalg.norm(d + gt) / np.linalg.norm(d):.3e}   "
        f"sumV.D(f)/sum V|D f| = {np.max(np.abs(flux) / scale.sum(0)):.3e}   "
        f"max|G(1)|/|G|_inf = {np.max(np.abs(g.sum(1))) / np.max(np.abs(g).sum(1)):.3e}"
    )

# A1: operators
cap = capture()
state0 = cap["state"]
shape = tuple(int(n) for n in state0.density.shape)
mats = jax.tree.map(np.asarray, on(cap, operator_matrices)(state0))
act = np.flatnonzero(mats["active"].ravel())
sub = lambda m: m[np.ix_(act, act)]
vol = mats["volume"].ravel()[act]
core = mats["core"].ravel()[act].astype(float)
Gs, Ds, Gp, Dp, Gl, Dl = (sub(mats[k]) for k in ("Gs", "Ds", "Gp", "Dp", "Gl", "Dl"))
Gc, Dc = Gs + (1 - core)[:, None] * Gl, Ds + (1 - core)[:, None] * Dl  # support core + legacy closure rows
print(f"[A1] {len(act)} active cells of {np.prod(shape)}; support-core rows {int(core.sum())}")
print("[A1] adjointness of the parallel operator pairs (homogeneous wall data):")
adjoint_report("support pair (sheath, conduction)", Gs, Ds, vol)
adjoint_report("support core + legacy wall rows", Gc, Dc, vol)
adjoint_report("legacy mapped (G_compat, D_flux)", Gl, Dl, vol)
adjoint_report("current/phi pair", Gp, Dp, vol)
vd = vol[:, None] * (Ds @ Gs)  # conduction operator D(kappa G .) with kappa = 1; kappa > 0 keeps the structure
print(
    f"  conduction D G (kappa=1): |V(DG)-(V DG)^T|/|V DG| = {np.linalg.norm(vd - vd.T) / np.linalg.norm(vd):.3e}, "
    f"max eig sym(V D G) = {np.max(sla.eigvalsh(0.5 * (vd + vd.T))):.3e} (<= 0 for a dissipative operator)"
)

# A1: spectra of the linearised blocks
def total_rate(model, x):
    """Explicit RHS plus the selected wall-leg rows (implicit in the time stepper) of a state."""
    rate = model.evaluate_stage(x, phi_owned=x.phi, short_leg_selection_dt=DT)
    info = model.apply_short_leg_implicit_material_step(x, DT, phi_owned=x.phi, return_increment=True)[2]
    wall = info["selected_complete_residual_owner"]
    return rate.replace(density=rate.density + wall[..., 0], Te=rate.Te + wall[..., 1], Ti=rate.Ti + wall[..., 2],
                        Vi=rate.Vi + wall[..., 3], Ve=rate.Ve + wall[..., 4])


def conduction_rate(model, base, temperature):
    """Frozen-kappa Spitzer-Harm rate D(kappa G T)/(3/2 n) of the electron temperature (the implicit step's operator)."""
    gradient, divergence, _ = model._fci_support_core_pair(
        face_bc=model._face_bcs(base), context=model._stencil_builder_context()
    )
    kappa = conductivity(model._expand_owner_field_for_stencil(base.Te), model._expand_owner_field_for_stencil(base.density),
                         float(model.parameters.electron_conduction_kappa0), float(model.parameters.mi_over_me),
                         float(model.parameters.heat_flux_limiter), gradient)
    return model._project_fine_cell_term(divergence(kappa * gradient(model._expand_owner_field_for_stencil(temperature)))) / (
        1.5 * jnp.maximum(base.density, 1e-30))


def block_jvp(model, base, dfirst, dsecond, block):
    """Jacobian-vector product of the (Ve, Te) or (Ve, phi, omega) block rate at ``base``."""
    if block == "Ve-Te":
        def rate_te_ve(te, ve):
            r = total_rate(model, base.replace(Te=te, Ve=ve))
            return r.Te + (conduction_rate(model, base, te) if model.parallel_conduction else 0.0), r.Ve

        return jax.jvp(rate_te_ve, (base.Te, base.Ve), (dfirst, dsecond))[1]
    pair_off = dataclasses.replace(model, implicit_current_phi_pair=False)

    def rate(ve, omega):
        x = base.replace(Ve=ve, vorticity=omega)
        x = x.replace(phi=pair_off.reconstruct_phi(x))
        r = total_rate(pair_off, x)
        return r.Ve, r.vorticity

    return jax.jvp(rate, (base.Ve, base.vorticity), (dfirst, dsecond))[1]


def eigen_block(cap, base, block):
    """Largest-real-part eigenpairs of the dense block Jacobian (ARPACK did not converge: spectrum is stiff and oscillatory)."""
    n = len(act)
    scatter = lambda x: jnp.zeros(shape).ravel().at[act].set(x).reshape(shape)

    def columns(model, state):
        def column(i):
            e = jnp.zeros(2 * n).at[i].set(1.0)
            a, b = block_jvp(model, state, scatter(e[:n]), scatter(e[n:]), block)
            return jnp.concatenate([a.ravel()[act], b.ravel()[act]])

        return jax.lax.map(column, jnp.arange(2 * n), batch_size=16)

    jac = np.asarray(on(cap, columns)(base)).T
    values = sla.eigvals(jac)
    top = values[np.argsort(-values.real)[:EIGS]]
    vectors = []
    for lam in top:  # inverse iteration for the eigenvector
        v = np.random.default_rng(0).standard_normal(2 * n).astype(complex)
        for _ in range(3):
            v = np.linalg.solve(jac - (lam + 1e-7 * abs(lam)) * np.eye(2 * n), v)
            v /= np.linalg.norm(v)
        vectors.append(v)
    return top, np.array(vectors).T, values.real.max(), int(np.sum(values.real > 0))


def along_b_correlation(plus, base, field):
    """<f, f_plus>/<f, f> of the forward field-line neighbour value: +1 smooth along b, -1 for period 2."""
    f = np.zeros(shape).ravel()
    f[act] = field
    p = np.asarray(plus(base, jnp.asarray(f.reshape(shape)))).ravel()[act]
    return np.sum(vol * field * p) / max(np.sum(vol * field**2), 1e-300)

wall_cells = mats["wall"].ravel()[act]
for label, changes in SHEATH_CHANGES.items():
    cap_b = cap if label.startswith("with ") else capture(**changes)
    base = cap_b["state"]
    for block, names in (("Ve-Te", ("Te", "Ve")), ("Ve-phi-omega", ("Ve", "omega"))):
        values, vectors, abscissa, unstable = eigen_block(cap_b, base, block)
        plus = on(cap_b, lambda model, state, f: forward_values(model, model._face_bcs(state), f)[1])
        n = len(act)
        print(f"[A1] {label}, {block} block: spectral abscissa {abscissa:+.3e}, {unstable} of {2 * n} eigenvalues with Re > 0; the {len(values)} of largest real part:")
        for lam, vec in sorted(zip(values, vectors.T), key=lambda p: -p[0].real):
            parts = (vec[:n], vec[n:])
            total = sum(np.sum(vol * np.abs(p) ** 2) for p in parts)
            share = [np.sum(vol * np.abs(p) ** 2) / total for p in parts]
            rho = [(along_b_correlation(plus, base, p.real) + along_b_correlation(plus, base, p.imag)) / 2 for p in parts]
            near_wall = [np.sum((vol * np.abs(p) ** 2)[wall_cells]) / np.sum(vol * np.abs(p) ** 2) for p in parts]
            print(f"     lambda = {lam.real:+.3e}{lam.imag:+.3e}i  |lam| dt = {abs(lam) * DT:.2f}  "
                  f"energy share {names[0]}/{names[1]} = {share[0]:.2f}/{share[1]:.2f}  "
                  f"along-b correlation {rho[0]:+.2f}/{rho[1]:+.2f} (-1: period 2)  wall-cell share {near_wall[0]:.2f}/{near_wall[1]:.2f}")

# A2: time and host synchronisations of one step
def callback_count(jaxpr):
    """Host callbacks (pure/io/debug) in a jaxpr, recursively."""
    total = 0
    for eqn in jaxpr.eqns:
        total += "callback" in eqn.primitive.name
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            total += callback_count(sub)
    return total

phases = {
    "explicit RHS": lambda model, x: total_rate(model, x),
    "implicit wall legs": lambda model, x: model.apply_short_leg_implicit_material_step(x, DT, phi_owned=x.phi),
    "Spitzer conduction solve": lambda model, x: model.solve_implicit_parallel_conduction(x, solve_dt=DT)[1],
    "current/phi pair solve": lambda model, x: model.solve_implicit_current_phi_pair(x, solve_dt=DT)[1],
    "phi reconstruction": lambda model, x: model.reconstruct_phi(x),
}
print(f"[A2] one 16^3 stage of each phase on {jax.devices()[0].platform} (compiled, median of 5 after warm-up):")
for name, fn in phases.items():
    f = on(cap, fn)
    jax.block_until_ready(f(state0))
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(f(state0))
        times.append(time.perf_counter() - t0)
    count = callback_count(jax.make_jaxpr(f)(state0).jaxpr)
    print(f"     {name:<26s} {np.median(times):7.3f} s   host callbacks in the jaxpr: {count}")

counts = {}
for attribute in ("block_until_ready", "__array__", "__float__", "__int__", "__bool__", "item"):
    original = getattr(type(state0.density), attribute)

    def counted(self, *a, _o=original, _n=attribute, **k):
        counts[_n] = counts.get(_n, 0) + 1
        return _o(self, *a, **k)

    setattr(type(state0.density), attribute, counted)
steps = {}
for n_steps in (2, 6):
    counts.clear()
    name, cfg = deck.load_deck(DECK)
    cfg["geometry"] = dataclasses.replace(cfg["geometry"], cache_dir=Path(GEOMETRY_CACHE))
    cfg["fci_braginskii"] = dataclasses.replace(cfg["fci_braginskii"], restart_from=STATE, num_steps=n_steps,
                                                final_time=0.075 + n_steps * DT, save_every=n_steps, output=Path("output/phaseA/a2"))
    deck.BACKENDS[name].run(cfg, lambda m: None)
    steps[n_steps] = dict(counts)
per_step = {k: (steps[6].get(k, 0) - steps[2].get(k, 0)) / 4 for k in steps[6]}
print(f"[A2] host syncs per step from run_full_eb (jax.Array methods, difference of a 6- and a 2-step run): {per_step}")

# A3: electron energy budget
from drbx.fci_braginskii.native.fci_drb_EB_rhs import RHS_TERM_NAMES

TERM_TO_DENSITY = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5}  # Te term slot -> density slot with the same name


def electron_energy_rates(model, x):
    """Volume integrals of d/dt (3/2 n Te) by term, the wall legs, conduction, wall flux and the would-be Ohmic heating."""
    _, mass = model._ledger_masses()
    _, terms = model.evaluate_stage(x, phi_owned=x.phi, return_rhs_term_fields=True, short_leg_selection_dt=DT)
    energy = lambda dn, dte: jnp.sum(mass * 1.5 * (x.density * dte + x.Te * dn))
    rates = [energy(terms[0][TERM_TO_DENSITY[k]] if k in TERM_TO_DENSITY else 0.0, terms[1][k]) for k in range(7)]
    wall = model.apply_short_leg_implicit_material_step(x, DT, phi_owned=x.phi, return_increment=True)[2]
    w = wall["selected_complete_residual_owner"]
    ledger = model.wall_flux_ledger(x)
    kappa0 = float(model.parameters.electron_conduction_kappa0)
    ohmic = 1.612 * x.density**2 * (x.Vi - x.Ve) ** 2 / (kappa0 * jnp.maximum(x.Te, 1e-6) ** 1.5) if kappa0 else 0.0 * x.Te
    return jnp.stack(rates + [
        energy(w[..., 0], w[..., 1]),
        jnp.sum(mass * 1.5 * x.density * conduction_rate(model, x, x.Te)) if model.parallel_conduction else 0.0,
        ledger[2], ledger[9], ledger[12], jnp.sum(mass * ohmic),
    ])

config = cap["parameters"]
names = [RHS_TERM_NAMES[1][k] for k in range(7)] + ["implicit wall legs", "Spitzer conduction", "wall flux (ledger, outward)",
                                                     "W_e inventory", "parallel work source S_e", "Ohmic J^2/sigma if present"]
rates = np.asarray(on(cap, electron_energy_rates)(state0))
print(f"[A3] electron energy rates at the snapshot (n0 Te0 c_s0 L^2 per normalised time); Ve_nu = {config.Ve_nu}, "
      f"parallel_diffusion = {config.electron_temperature_chi_parallel}, perp D = {config.electron_temperature_D_perp}")
sink, power = abs(rates[7]), rates[:7][rates[:7] > 0].sum()
print(f"     input power (sum of the positive Te terms) = {power:+.4e}")
for name, value in zip(names, rates):
    print(f"     {name:<32s} {value:+.4e}   {100 * value / sink:+8.2f}% of the wall-leg loss   {100 * value / power:+8.2f}% of input power")
