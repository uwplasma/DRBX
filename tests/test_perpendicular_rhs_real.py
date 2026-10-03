"""Combined perpendicular RHS on the real N32 owner closure (P08 step 3, tasks 3.2-3.4; slow).

The bounded 12-owner closure of ``p_shared.owner_closure`` (real rows and geometry) is lowered into a
``PerpendicularPlan`` by ``p_shared.jax_replay.JaxOwnerClosure`` (which also evaluates the P06N boundary data at the
plan's tables). The state is a real five-field state: the columns and kinds of the first ``main_*`` P06N variant of
the frozen catalogue (mixed Dirichlet / Neumann), its boundary data, ``phi`` the variant's fifth column.

* G3.1 (consistency): the combined terms equal the separate operator calls (``p05_terms`` with pairs ``(phi, g)`` scaled
  by ``1/rho_star``, ``p06_action`` total + correction, ``-D * p07_action``) at the closure owners.
* G3.4: eager == jit bitwise; JVP vs central finite difference; cost of one combined call vs the three separate calls
  (median time after warm-up, XLA-compiled temporary memory).
"""
from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]  # .../HSX drbx
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

GEOMETRY = WORKSPACE / "geometry_artifacts/rlp_convergence_32_48_64_20260917"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
N = 32
CAMPAIGNS = ("p06n",)
REPORT: dict = {}
RHO, TAU, DIFF = 0.05, 1.0, {"density": 1.0e-2, "Te": 2.0e-2, "Ti": 3.0e-2, "vorticity": 4.0e-2}
PRODUCTION = ("poisson_bracket", "curvature", "perpendicular_diffusion")

pytestmark = pytest.mark.slow

_have = ((GEOMETRY / f"{N}x{N}x{N}" / "base_geometry.npz").is_file() and SIDECAR.is_file())
needs_inputs = pytest.mark.skipif(not _have, reason="HSX N32 geometry/sidecar inputs are unavailable")


@pytest.fixture(scope="module")
def setup():
    from drbx.native.fci_perpendicular_p06_operator import bc_columns
    from drbx.native.fci_perpendicular_rhs import FIELDS, PHI, PerpendicularParams
    from p_shared import jax_replay as jr
    from p_shared import owner_closure as oc
    from p_shared.replay_support import DEFAULT_PATHS, build_environment
    from tests import p06n_owner_values_live as live

    env = build_environment(n=N, input_root=WORKSPACE, sidecar_path=SIDECAR, curvature="fd", face_quadrature="q3", inner_support="profile7")
    owners = np.asarray(sorted(set(oc.select_owners(env.t, env.census).values())), dtype=np.int64)
    built = oc.build_owner_rows(env, owners.tolist(), provider=oc.load_provider_for_env(SIDECAR, curvature="fd", face_quadrature="q3"))
    oracle = live.load_oracle_owner_values(env, dict(DEFAULT_PATHS), CAMPAIGNS)     # P06N owner averages on the fly
    closure = jr.JaxOwnerClosure(env, built, CAMPAIGNS, oracle)
    adapter = closure.adapters["p06n"]
    variant = next(v for v in adapter.variant_names if v.startswith("main_"))
    rec = adapter.reconstructions[variant]
    columns = np.asarray(rec.columns)
    layout = (*FIELDS, PHI)
    state = {name: np.asarray(adapter.owner_values[:, c]) for name, c in zip(FIELDS, columns[:4])}
    phi = np.asarray(adapter.owner_values[:, columns[4]])
    return dict(closure=closure, plan=closure.plan, owners=owners, variant=variant, state=state, phi=phi,
                bc=bc_columns(closure.bc["p06n"], columns), kinds=dict(zip(layout, rec.field_kinds)),
                params=PerpendicularParams(rho_star=RHO, tau=TAU, diffusion=dict(DIFF)),
                kinds_tuple=tuple(rec.field_kinds))


@pytest.fixture(scope="module")
def rhs(setup):
    from drbx.native.fci_perpendicular_rhs import perpendicular_rhs
    s = setup
    return perpendicular_rhs(s["plan"], s["state"], s["phi"], s["bc"], s["kinds"], s["params"])


def _separate(s):
    """The three separate operator calls on the same five columns (layout ``density, Te, Ti, vorticity, phi``)."""
    from drbx.native.fci_perpendicular_p05_operator import p05_terms
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_rhs import FIELDS
    fields = jnp.stack([jnp.asarray(s["state"][f]) for f in FIELDS] + [jnp.asarray(s["phi"])], axis=1)
    kinds, bc = s["kinds_tuple"], s["bc"]
    t = p05_terms(s["plan"], fields, bc, kinds, [(4, i) for i in range(4)])
    a = p06_action(s["plan"], fields, bc, kinds)
    p7 = p07_action(s["plan"], fields[:, :4], bc_columns(bc, np.arange(4)), kinds[:4])
    d = jnp.asarray([DIFF[f] for f in FIELDS])
    return dict(poisson_bracket=(t.centered_owner + t.jump_owner) / RHO, curvature=a.total + a.correction,
                perpendicular_diffusion=-d[None, :] * p7, q1=a.total, correction=a.correction)


@needs_inputs
def test_real_closure_state_is_the_mixed_kind_main_variant(setup):
    s = setup
    assert s["variant"].startswith("main_")
    assert "neumann" in s["kinds_tuple"] and "dirichlet" in s["kinds_tuple"]
    plan = s["plan"]
    assert plan.faces.wall.any() and plan.faces.has_missing_side and plan.cells.conditioned.any()
    REPORT["variant"] = s["variant"]; REPORT["kinds"] = s["kinds_tuple"]


@needs_inputs
def test_g31_combined_equals_the_separate_operator_calls(setup, rhs):
    from drbx.native.fci_perpendicular_rhs import FIELDS
    owners = setup["owners"]
    sep = _separate(setup)
    diffs, scales = {}, {}
    for i, f in enumerate(FIELDS):
        for key in PRODUCTION:
            mine, ref = np.asarray(rhs.terms[f][key])[owners], np.asarray(sep[key])[owners, i]
            diffs[(f, key)] = float(np.max(np.abs(mine - ref)))
            scales[(f, key)] = float(np.max(np.abs(ref)))
        for key, ref in (("curvature_q1", sep["q1"]), ("curvature_correction", sep["correction"])):
            mine = np.asarray(rhs.detail[f][key])[owners]
            diffs[(f, key)] = float(np.max(np.abs(mine - np.asarray(ref)[owners, i])))
            scales[(f, key)] = float(np.max(np.abs(np.asarray(ref)[owners, i])))
    worst_abs = max(diffs.values())
    worst_rel = max(diffs[k] / max(scales[k], 1e-300) for k in diffs)
    REPORT["g31"] = dict(max_abs=worst_abs, max_rel_to_own_scale=worst_rel,
                         term_scales={f"{f}/{k}": v for (f, k), v in scales.items() if k in PRODUCTION})
    print("\nG3.1 real N32 (%s): max |combined - separate| = %.3g abs, %.3g of the term's scale; term scales %s"
          % (setup["variant"], worst_abs, worst_rel, {f"{f}/{k}": round(v, 4) for (f, k), v in scales.items()
                                                       if k in PRODUCTION}))
    assert worst_rel <= 1e-13, diffs
    # every term is non-trivial at the closure owners
    assert min(scales[(f, k)] for f in FIELDS for k in PRODUCTION) > 0
    print("P06 q3 counters:", {k: int(v) for k, v in rhs.diagnostics.items() if k != "antisymmetry"})


@needs_inputs
def test_g34_real_plan_eager_equals_jit(setup, rhs):
    from drbx.native.fci_perpendicular_rhs import perpendicular_rhs
    s = setup
    kinds = s["kinds"]
    jitted = jax.jit(lambda p, st, ph, b, prm: perpendicular_rhs(p, st, ph, b, kinds, prm, raw_pairs=[("density", "Te")]))(
        s["plan"], {k: jnp.asarray(v) for k, v in s["state"].items()}, jnp.asarray(s["phi"]), s["bc"], s["params"])
    eager = perpendicular_rhs(s["plan"], s["state"], s["phi"], s["bc"], kinds, s["params"], raw_pairs=[("density", "Te")])
    la, lb = jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted)
    assert len(la) == len(lb)
    worst = 0.0
    for a, b in zip(la, lb):
        # Roundoff, not bitwise, since the face state is pruned to the columns P05 / P06 read (P09 fix 1): XLA fuses
        # the pruned face producers into the P05 jump differently standalone and inside an outer jit (<= 1 ulp).
        a, b = np.asarray(a), np.asarray(b)
        scale = max(float(np.max(np.abs(b))), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b))) / scale)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-14 * scale)
    REPORT["eager_equals_jit_max_rel_diff"] = worst


@needs_inputs
def test_g34_real_jvp_matches_central_finite_difference(setup):
    from drbx.native.fci_perpendicular_rhs import FIELDS, perpendicular_rhs
    s = setup
    owners = jnp.asarray(s["owners"])
    x = jnp.stack([jnp.asarray(s["state"][f]) for f in FIELDS] + [jnp.asarray(s["phi"])], axis=1)
    tangent = jnp.asarray(np.random.default_rng(9).normal(size=x.shape))

    def fun(z):
        out = perpendicular_rhs(s["plan"], {f: z[:, i] for i, f in enumerate(FIELDS)}, z[:, 4], s["bc"], s["kinds"],
                                s["params"])
        return {f: {t: a[owners] for t, a in out.terms[f].items()} for f in FIELDS}

    _, jvp = jax.jvp(fun, (x,), (tangent,))
    errors = {}
    for h in (1e-6, 1e-7):
        plus, minus = fun(x + h * tangent), fun(x - h * tangent)
        for f in FIELDS:
            for t in PRODUCTION:
                fd = np.asarray((plus[f][t] - minus[f][t]) / (2 * h))
                j = np.asarray(jvp[f][t])
                errors[(h, f, t)] = float(np.max(np.abs(j - fd)) / np.max(np.abs(fd)))
    by_h = {h: max(v for (hh, _, _), v in errors.items() if hh == h) for h in (1e-6, 1e-7)}
    by_term = {t: max(v for (h, _, tt), v in errors.items() if tt == t and h == 1e-6) for t in PRODUCTION}
    REPORT["jvp_vs_fd"] = dict(by_step=by_h, by_term_h1e6=by_term)
    print("\nG3.4 real N32 JVP vs central FD, max relative error by step:", by_h, "by term (h=1e-6):", by_term)
    assert by_h[1e-6] <= 1e-6, errors


def _median_time(fn, repeats=7):
    jax.block_until_ready(fn())                                       # warm-up (compile)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - started)
    return statistics.median(times), min(times)


def _memory(jitted, *args):
    m = jitted.lower(*args).compile().memory_analysis()
    return dict(temp=int(m.temp_size_in_bytes), output=int(m.output_size_in_bytes),
                argument=int(m.argument_size_in_bytes))


@needs_inputs
def test_g34_cost_combined_vs_separate(setup):
    """One combined call vs the three separate operator calls (each with its own reconstruction)."""
    from drbx.native.fci_perpendicular_p05_operator import p05_terms
    from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action
    from drbx.native.fci_perpendicular_p07_operator import p07_action
    from drbx.native.fci_perpendicular_rhs import FIELDS, perpendicular_rhs
    s = setup
    plan, bc, kinds_t = s["plan"], s["bc"], s["kinds_tuple"]
    fields = jnp.stack([jnp.asarray(s["state"][f]) for f in FIELDS] + [jnp.asarray(s["phi"])], axis=1)
    pairs = [(4, i) for i in range(4)]
    bc4 = bc_columns(bc, np.arange(4))
    d = jnp.asarray([DIFF[f] for f in FIELDS])

    def combined():
        out = perpendicular_rhs(plan, s["state"], s["phi"], bc, s["kinds"], s["params"])
        return out.total

    def separate():
        t = p05_terms(plan, fields, bc, kinds_t, pairs)
        a = p06_action(plan, fields, bc, kinds_t)
        p7 = p07_action(plan, fields[:, :4], bc4, kinds_t[:4])
        return (t.centered_owner + t.jump_owner) / RHO + a.total + a.correction - d[None, :] * p7

    j_p05 = jax.jit(lambda p, f, b: p05_terms(p, f, b, kinds_t, pairs))
    j_p06 = jax.jit(lambda p, f, b: p06_action(p, f, b, kinds_t))
    j_p07 = jax.jit(lambda p, f, b: p07_action(p, f[:, :4], bc_columns(b, np.arange(4)), kinds_t[:4]))
    j_comb = jax.jit(lambda p, st, ph, b, prm: perpendicular_rhs(p, st, ph, b, s["kinds"], prm))
    state_j = {k: jnp.asarray(v) for k, v in s["state"].items()}
    phi_j = jnp.asarray(s["phi"])

    def combined_jit():
        return j_comb(plan, state_j, phi_j, bc, s["params"])

    def separate_jit():
        return j_p05(plan, fields, bc), j_p06(plan, fields, bc), j_p07(plan, fields, bc)

    times = {name: _median_time(fn, repeats=15) for name, fn in (
        ("combined_api", combined), ("separate_api", separate), ("combined_jit", combined_jit),
        ("separate_jit", separate_jit))}
    mem_sep = {name: _memory(j, plan, fields, bc) for name, j in (("p05", j_p05), ("p06", j_p06), ("p07", j_p07))}
    mem_comb = _memory(j_comb, plan, state_j, phi_j, bc, s["params"])
    cost = dict(times_median_min_s=times, memory_combined=mem_comb, memory_separate=mem_sep,
                temp_separate_peak_sequential=max(m["temp"] for m in mem_sep.values()),
                temp_separate_sum=sum(m["temp"] for m in mem_sep.values()),
                plan_cells=int(len(plan.cells.raw_ids)), plan_faces=int(len(plan.faces.census_row)))
    REPORT["cost"] = cost
    print("\nG3.4 cost on the N32 closure plan (%d cells, %d faces), median / min of 15 after warm-up:"
          % (cost["plan_cells"], cost["plan_faces"]))
    for name, (med, low) in times.items():
        print("  %-13s %.4f s / %.4f s" % (name, med, low))
    print("  ratio separate/combined: api x%.2f, jit-level x%.2f (median)"
          % (times["separate_api"][0] / times["combined_api"][0], times["separate_jit"][0] / times["combined_jit"][0]))
    print("  XLA temp bytes: combined %d; separate p05 %d + p06 %d + p07 %d (sum %d, max %d)"
          % (mem_comb["temp"], mem_sep["p05"]["temp"], mem_sep["p06"]["temp"], mem_sep["p07"]["temp"],
             cost["temp_separate_sum"], cost["temp_separate_peak_sequential"]))
    assert all(med > 0 for med, _ in times.values()) and mem_comb["temp"] >= 0
