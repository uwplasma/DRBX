"""The JAX stage of the P08 step-5.3 campaign: both arms of every pinned variant on the full grid, plus the solver gates.

One process per grid, after the reference stage:

1. :func:`prepare` -- ``build_environment`` with the pinned operator options, the step-4 artifact (``load_artifact``,
   options check, artifact-identity check against ``provenance/inputs.json``), the FULL plan (``lower_plan``: cells,
   faces and P07), the Dirichlet phi solver built once (``build_phi_solver``), and the frozen P06N owner values
   (``_load_oracle_owner_values``, path key ``p05n_p06n_upwind``: the exact owner average of every catalogue column).
2. per variant (:func:`run_variant`; a ``runner`` unit per variant, so a crash resumes):

   * the five columns ``n, Te, Ti, omega, phi`` of the variant (frozen owner values), its kinds (pinned in the
     configuration and checked against the catalogue) and the boundary data of the five columns at the plan's
     Dirichlet / Neumann points (``boundary_data_from_callables`` with the exact P06N state; the same data
     ``JaxOwnerClosure`` builds, restricted to the variant's five columns instead of the whole catalogue);
   * **prescribed arm**: ``perpendicular_rhs`` with the exact owner-averaged ``phi_bar``;
   * **solved arm**: ``psi = phi + tau Ti``; the P07 Dirichlet problem ``A psi_h = O(psi) - B g`` with ``g`` the exact
     trace of psi at the plan's Dirichlet points (value, tangential gradient; from the same five-column data) and
     ``O(psi)`` the reference stage's positive owner operator; ``phi_h = psi_h - tau Ti_bar``; ``perpendicular_rhs`` with
     ``phi_h`` (cold start; iterations and seconds recorded);
   * **solver gates**: (a) discrete consistency: ``rhs_c = A psi_bar + B g`` solved through the same solver must return
     ``psi_bar`` to a relative M-weighted error <= ``consistency_tolerance`` (this is the linear-solve error of the
     solver, reported per variant); (b) ``||O(psi) - (A psi_bar + B g)|| / ||O(psi)||`` (the operator's N - O for psi;
     report only, and a check of the sign of ``O``); (c) every solve converged.

The arrays of both arms (every field, every term, plus the potentials) are checkpointed per variant for the reduction.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from drbx.native.fci_perpendicular_phi_solver import build_phi_solver, phi_boundary_term, solve_phi   # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import (                                         # noqa: E402
    BoundaryData, boundary_data_from_callables)
from drbx.native.fci_perpendicular_rhs import FIELDS, PHI, PerpendicularParams, perpendicular_rhs      # noqa: E402

from p_shared import runner                                                                              # noqa: E402
from p_shared.replay_support import owner_weighted_l2                                                    # noqa: E402
from p08_step5_combined import references                                                                # noqa: E402
from p08_step5_combined.checkpoints import (                                                              # noqa: E402
    arm_key, artifact_sha, jax_identity, jsonable, variant_unit)

SCHEMA = "drbx.p08-step5-combined-jax.v1"
#: the terms of both arms' arrays (names of ``references.REF_TERMS``; ``curvature = material + remainder (+ q3 correction)``)
TERMS = references.REF_TERMS
_PRODUCTION = {"poisson_bracket": "poisson_bracket", "curvature": "curvature",
               "perpendicular_diffusion": "perpendicular_diffusion"}
_DETAIL = ("curvature_material", "curvature_remainder")
TI_SLOT = FIELDS.index("Ti")
PHI_SLOT = len(FIELDS)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Preparation: environment, artifact, full plan, phi solver, frozen owner values
# ---------------------------------------------------------------------------
def prepare(*, n: int, cfg: dict, options: dict, step4, input_root, paths: dict, inputs: dict, env=None,
            log=_log) -> SimpleNamespace:
    """Everything the variants share (``env`` may be passed to reuse an already built environment)."""
    import jax

    from p_shared import campaign_fields as cf
    from p_shared import replay_units as ru
    from p_shared.replay_support import build_environment
    from p_shared import perpendicular_reference_rhs as prr
    from p08_step2_global import replay
    from p08_step4_global.smoke import check_artifact_options

    step4 = Path(step4)
    seconds, mark = {}, [time.perf_counter()]

    def lap(name):
        now = time.perf_counter()
        seconds[name] = now - mark[0]
        mark[0] = now
        log(f"N{n}: {name} {seconds[name]:.1f} s")

    if env is None:
        env = build_environment(n=n, input_root=Path(input_root), sidecar_path=step4 / "localized_sidecar.json",
                                **options)
    lap("environment")
    artifact = replay.load_artifact(step4 / "artifact", n)
    check_artifact_options(artifact["identity"], options)
    art_sha = runner.digest(artifact["identity"])
    recorded = (inputs.get("grids", {}).get(str(n)) or {}).get("artifact_identity_sha256")
    if recorded is not None and recorded != art_sha:
        raise ValueError(f"artifact identity of N{n} differs from provenance/inputs.json")
    lap("load_artifact")
    plan, plan_summary = replay.lower_plan(env, artifact, step4 / "artifact", n, device_put=False)
    n_owners = int(len(plan.p07.owner_volume))
    if n_owners != len(env.t.vol):
        raise ValueError(f"P07 owner count {n_owners} != environment owner count {len(env.t.vol)}")
    lap("lower_plan")
    solver = build_phi_solver(plan, raw_to_owner=env.t.ro, n=n, factor_dtype=cfg["phi_solve_factor_dtype"],
                              rtol=float(cfg["phi_solve_rtol"]), restart=int(cfg["phi_solve_restart"]),
                              max_restarts=int(cfg["phi_solve_max_restarts"]))
    if not np.array_equal(np.asarray(solver.op.dirichlet_points), np.asarray(plan.dirichlet_points)):
        raise ValueError("the solver's Dirichlet points differ from the plan's")
    lap("phi_solver")
    if cfg.get("device_put_plan", True):
        plan = jax.device_put(plan)
    lap("device_put")
    oracle = ru._load_oracle_owner_values(env, dict(paths), ("p06n",))
    adapter = cf.P06NAdapter(env.ref, env.t.g.eta_period, oracle["p06n"]["owner_values"])
    lap("owner_values")
    return SimpleNamespace(
        n=int(n), env=env, plan=plan, solver=solver, adapter=adapter, owner_volume=np.asarray(env.t.vol),
        exact_state=lambda variant: prr.P06NState(adapter, variant), artifact_identity_sha256=art_sha,
        plan_summary=plan_summary, setup_seconds={**seconds, "solver": dict(solver.setup_seconds)},
        boundary_batch=int(cfg["boundary_batch"]))


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------
def variant_boundary_data(prep, variant: str) -> BoundaryData:
    """The five-column boundary data of ``variant`` at the plan's Dirichlet / Neumann points: exact traces and gradients
    of the variant's fields (``phi`` last), and the physical-normal data ``g_N = a . grad f`` of every column."""
    from p_shared.jax_replay import batched_callable

    state = prep.exact_state(variant)

    def dirichlet(points):
        value, gradient = state.values_gradients(points)
        return np.asarray(value).T, np.moveaxis(np.asarray(gradient), 0, -1)

    def normal(points):
        points = np.asarray(points, dtype=np.float64)
        a = np.asarray(prep.env.normal_coefficients(points))
        _v, gradient = state.values_gradients(points)
        return np.einsum("qa,qaf->qf", a, np.moveaxis(np.asarray(gradient), 0, -1))

    return boundary_data_from_callables(prep.plan, batched_callable(dirichlet, prep.boundary_batch),
                                        batched_callable(normal, prep.boundary_batch))


def psi_boundary_data(bc: BoundaryData, tau: float) -> BoundaryData:
    """The exact Dirichlet data of ``psi = phi + tau Ti`` (value and tangential gradient) from the five-column data."""
    value = np.asarray(bc.dirichlet_value)
    tangential = np.asarray(bc.dirichlet_tangential)
    psi_value = value[:, PHI_SLOT] + tau * value[:, TI_SLOT]
    psi_tangential = tangential[:, :, PHI_SLOT] + tau * tangential[:, :, TI_SLOT]
    return BoundaryData(psi_value[:, None], psi_tangential[:, :, None], None)


# ---------------------------------------------------------------------------
# One variant
# ---------------------------------------------------------------------------
def collect(out) -> dict:
    """``{(field, term): (n_owners,) float64}`` of a ``PerpendicularTerms``."""
    arrays = {}
    for field in FIELDS:
        for term in _PRODUCTION:
            arrays[(field, term)] = np.asarray(out.terms[field][_PRODUCTION[term]], dtype=np.float64)
        arrays[(field, "total")] = np.asarray(out.total[field], dtype=np.float64)
        for term in _DETAIL:
            arrays[(field, term)] = np.asarray(out.detail[field][term], dtype=np.float64)
    return arrays


def _solve_info(info: dict) -> dict:
    return {k: info.get(k) for k in ("iterations", "converged", "residual_norm", "relative_residual", "rhs_norm",
                                      "seconds")}


def run_variant(prep, variant: str, cfg: dict, *, omega_rhs=None, log=_log) -> tuple[dict, dict]:
    """Both arms and the solver gates of ``variant``; returns ``(arrays, info)``.

    ``omega_rhs`` is the reference stage's positive owner operator ``O(psi)`` (the right-hand side of the solved arm).
    ``None`` (the bounded preflight, which has the reference on a subset only) uses ``A psi_bar + B g`` instead, so the
    solved arm then reproduces the prescribed one up to the linear-solve error."""
    started = time.perf_counter()
    spec = cfg["variants"][variant]
    tau = float(cfg["params"]["tau"])
    params = PerpendicularParams(rho_star=float(cfg["params"]["rho_star"]), tau=tau,
                                 diffusion={f: float(cfg["params"]["diffusion"][f]) for f in FIELDS})
    rec = prep.adapter.reconstructions[variant]
    columns = np.asarray(rec.columns)
    kinds = tuple(rec.field_kinds)
    if kinds != tuple(spec["kinds"]):
        raise ValueError(f"variant {variant!r}: kinds {kinds} differ from the pinned {tuple(spec['kinds'])}")
    values = np.asarray(prep.adapter.owner_values)
    state = {name: np.asarray(values[:, c], dtype=np.float64) for name, c in zip(FIELDS, columns[:4])}
    phi_bar = np.asarray(values[:, columns[4]], dtype=np.float64)
    ti_bar = state["Ti"]
    psi_bar = phi_bar + tau * ti_bar
    volume = np.asarray(prep.owner_volume, dtype=np.float64)
    kind_map = dict(zip((*FIELDS, PHI), kinds))
    seconds: dict = {}

    def lap(name, since):
        seconds[name] = time.perf_counter() - since
        return time.perf_counter()

    t0 = time.perf_counter()
    bc5 = variant_boundary_data(prep, variant)
    bc_psi = psi_boundary_data(bc5, tau)
    t0 = lap("boundary_data", t0)

    presc = collect(perpendicular_rhs(prep.plan, state, phi_bar, bc5, kind_map, params))
    t0 = lap("prescribed_rhs", t0)

    solver = prep.solver
    boundary_term = np.asarray(phi_boundary_term(solver, bc_psi), dtype=np.float64)
    a_psi = np.asarray(solver.op.matrix @ psi_bar, dtype=np.float64)
    psi_n = a_psi + boundary_term                                            # discrete P07 operator on psi_bar (+ B g)
    t0 = lap("operator_apply", t0)

    # gate (a): discrete consistency -- solve A x = (A psi_bar + B g) - B g, must return psi_bar
    x_c, info_c = solve_phi(solver, psi_n, bc_psi)
    norm_psi = owner_weighted_l2(psi_bar, volume)
    consistency = float(owner_weighted_l2(x_c - psi_bar, volume) / norm_psi) if norm_psi else float("inf")
    t0 = lap("consistency_solve", t0)
    del x_c

    rhs_source = "reference_O" if omega_rhs is not None else "discrete_operator"
    rhs = np.asarray(omega_rhs, dtype=np.float64) if omega_rhs is not None else psi_n
    if rhs.shape != psi_bar.shape:
        raise ValueError(f"the psi right-hand side has shape {rhs.shape}, expected {psi_bar.shape}")
    # gate (b): N - O of psi (report only; also the sign check of O)
    norm_rhs = owner_weighted_l2(rhs, volume)
    n_minus_o = {"l2": float(owner_weighted_l2(psi_n - rhs, volume)),
                 "relative": float(owner_weighted_l2(psi_n - rhs, volume) / norm_rhs) if norm_rhs else None,
                 "reference_l2": float(norm_rhs), "rhs_source": rhs_source}

    psi_h, info_s = solve_phi(solver, rhs, bc_psi)                           # cold start
    phi_h = psi_h - tau * ti_bar
    t0 = lap("solved_solve", t0)
    solved = collect(perpendicular_rhs(prep.plan, state, phi_h, bc5, kind_map, params))
    t0 = lap("solved_rhs", t0)

    arrays = {"phi_bar": phi_bar, "ti_bar": ti_bar, "psi_bar": psi_bar, "psi_h": np.asarray(psi_h, dtype=np.float64),
              "phi_h": np.asarray(phi_h, dtype=np.float64), "psi_N": psi_n}
    for field in FIELDS:
        for term in TERMS:
            arrays[arm_key("presc", field, term)] = presc[(field, term)]
            arrays[arm_key("solved", field, term)] = solved[(field, term)]
    finite = bool(all(np.all(np.isfinite(a)) for a in arrays.values()))
    converged = bool(info_c["converged"] and info_s["converged"])
    gates = {"consistency": bool(consistency <= float(cfg["consistency_tolerance"])), "converged": converged,
             "finite": finite}
    info = {"variant": variant, "field_set": spec["field_set"], "constant": bool(spec["constant"]),
            "columns": [int(c) for c in columns], "kinds": list(kinds), "rhs_source": rhs_source,
            "consistency": {"relative_error_M": consistency, "tolerance": float(cfg["consistency_tolerance"]),
                            "pass": gates["consistency"], "solve": _solve_info(info_c)},
            "n_minus_o_psi": n_minus_o, "solved_solve": _solve_info(info_s), "gates": gates,
            "gates_pass": bool(all(gates.values())), "seconds": {**seconds, "total": time.perf_counter() - started}}
    log(f"N{prep.n} {variant}: consistency {consistency:.2e} ({info_c['iterations']} it), solved solve "
        f"{info_s['iterations']} it {info_s['seconds']:.1f} s, N-O(psi) "
        f"{n_minus_o['relative'] if n_minus_o['relative'] is None else format(n_minus_o['relative'], '.2e')}, "
        f"gates {'PASS' if info['gates_pass'] else 'FAIL'}")
    return arrays, info


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def jax_stage(*, n: int, cfg: dict, options: dict, step4, input_root, output, identity: str, inputs: dict,
              paths: dict, log=_log) -> dict:
    """Run (or resume, or skip) the JAX stage of grid ``n``: one checkpointed ``runner`` unit per variant."""
    output = Path(output)
    started = time.time()
    refs = references.load_references(output, n, identity)
    owner_volume, _regions = references.load_context_file(output, n)
    jid = jax_identity(identity, n, artifact_sha(inputs, n))
    work = references.work_dir(output)
    variants = list(cfg["variants"])
    todo = [v for v in variants if not runner.valid_unit(work, variant_unit(n, v), jid)]
    resumed = [v for v in variants if v not in todo]
    seconds = {}
    prep = None
    if todo:
        prep = prepare(n=n, cfg=cfg, options=options, step4=step4, input_root=input_root, paths=paths, inputs=inputs,
                       log=log)
        if len(prep.owner_volume) != len(owner_volume) or not np.array_equal(prep.owner_volume, owner_volume):
            raise ValueError("the owner volumes of the references stage differ from the JAX stage's environment")
        for variant in todo:
            t0 = time.time()
            fs = cfg["variants"][variant]["field_set"]
            omega_rhs = refs[references.psi_key(fs, "O")]
            arrays, info = run_variant(prep, variant, cfg, omega_rhs=omega_rhs, log=log)
            runner.write_unit(work, variant_unit(n, variant), jid, chunks={"chunk": arrays}, started=t0,
                              extra={"info": jsonable(info)})
            seconds[variant] = time.time() - t0
            del arrays
            gc.collect()
    summary = {"schema": SCHEMA, "identity": identity, "jax_identity": jid, "n": int(n), "variants": variants,
               "computed": todo, "resumed": resumed, "variant_seconds": seconds,
               "setup_seconds": None if prep is None else prep.setup_seconds,
               "plan": None if prep is None else prep.plan_summary,
               "artifact_identity_sha256": None if prep is None else prep.artifact_identity_sha256,
               "wall_seconds": time.time() - started}
    runner.write_json(output / f"N{int(n)}" / "jax_stage.json", jsonable(summary))
    return summary
