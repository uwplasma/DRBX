"""The JAX stage of the P08 step-6 campaign: both arms of the transverse variants on the full grid, plus the solver gates.

The 5.3 machinery (:func:`p08_step5_combined.jaxstage.run_variant`: prescribed arm, solved arm with the Dirichlet psi
solve, the three solver gates, the boundary data) is reused unchanged. Only :func:`prepare` differs: it takes the
artifact from the re-freeze campaign folder (read only), the sidecar from this campaign's folder, and instead of the
frozen P06N oracle owner values and the ``P06NAdapter`` it supplies this campaign's own owner averages and the
transverse states through an adapter exposing exactly what ``run_variant`` / ``variant_boundary_data`` read:
``adapter.reconstructions[variant].columns / .field_kinds``, ``adapter.owner_values`` and ``exact_state(variant)``.
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

from p_shared import runner                                                                              # noqa: E402
from p08_step5_combined import jaxstage as jax5                                                          # noqa: E402
from p08_step5_combined import references as ref5                                                        # noqa: E402
from p08_step5_combined.checkpoints import artifact_sha, jax_identity, jsonable, variant_unit            # noqa: E402
from p08_step6_global import references as refs6                                                         # noqa: E402

SCHEMA = "drbx.p08-step6-jax.v1"
run_variant = jax5.run_variant
variant_boundary_data = jax5.variant_boundary_data


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_adapter(cfg: dict, values: np.ndarray, column_names, period: float) -> SimpleNamespace:
    """What ``run_variant`` needs of a field adapter, for the transverse variants (all five fields Dirichlet)."""
    from p_shared.campaign_fields import Reconstruction

    columns = refs6.columns_from_config(cfg, period)
    if list(column_names) != list(columns.names):
        raise ValueError(f"owner-value columns {list(column_names)} differ from the configuration {list(columns.names)}")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns.names):
        raise ValueError(f"owner values have shape {values.shape}, expected (owners, {len(columns.names)})")
    reconstructions = {name: Reconstruction(np.asarray(columns.column_index(spec["field_set"]), dtype=np.int64),
                                            tuple(spec["kinds"])) for name, spec in cfg["variants"].items()}
    states = columns.states()
    return SimpleNamespace(reconstructions=reconstructions, owner_values=values, states=states,
                           exact_state=lambda variant: states[cfg["variants"][variant]["field_set"]])


def prepare(*, n: int, cfg: dict, options: dict, artifact_root, sidecar_path, input_root, inputs: dict,
            owner_values: np.ndarray, column_names, env=None, log=_log) -> SimpleNamespace:
    """Everything the variants share (``env`` may be passed to reuse an already built environment)."""
    import jax

    from drbx.native.fci_perpendicular_phi_solver import build_phi_solver
    from p_shared.replay_support import build_environment
    from p08_step2_global import replay
    from p08_step4_global.smoke import check_artifact_options

    seconds, mark = {}, [time.perf_counter()]

    def lap(name):
        now = time.perf_counter()
        seconds[name] = now - mark[0]
        mark[0] = now
        log(f"N{n}: {name} {seconds[name]:.1f} s")

    if env is None:
        env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), **options)
    lap("environment")
    artifact = replay.load_artifact(Path(artifact_root), n)
    check_artifact_options(artifact["identity"], options)
    art_sha = runner.digest(artifact["identity"])
    recorded = (inputs.get("grids", {}).get(str(n)) or {}).get("artifact_identity_sha256")
    if recorded is None or recorded != art_sha:
        raise ValueError(f"artifact identity of N{n} differs from provenance/inputs.json")
    lap("load_artifact")
    plan, plan_summary = replay.lower_plan(env, artifact, Path(artifact_root), n, device_put=False)
    n_owners = int(len(plan.p07.owner_volume))
    if n_owners != len(env.t.vol) or len(owner_values) != n_owners:
        raise ValueError(f"owner counts differ: plan {n_owners}, environment {len(env.t.vol)}, "
                         f"owner values {len(owner_values)}")
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
    adapter = build_adapter(cfg, owner_values, column_names, float(env.t.g.eta_period))
    lap("adapter")
    return SimpleNamespace(
        n=int(n), env=env, plan=plan, solver=solver, adapter=adapter, owner_volume=np.asarray(env.t.vol),
        exact_state=adapter.exact_state, artifact_identity_sha256=art_sha, plan_summary=plan_summary,
        setup_seconds={**seconds, "solver": dict(solver.setup_seconds)}, boundary_batch=int(cfg["boundary_batch"]))


def jax_stage(*, n: int, cfg: dict, options: dict, artifact_root, sidecar_path, input_root, output, identity: str,
              inputs: dict, log=_log) -> dict:
    """Run (or resume, or skip) the JAX stage of grid ``n``: one checkpointed ``runner`` unit per variant."""
    output = Path(output)
    started = time.time()
    refs = ref5.load_references(output, n, identity)
    owner_volume, _regions = ref5.load_context_file(output, n)
    values, column_names, ov_manifest = refs6.load_owner_values(output, n, identity)
    jid = jax_identity(identity, n, artifact_sha(inputs, n))
    work = ref5.work_dir(output)
    variants = list(cfg["variants"])
    todo = [v for v in variants if not runner.valid_unit(work, variant_unit(n, v), jid)]
    resumed = [v for v in variants if v not in todo]
    seconds = {}
    prep = None
    if todo:
        prep = prepare(n=n, cfg=cfg, options=options, artifact_root=artifact_root, sidecar_path=sidecar_path,
                       input_root=input_root, inputs=inputs, owner_values=values, column_names=column_names, log=log)
        if len(prep.owner_volume) != len(owner_volume) or not np.array_equal(prep.owner_volume, owner_volume):
            raise ValueError("the owner volumes of the references stage differ from the JAX stage's environment")
        for variant in todo:
            t0 = time.time()
            fs = cfg["variants"][variant]["field_set"]
            omega_rhs = refs[ref5.psi_key(fs, "O")]
            arrays, info = run_variant(prep, variant, cfg, omega_rhs=omega_rhs, log=log)
            runner.write_unit(work, variant_unit(n, variant), jid, chunks={"chunk": arrays}, started=t0,
                              extra={"info": jsonable(info)})
            seconds[variant] = time.time() - t0
            del arrays
            gc.collect()
    summary = {"schema": SCHEMA, "identity": identity, "jax_identity": jid, "n": int(n), "variants": variants,
               "computed": todo, "resumed": resumed, "variant_seconds": seconds,
               "owner_values_sha256": ov_manifest["sha256"],
               "setup_seconds": None if prep is None else prep.setup_seconds,
               "plan": None if prep is None else prep.plan_summary,
               "artifact_identity_sha256": None if prep is None else prep.artifact_identity_sha256,
               "wall_seconds": time.time() - started}
    runner.write_json(output / f"N{int(n)}" / "jax_stage.json", jsonable(summary))
    return summary
