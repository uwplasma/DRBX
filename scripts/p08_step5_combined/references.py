"""The reference stage of the P08 step-5.3 campaign: re-frozen owner references of the final operator.

Per grid, every owner of the full grid (owner chunks via ``runner.chunk_units``, a spawn process pool through
``runner.run_stage``; each worker builds its own ``build_environment(..., **options)`` once in the initializer):

* for every *field set* (the distinct catalogue fields of the pinned variants: ``main`` is shared by
  ``main_phi_dirichlet`` and ``dirichlet_rich``) ``p_shared.perpendicular_reference_rhs.reference_rhs`` for the four
  fields n, Te, Ti, omega: ``poisson_bracket`` (P05N ``raw_R``), ``curvature`` / ``curvature_material`` /
  ``curvature_remainder`` (P06N evolution-weighted ``raw_R``), ``perpendicular_diffusion`` (P07N ``O_q3`` exact face
  flux, ``D_f div(P grad f)``) and ``total``;
* ``psi__<fs>__O``: the owner average of the *positive* operator ``-div(P grad psi)``, ``psi = phi + tau Ti``, exactly
  like the P07N ``O_q3`` reference (``diffusion_reference(..., positive_operator=True)`` with the exact gradient of psi
  on the q3 face nodes of every P07 face) -- the right-hand side of the psi solve;
* ``psi__<fs>__R_mid``: the diffusion midpoint reference ``-div(P grad psi)`` at the raw midpoints, owner-projected by
  raw volume over owner volume, with the AUTODIFF divergence of ``J P`` (``curvature_reference.perpendicular_geometry
  (..., method="autodiff")``) and the exact gradient and Hessian of psi: numerator ``div . grad psi + tensor : Hess
  psi``, divided by ``|J|``.

The merged arrays go to ``N{n}/references/references.npz`` (+ ``manifest.json`` with the sha256), and
``N{n}/context.npz`` holds the owner volumes and the P06N region masks used by the reduction. Chunk checkpoints live
under ``<output>/work/_chunks/N{n}/references`` (``runner`` units; a crash resumes).

The chunk compute (:func:`reference_chunk`) is a pure function of an environment, an owner set and the P06N state
adapters, so the bounded preflight calls it on a small owner subset without a pool.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import runner                                                       # noqa: E402

SCHEMA = "drbx.p08-step5-combined-references.v1"
FIELDS = ("density", "Te", "Ti", "vorticity")
#: slots of the five-column P06N state ``n, Te, Ti, omega, phi``
TI_SLOT, PHI_SLOT = 2, 4
#: the per-field reference terms stored (production's term names; ``curvature = material + remainder``)
REF_TERMS = ("poisson_bracket", "curvature", "curvature_material", "curvature_remainder", "perpendicular_diffusion",
             "total")
STAGE = "references"
MERGED = "references.npz"
MANIFEST = "manifest.json"
CONTEXT = "context.npz"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ref_key(field_set: str, field: str, term: str) -> str:
    return f"ref__{field_set}__{field}__{term}"


def psi_key(field_set: str, name: str) -> str:
    return f"psi__{field_set}__{name}"


def _prr():
    from p_shared import perpendicular_reference_rhs
    return perpendicular_reference_rhs


# ---------------------------------------------------------------------------
# Catalogue check (the pinned variants / field sets against the frozen P06N catalogue)
# ---------------------------------------------------------------------------
def check_catalogue(cfg: dict, tables=None) -> dict:
    """Refuse a configuration that drifts from the frozen P06N catalogue: every pinned variant exists with exactly the
    pinned boundary kinds, all variants of one field set share their five fields, different field sets differ, and the
    field-set representative is one of its variants. Returns ``{field_set: field-name tuple}``."""
    if tables is None:
        import p06n_field_derived_global.core as p06n_core
        tables = p06n_core.CATALOGUE_TABLES
    names: dict = {}
    for variant, spec in cfg["variants"].items():
        if variant not in tables.variant_spec:
            raise ValueError(f"variant {variant!r} is not in the frozen P06N catalogue")
        fields = tuple(name for name, _bc in tables.variant_spec[variant])
        kinds = tuple("neumann" if b else "dirichlet" for b in tables.is_neumann[variant])
        if kinds != tuple(spec["kinds"]):
            raise ValueError(f"variant {variant!r}: catalogue kinds {kinds} differ from the pinned {tuple(spec['kinds'])}")
        fs = spec["field_set"]
        if fs not in cfg["field_sets"]:
            raise ValueError(f"variant {variant!r} names the unknown field set {fs!r}")
        if names.setdefault(fs, fields) != fields:
            raise ValueError(f"field set {fs!r}: variants disagree on the catalogue fields")
    if len(set(names.values())) != len(names):
        raise ValueError("two field sets have identical catalogue fields; merge them")
    for fs, rec in cfg["field_sets"].items():
        if fs not in names:
            raise ValueError(f"field set {fs!r} has no variant")
        if cfg["variants"].get(rec["representative"], {}).get("field_set") != fs:
            raise ValueError(f"representative {rec['representative']!r} is not a variant of field set {fs!r}")
    return names


# ---------------------------------------------------------------------------
# The owner-chunk compute
# ---------------------------------------------------------------------------
def psi_gradient_function(state, tau: float):
    """``points (Q, 3) -> (Q, 1, 3)``: the exact gradient of ``psi = phi + tau Ti`` (the ``exact_gradients`` callback of
    ``diffusion_reference``)."""
    def gradients(points):
        _v, g = state.values_gradients(points)
        g = np.asarray(g)
        return (g[PHI_SLOT] + tau * g[TI_SLOT])[:, None, :]
    return gradients


def psi_positive_operator(env, support, state, tau: float) -> np.ndarray:
    """``(n_owners,)`` owner average of the POSITIVE operator ``-div(P grad psi)`` (the P07N ``O_q3`` construction)."""
    prr = _prr()
    out = prr.diffusion_reference(env, support, psi_gradient_function(state, tau), positive_operator=True)
    return np.asarray(out, dtype=np.float64)[:, 0]


def psi_midpoint_reference(env, support, state, tau: float) -> np.ndarray:
    """``(n_owners,)`` owner projection of ``-div(P grad psi)`` at the raw midpoints with the autodiff divergence of
    ``J P`` (the ``_perpendicular_operator`` formula: ``div . grad psi + tensor : Hess psi`` over ``|J|``, negated to the
    positive convention), projected by raw volume over owner volume like the P07N ``R``."""
    from p_shared import replay_units as ru
    from p_shared.curvature_reference import perpendicular_geometry

    prr = _prr()
    points = support.points
    _v, g, hessian = state.values_gradients_hessians(points)
    g, hessian = np.asarray(g), np.asarray(hessian)
    grad_psi = g[PHI_SLOT] + tau * g[TI_SLOT]
    hess_psi = hessian[PHI_SLOT] + tau * hessian[TI_SLOT]
    tensor, divergence = perpendicular_geometry(env.ref, points, method="autodiff")
    jac = np.asarray(support.raw_geometry(env)["J"], dtype=np.float64)
    numerator = np.einsum("qj,qj->q", np.asarray(divergence), grad_psi)
    numerator = numerator + np.einsum("qij,qij->q", np.asarray(tensor), hess_psi)
    pointwise = numerator / np.maximum(np.abs(jac), 1.0e-30)                   # +div(P grad psi)
    pair = ru._sparse_scatter((-pointwise)[:, None], support.raw_volume, support.raw_owner)
    return prr._gather(pair, support.owners)[:, 0] / support.owner_volume


def reference_chunk(env, owners, *, states: dict, params: dict) -> dict:
    """The reference arrays of ``owners`` (one chunk or a bounded subset) for every field set in ``states``
    (``{field_set: P06NState}``); ``params`` is the configuration's ``params`` block."""
    prr = _prr()
    owners = np.asarray(owners, dtype=np.int64)
    support = prr.owner_support(env, owners)
    ref_params = prr.ReferenceParams(rho_star=float(params["rho_star"]), tau=float(params["tau"]),
                                     diffusion={f: float(params["diffusion"][f]) for f in FIELDS})
    tau = float(params["tau"])
    out: dict = {"owners": owners}
    for fs, state in states.items():
        ref = prr.reference_rhs(env, owners, state, ref_params, FIELDS, support=support)
        for field in FIELDS:
            for term in REF_TERMS:
                out[ref_key(fs, field, term)] = np.asarray(ref[field][term], dtype=np.float64)
        out[psi_key(fs, "O")] = psi_positive_operator(env, support, state, tau)
        out[psi_key(fs, "R_mid")] = psi_midpoint_reference(env, support, state, tau)
    for key, value in out.items():
        if value.shape != owners.shape:
            raise ValueError(f"reference array {key} has shape {value.shape}, expected {owners.shape}")
    return out


def reference_states(env, cfg: dict) -> dict:
    """``{field_set: P06NState}`` (exact fields of the field set's representative variant)."""
    prr = _prr()
    return {fs: prr.p06n_state(env, rec["representative"]) for fs, rec in cfg["field_sets"].items()}


# ---------------------------------------------------------------------------
# Worker side (spawn pool)
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def init_worker(settings: dict) -> None:
    """Pool initializer: CPU backend check, one environment per worker, the field-set states."""
    from p_shared.replay_support import build_environment
    runner.require_cpu_backend()
    env = build_environment(n=int(settings["n"]), input_root=Path(settings["input_root"]),
                            sidecar_path=Path(settings["sidecar_path"]), **settings["operator_options"])
    _WORKER.clear()
    _WORKER.update(env=env, states=reference_states(env, settings["cfg"]), params=settings["cfg"]["params"],
                   work=settings["work"], identity=settings["identity"])


def _init_trampoline(settings: dict) -> None:           # ``runner.run_stage`` calls ``initializer(*initargs)``
    init_worker(settings)


def compute_unit(unit: dict) -> dict:
    """Compute and checkpoint one owner chunk (runs in a worker)."""
    started = time.time()
    owners = np.arange(unit["start"], unit["stop"], dtype=np.int64)
    arrays = reference_chunk(_WORKER["env"], owners, states=_WORKER["states"], params=_WORKER["params"])
    receipt = runner.write_unit(_WORKER["work"], unit, _WORKER["identity"], chunks={"chunk": arrays}, started=started)
    return {"seconds": receipt["seconds"], "peak_rss_gib": receipt["peak_rss_gib"]}


# ---------------------------------------------------------------------------
# Controller side
# ---------------------------------------------------------------------------
def load_owner_context(n: int, input_root) -> tuple:
    """``(owner_volume (n_owners,), {region: bool mask})`` of grid ``n`` (the P06N region masks of the step-2 comparison)."""
    import p06n_field_derived_global.core as p06n_core
    from perpendicular_structured.reconstruction import load_context
    t = load_context(int(n), str(input_root))
    return np.asarray(t.vol, dtype=np.float64), {k: np.asarray(v, dtype=bool) for k, v in
                                                 p06n_core.regional_masks(t).items()}


def references_dir(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}" / "references"


def context_path(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}" / CONTEXT


def work_dir(output) -> Path:
    return Path(output) / "work"


def reference_identity(identity: str, n: int, cfg: dict) -> str:
    """Identity of the chunk checkpoints: the campaign identity, the grid and the chunking."""
    return runner.digest({"campaign": identity, "stage": STAGE, "n": int(n),
                          "owner_chunk_size": int(cfg["owner_chunk_size"])})


def save_context(output, n: int, owner_volume, regions: dict) -> Path:
    path = context_path(output, n)
    runner.save_npz(path, owner_volume=np.asarray(owner_volume, dtype=np.float64),
                    **{f"region__{k}": np.asarray(v, dtype=bool) for k, v in regions.items()})
    return path


def load_context_file(output, n: int) -> tuple:
    path = context_path(output, n)
    if not path.is_file():
        raise ValueError(f"no owner context for N{n}: run the references stage first ({path})")
    with np.load(path, allow_pickle=False) as z:
        regions = {name[len("region__"):]: z[name].copy() for name in z.files if name.startswith("region__")}
        return z["owner_volume"].copy(), regions


def merge_chunks(work, units: list, identity: str, n_owners: int) -> dict:
    """The chunk arrays concatenated in unit order (``owners`` must be exactly ``0..n_owners-1``)."""
    parts: dict = {}
    for unit in units:
        if not runner.valid_unit(work, unit, identity):
            raise ValueError(f"reference chunk {unit} is missing or invalid")
        with np.load(runner.unit_path(work, unit), allow_pickle=False) as z:
            for name in z.files:
                parts.setdefault(name, []).append(z[name].copy())
    merged = {name: np.concatenate(arrays) for name, arrays in parts.items()}
    if not np.array_equal(merged["owners"], np.arange(n_owners)):
        raise ValueError("merged reference chunks do not cover the owners 0..n_owners-1 exactly once")
    return merged


def _manifest_valid(output, n: int, identity: str) -> dict | None:
    folder = references_dir(output, n)
    manifest_path, merged_path = folder / MANIFEST, folder / MERGED
    if not (manifest_path.is_file() and merged_path.is_file() and context_path(output, n).is_file()):
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != identity or manifest.get("sha256") != runner.sha256_file(merged_path):
        return None
    return manifest


def references_stage(*, n: int, cfg: dict, options: dict, input_root, sidecar_path, output, identity: str,
                     workers: int, max_tasks_per_worker=None) -> dict:
    """Run (or resume, or skip) the reference stage of grid ``n``; returns its manifest."""
    output = Path(output)
    previous = _manifest_valid(output, n, identity)
    if previous is not None:
        _log(f"N{n}: references already merged ({previous['n_owners']} owners); skipping")
        return {**previous, "skipped": True}
    started = time.time()
    check_catalogue(cfg)
    owner_volume, regions = load_owner_context(n, input_root)
    n_owners = int(len(owner_volume))
    save_context(output, n, owner_volume, regions)
    unit_identity = reference_identity(identity, n, cfg)
    work = work_dir(output)
    units = runner.chunk_units(STAGE, n, n_owners, int(cfg["owner_chunk_size"]))
    settings = {"n": int(n), "input_root": str(input_root), "sidecar_path": str(sidecar_path),
                "operator_options": dict(options), "cfg": cfg, "work": str(work), "identity": unit_identity}
    _log(f"N{n}: {n_owners} owners in {len(units)} chunks of {cfg['owner_chunk_size']}, {workers} workers")
    execution = runner.run_stage(work, STAGE, units, unit_identity, compute=compute_unit,
                                 initializer=_init_trampoline, initargs=(settings,), workers=int(workers),
                                 max_tasks_per_worker=max_tasks_per_worker)
    merged = merge_chunks(work, units, unit_identity, n_owners)
    merged_path = references_dir(output, n) / MERGED
    runner.save_npz(merged_path, **{k: v for k, v in merged.items() if k != "owners"})
    manifest = {"schema": SCHEMA, "identity": identity, "n": int(n), "n_owners": n_owners,
                "sha256": runner.sha256_file(merged_path), "bytes": int(merged_path.stat().st_size),
                "keys": sorted(k for k in merged if k != "owners"), "terms": list(REF_TERMS),
                "field_sets": {fs: rec["representative"] for fs, rec in cfg["field_sets"].items()},
                "params": cfg["params"], "operator_options": dict(options), "units": len(units),
                "execution": execution, "wall_seconds": time.time() - started}
    runner.write_json(references_dir(output, n) / MANIFEST, manifest)
    del merged
    gc.collect()
    return {**manifest, "skipped": False}


def load_references(output, n: int, identity: str) -> dict:
    """The merged reference arrays of grid ``n`` (manifest identity and sha256 checked)."""
    manifest = _manifest_valid(Path(output), n, identity)
    if manifest is None:
        raise ValueError(f"no valid merged references for N{n} under {references_dir(output, n)}; "
                         "run the references stage first")
    with np.load(references_dir(output, n) / MERGED, allow_pickle=False) as z:
        return {name: z[name].copy() for name in z.files}
