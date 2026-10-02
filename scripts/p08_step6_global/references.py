"""The two pooled stages in front of the JAX stage of the P08 step-6 campaign: ``owner_values`` and ``references``.

``owner_values`` (per grid, owner chunks via ``runner.chunk_units`` + a spawn process pool): the owner averages of every
physical column of the configuration (:func:`fields.owner_average_chunk`, the frozen P06N routine) are written to
``N{n}/owner_values.npz`` (``values (n_owners, n_columns)``, ``columns``) with ``N{n}/owner_values.json`` (identity,
sha256, column names). Workers load the grid context only (no environment).

``references`` (per grid): the step-5.3 reference chunk (``p08_step5_combined.references.reference_chunk``) for the
TRANSVERSE field sets, pooled over owner chunks exactly like 5.3; each worker builds one environment with the pinned
four options. It writes ``N{n}/references/references.npz`` + ``manifest.json`` and ``N{n}/context.npz`` (owner volumes,
the P06N region masks and the u-band masks) in the layout ``reduction.reduce_grid`` reads, so the 5.3 reduction runs
unchanged on this campaign's folder.

u-bands: owners are single-ring (checked), the owner's ``u`` is its ring centre ``t.pts[first member, 0]``; the band
edges come from the configuration; a band is ``lo <= u < hi`` (the last one closed at the top).
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
from p08_step5_combined import references as ref5                                 # noqa: E402
from p08_step6_global import fields as tfields                                    # noqa: E402

OWNER_VALUES_SCHEMA = "drbx.p08-step6-owner-values.v1"
REFERENCES_SCHEMA = "drbx.p08-step6-references.v1"
OWNER_VALUES_STAGE = "owner_values"
REFERENCES_STAGE = ref5.STAGE
OWNER_VALUES = "owner_values.npz"
OWNER_VALUES_MANIFEST = "owner_values.json"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Owner u and the u-bands
# ---------------------------------------------------------------------------
def band_name(lo: float, hi: float) -> str:
    return f"uband_{lo:.2f}-{hi:.2f}"


def owner_u(t) -> np.ndarray:
    """``(n_owners,)`` ring-centre ``u`` of every owner. Refuses an owner whose raw cells lie on several rings."""
    u = np.asarray(t.pts[:, 0], dtype=np.float64)[np.asarray(t.order)]
    starts = np.asarray(t.starts)[:-1]
    lo, hi = np.minimum.reduceat(u, starts), np.maximum.reduceat(u, starts)
    if np.any(lo != hi):
        raise ValueError("an owner spans several radial rings; the u-band of an owner is not defined")
    return lo.copy()


def band_masks(u: np.ndarray, edges) -> dict:
    """``{band name: bool mask over owners}`` for consecutive edges; ``[lo, hi)`` and the last band ``[lo, hi]``."""
    edges = [float(e) for e in edges]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"band edges must be increasing, got {edges}")
    masks = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        masks[band_name(lo, hi)] = (u >= lo) & ((u <= hi) if last else (u < hi))
    return masks


def load_owner_context(n: int, input_root, edges) -> tuple:
    """``(owner_volume, {region: mask})``: the P06N region masks of 5.3 plus the u-band masks. The bands must partition
    the owners."""
    from perpendicular_structured.reconstruction import load_context
    import p06n_field_derived_global.core as p06n_core

    t = load_context(int(n), str(input_root))
    regions = {k: np.asarray(v, dtype=bool) for k, v in p06n_core.regional_masks(t).items()}
    bands = band_masks(owner_u(t), edges)
    count = sum(m.astype(int) for m in bands.values())
    if not np.all(count == 1):
        raise ValueError("the u-bands do not partition the owners (check the band edges against [0, 1])")
    clash = set(regions) & set(bands)
    if clash:
        raise ValueError(f"region name clash: {sorted(clash)}")
    return np.asarray(t.vol, dtype=np.float64), {**regions, **bands}


# ---------------------------------------------------------------------------
# Folder layout / identities
# ---------------------------------------------------------------------------
def owner_values_path(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}" / OWNER_VALUES


def owner_values_manifest_path(output, n: int) -> Path:
    return Path(output) / f"N{int(n)}" / OWNER_VALUES_MANIFEST


def owner_values_identity(identity: str, n: int, cfg: dict) -> str:
    """Identity of the owner-value chunk checkpoints: campaign, grid, chunking."""
    return runner.digest({"campaign": identity, "stage": OWNER_VALUES_STAGE, "n": int(n),
                          "owner_chunk_size": int(cfg["owner_chunk_size"])})


def columns_from_config(cfg: dict, period: float) -> tfields.ColumnSet:
    return tfields.ColumnSet(cfg["fields"], {fs: rec["fields"] for fs, rec in cfg["field_sets"].items()}, period)


# ---------------------------------------------------------------------------
# owner_values stage
# ---------------------------------------------------------------------------
_OV: dict = {}


def init_owner_values_worker(settings: dict) -> None:
    """Pool initializer: CPU backend check, the grid context (no environment), the columns."""
    from perpendicular_structured.reconstruction import load_context
    runner.require_cpu_backend()
    t = load_context(int(settings["n"]), str(settings["input_root"]))
    _OV.clear()
    _OV.update(t=t, columns=columns_from_config(settings["cfg"], float(t.g.eta_period)), work=settings["work"],
               identity=settings["identity"])


def _ov_trampoline(settings: dict) -> None:
    init_owner_values_worker(settings)


def compute_owner_values_unit(unit: dict) -> dict:
    started = time.time()
    values = tfields.owner_average_chunk(_OV["t"], _OV["columns"].values, int(unit["start"]), int(unit["stop"]))
    receipt = runner.write_unit(_OV["work"], unit, _OV["identity"], chunks={"chunk": {"values": values}}, started=started)
    return {"seconds": receipt["seconds"], "peak_rss_gib": receipt["peak_rss_gib"]}


def _owner_values_valid(output, n: int, identity: str) -> dict | None:
    manifest_path, path = owner_values_manifest_path(output, n), owner_values_path(output, n)
    if not (manifest_path.is_file() and path.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != identity or manifest.get("sha256") != runner.sha256_file(path):
        return None
    return manifest


def owner_values_stage(*, n: int, cfg: dict, input_root, output, identity: str, workers: int,
                       max_tasks_per_worker=None) -> dict:
    """Run (or resume, or skip) the pooled owner-average stage of grid ``n``; returns its manifest."""
    from perpendicular_structured.reconstruction import load_context

    output = Path(output)
    previous = _owner_values_valid(output, n, identity)
    if previous is not None:
        _log(f"N{n}: owner values already merged ({previous['n_owners']} owners); skipping")
        return {**previous, "skipped": True}
    started = time.time()
    t = load_context(int(n), str(input_root))
    n_owners = int(len(t.vol))
    names = tuple(cfg["fields"])
    unit_identity = owner_values_identity(identity, n, cfg)
    work = ref5.work_dir(output)
    units = runner.chunk_units(OWNER_VALUES_STAGE, n, n_owners, int(cfg["owner_chunk_size"]))
    settings = {"n": int(n), "input_root": str(input_root), "cfg": cfg, "work": str(work), "identity": unit_identity}
    period = float(t.g.eta_period)
    _log(f"N{n}: owner values of {len(names)} columns, {n_owners} owners in {len(units)} chunks, {workers} workers")
    del t
    execution = runner.run_stage(work, OWNER_VALUES_STAGE, units, unit_identity, compute=compute_owner_values_unit,
                                 initializer=_ov_trampoline, initargs=(settings,), workers=int(workers),
                                 max_tasks_per_worker=max_tasks_per_worker)
    parts = []
    for unit in units:
        if not runner.valid_unit(work, unit, unit_identity):
            raise ValueError(f"owner-value chunk {unit} is missing or invalid")
        with np.load(runner.unit_path(work, unit), allow_pickle=False) as z:
            parts.append(z["values"].copy())
    values = np.concatenate(parts)
    if values.shape != (n_owners, len(names)):
        raise ValueError(f"merged owner values have shape {values.shape}, expected {(n_owners, len(names))}")
    path = owner_values_path(output, n)
    runner.save_npz(path, values=values, columns=np.asarray(names))
    manifest = {"schema": OWNER_VALUES_SCHEMA, "identity": identity, "n": int(n), "n_owners": n_owners,
                "columns": list(names), "eta_period": period, "sha256": runner.sha256_file(path),
                "bytes": int(path.stat().st_size), "units": len(units), "execution": execution,
                "wall_seconds": time.time() - started}
    runner.write_json(owner_values_manifest_path(output, n), manifest)
    return {**manifest, "skipped": False}


def load_owner_values(output, n: int, identity: str) -> tuple:
    """``(values (n_owners, n_columns), column names, manifest)`` (manifest identity and sha256 checked)."""
    manifest = _owner_values_valid(Path(output), n, identity)
    if manifest is None:
        raise ValueError(f"no valid owner values for N{n} under {owner_values_path(output, n)}; "
                         "run the owner_values stage first")
    with np.load(owner_values_path(output, n), allow_pickle=False) as z:
        return z["values"].copy(), [str(c) for c in z["columns"]], manifest


# ---------------------------------------------------------------------------
# references stage (5.3's chunk compute on the transverse states)
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def reference_states(env, cfg: dict) -> dict:
    """``{field_set: TransverseState}`` on ``env``'s eta period."""
    return columns_from_config(cfg, float(env.t.g.eta_period)).states()


def init_worker(settings: dict) -> None:
    """Pool initializer: CPU backend check, one environment per worker, the transverse states."""
    from p_shared.replay_support import build_environment
    runner.require_cpu_backend()
    env = build_environment(n=int(settings["n"]), input_root=Path(settings["input_root"]),
                            sidecar_path=Path(settings["sidecar_path"]), **settings["operator_options"])
    _WORKER.clear()
    _WORKER.update(env=env, states=reference_states(env, settings["cfg"]), params=settings["cfg"]["params"],
                   work=settings["work"], identity=settings["identity"])


def _init_trampoline(settings: dict) -> None:
    init_worker(settings)


def compute_unit(unit: dict) -> dict:
    started = time.time()
    owners = np.arange(unit["start"], unit["stop"], dtype=np.int64)
    arrays = ref5.reference_chunk(_WORKER["env"], owners, states=_WORKER["states"], params=_WORKER["params"])
    receipt = runner.write_unit(_WORKER["work"], unit, _WORKER["identity"], chunks={"chunk": arrays}, started=started)
    return {"seconds": receipt["seconds"], "peak_rss_gib": receipt["peak_rss_gib"]}


def references_stage(*, n: int, cfg: dict, options: dict, input_root, sidecar_path, output, identity: str,
                     workers: int, max_tasks_per_worker=None) -> dict:
    """Run (or resume, or skip) the reference stage of grid ``n``; returns its manifest."""
    output = Path(output)
    previous = ref5._manifest_valid(output, n, identity)
    if previous is not None:
        _log(f"N{n}: references already merged ({previous['n_owners']} owners); skipping")
        return {**previous, "skipped": True}
    started = time.time()
    owner_volume, regions = load_owner_context(n, input_root, cfg["u_band_edges"])
    n_owners = int(len(owner_volume))
    ref5.save_context(output, n, owner_volume, regions)
    unit_identity = ref5.reference_identity(identity, n, cfg)
    work = ref5.work_dir(output)
    units = runner.chunk_units(REFERENCES_STAGE, n, n_owners, int(cfg["owner_chunk_size"]))
    settings = {"n": int(n), "input_root": str(input_root), "sidecar_path": str(sidecar_path),
                "operator_options": dict(options), "cfg": cfg, "work": str(work), "identity": unit_identity}
    _log(f"N{n}: {n_owners} owners in {len(units)} chunks of {cfg['owner_chunk_size']}, {workers} workers")
    execution = runner.run_stage(work, REFERENCES_STAGE, units, unit_identity, compute=compute_unit,
                                 initializer=_init_trampoline, initargs=(settings,), workers=int(workers),
                                 max_tasks_per_worker=max_tasks_per_worker)
    merged = ref5.merge_chunks(work, units, unit_identity, n_owners)
    merged_path = ref5.references_dir(output, n) / ref5.MERGED
    runner.save_npz(merged_path, **{k: v for k, v in merged.items() if k != "owners"})
    manifest = {"schema": REFERENCES_SCHEMA, "identity": identity, "n": int(n), "n_owners": n_owners,
                "sha256": runner.sha256_file(merged_path), "bytes": int(merged_path.stat().st_size),
                "keys": sorted(k for k in merged if k != "owners"), "terms": list(ref5.REF_TERMS),
                "field_sets": {fs: list(rec["fields"]) for fs, rec in cfg["field_sets"].items()},
                "params": cfg["params"], "operator_options": dict(options), "units": len(units),
                "regions": sorted(regions), "execution": execution, "wall_seconds": time.time() - started}
    runner.write_json(ref5.references_dir(output, n) / ref5.MANIFEST, manifest)
    del merged
    gc.collect()
    return {**manifest, "skipped": False}


load_references = ref5.load_references
