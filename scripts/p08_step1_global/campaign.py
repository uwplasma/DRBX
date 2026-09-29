#!/usr/bin/env python3
"""P08 step-1 global campaign: builds the per-grid row artifact (schema
``drbx.p-row-artifact.v2``) and replays it, unit-parallel, against the six
frozen accepted-campaign oracles (P05, P05N frozen/upwind, P06N, P06 legacy,
P07, P07N) -- see ``work/p08_step1_consolidation_design_20260928/design.md``
and this package's ``README.md``.

Stages: **artifact** (geometry, then cells/faces/p07 rows -- delegates to
``p_shared.build_artifact.run_full_build`` through one small adapter,
:func:`_build_artifact`, so that module's own build entry point can change
without touching this file in more than one place) -> **replay units**
(``cells``/``faces``/``p07``, via ``p_shared.replay_units`` and
``p_shared.runner.run_stage``) -> **reduction** (``p_shared.replay_units
.reduce_grid``) -> **validation**.

Do not put this package's own directory first on ``sys.path`` (no
stdlib-shadowing name here, but ``DRBX/scripts`` must be on the path for
every ``p_shared``/other-campaign import to resolve). Run as
``python -m p08_step1_global.campaign`` from ``DRBX/scripts``.

Never edit ``p_shared/build_artifact.py``, ``p_shared/runner.py`` or
``p_shared/provider.py`` from here (another agent may still be changing
``build_artifact.py``'s internals) -- everything this module needs from
them is called through their public/documented surface
(``run_full_build``, ``runner``'s documented helpers).
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import fcntl
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent            # .../DRBX/scripts
REPO = SCRIPTS.parent            # .../DRBX
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import runner                                    # noqa: E402
from p_shared import build_artifact as build_artifact_mod        # noqa: E402
from p_shared import replay_units as ru                          # noqa: E402
from p_shared import oracle_manifest as om                       # noqa: E402
from p_shared import owner_closure as oc                          # noqa: E402
from p_shared.replay_support import CAMPAIGN_FUNCS, CAMPAIGN_CATALOGUE_FILES  # noqa: E402
from p08_step1_global import bounded_build as bb                 # noqa: E402

GRIDS = (32, 48, 64)
STAGES = ("artifact", "cells", "faces", "p07")


# ---------------------------------------------------------------------------
# Small persistence helpers (mirrors p05n_field_derived_global/campaign.py
# and p06n_field_derived_global/campaign.py's own sha/digest/write/lock).
# ---------------------------------------------------------------------------
def sha(path) -> str:
    return runner.sha256_file(path)


def digest(obj) -> str:
    return runner.digest(obj)


def write(path, obj) -> None:
    runner.write_json(path, obj)


@contextmanager
def lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def config() -> dict:
    cfg = json.loads((HERE / "configuration.json").read_text())
    if cfg["schema"] != "drbx.p08-step1-global-campaign-v1" or cfg["resolutions"] != [32, 48, 64]:
        raise ValueError("unsupported P08 step-1 global campaign contract")
    if cfg["campaigns"] != list(CAMPAIGN_FUNCS):
        raise ValueError("campaign catalogue changed relative to p_shared.replay_support.CAMPAIGN_FUNCS")
    return cfg


def _input_manifest() -> dict:
    """The remote-ready immutable input manifest -- byte-identical to
    ``p07n_field_derived_global/input_manifest.json`` (the same 17 files),
    so this package's ``--input-root`` reuses exactly the same immutable
    input root the P05N/P06N/P07N remote campaigns use. A thin, named
    wrapper (rather than an inline read in :func:`verify`) so tests can
    monkeypatch it to exercise ``verify``'s identity/lock/resume logic
    against a synthetic, file-free workspace without also needing the real
    ~9 GB of immutable inputs on disk."""
    return json.loads((HERE / "input_manifest.json").read_text())


# ---------------------------------------------------------------------------
# Identity: every source this campaign's own logic + the artifact build +
# the replay depend on, the oracle manifest's own file hashes, and
# configuration.json's fixed numerical policy (design section 3's
# "Identity" -- "policy" here is folded through configuration.json, hashed
# directly rather than restated). Also every oracle campaign's own frozen
# catalogue file (see the task report's "catalogue pinning" finding:
# ``replay_units.py`` selects each oracle's catalogue table explicitly by
# file name -- ``p05n_catalogue.json``/``p05n_upwind_catalogue.json`` --
# independent of whichever catalogue a package's own (mutable)
# ``configuration.json`` currently points at, but the catalogue *file's own
# content* was not previously pinned into this identity) and
# ``p06_structured_global/configuration.json`` (its ``time`` value is
# hardcoded verbatim in ``replay_units.py``'s P06-legacy blocks -- currently
# consistent, but not otherwise guarded against drift).
# ---------------------------------------------------------------------------
SOURCE_FILES = [
    "scripts/p08_step1_global/campaign.py",
    "scripts/p08_step1_global/bounded_build.py",
    "scripts/p08_step1_global/configuration.json",
    "scripts/p08_step1_global/input_manifest.json",
    "scripts/p08_step1_global/oracle_manifest.json",
    "scripts/p_shared/replay_units.py",
    "scripts/p_shared/replay_support.py",
    "scripts/p_shared/apply.py",
    "scripts/p_shared/provider.py",
    "scripts/p_shared/oracle_manifest.py",
    "scripts/p_shared/runner.py",
    "scripts/p_shared/build_artifact.py",
    "src/drbx/stencils/artifact.py",
    "src/drbx/stencils/builder.py",
    "src/drbx/stencils/census.py",
    "src/drbx/stencils/geometry_arrays.py",
    "src/drbx/geometry/fci_perpendicular_reconstruction.py",
    "src/drbx/geometry/fci_perpendicular_neumann_trace.py",
    "src/drbx/geometry/fci_perpendicular_integrated_rows.py",
    "src/drbx/geometry/_fci_perpendicular_point_primitives.py",
    "src/drbx/native/fci_operators.py",
    "src/drbx/native/fci_perpendicular_face_corrections.py",
    "src/drbx/native/fci_perpendicular_integrated_rows.py",
    *CAMPAIGN_CATALOGUE_FILES.values(),
    "scripts/p06_structured_global/configuration.json",
]


def source_hashes() -> dict:
    return {p: sha(REPO / p) for p in SOURCE_FILES}


def oracle_paths(oracle_root: Path | None, input_root: Path) -> dict:
    """Resolve every campaign's oracle folder, remapped under
    ``oracle_root`` (a per-campaign mapping, task deliverable 2's
    ``--oracle-root``) if given, else under ``input_root`` (the workspace
    root) at ``configuration.json``'s ``oracle_default_paths``."""
    cfg = config()
    base = Path(oracle_root) if oracle_root is not None else Path(input_root)
    return {name: base / rel for name, rel in cfg["oracle_default_paths"].items()}


def oracle_manifest_for(*, input_root: Path, grids, campaigns=None) -> dict:
    """Build a *fresh* oracle manifest against ``input_root`` (never guessing
    a workspace root from ``<repo>.parent`` -- ``input_root`` is passed
    straight through to :func:`p_shared.oracle_manifest.build_manifest` as
    its explicit ``root``). :func:`verify` never calls this any more (it
    loads the *committed* manifest instead -- see
    :func:`committed_oracle_manifest`); this remains only for
    :func:`pack_oracles` (which asserts its own fresh build still equals the
    committed manifest before packing) and for tests."""
    cfg = config()
    campaigns = list(campaigns) if campaigns is not None else list(cfg["campaigns"])
    paths = oracle_paths(None, input_root)
    return om.build_manifest(campaigns=campaigns, grids=grids, paths=paths, root=input_root)


def committed_oracle_manifest() -> dict:
    """The frozen oracle manifest committed alongside this package
    (``scripts/p08_step1_global/oracle_manifest.json``), extracted once from
    the delivered ``pack-oracles`` tarball (see this package's README
    "Oracles" and the task report). :func:`verify` loads this file and
    verifies every listed file's hash under ``ORACLE_ROOT`` -- it never
    rebuilds the manifest from local file state any more, so a checkout that
    does not sit inside the workspace (a remote run, or this repo's own
    ``git archive`` export) never hits the ``<repo>.parent``/``relative_to``
    crash the un-fixed :func:`oracle_manifest_for` path used to raise."""
    return json.loads((HERE / "oracle_manifest.json").read_text())


def localize_sidecar(input_root: Path, output: Path) -> Path:
    """Localize the canonical continuum reference sidecar (same underlying
    continuum geometry P05N/P06N/P07N localize into their own output
    folders -- see ``p06n_field_derived_global/campaign.py``'s
    ``localize_sidecar`` and ``p07n_field_derived_global/campaign.py``'s
    equivalent inline code in ``verify()``, same function semantics here)
    into ``<output>/localized_sidecar.json``, rewriting its
    ``metric_cache``/``makegrid``/``artifact`` paths to point under
    ``input_root`` (the immutable input root ``input_manifest.json``'s
    paths are relative to) rather than at whatever workspace originally
    produced it. Once localized, refuses (raises) if the same output
    folder's already-localized copy would change -- exactly like every
    other on-disk artifact this campaign's ``verify`` guards."""
    input_root = Path(input_root); output = Path(output)
    cfg = config()
    side = json.loads((input_root / cfg["canonical_sidecar_relative"]).read_text())
    side["metric_cache"]["path"] = str((input_root / "hsx_metric_d58d392545fd3917efeb83b6.npz").resolve())
    side["makegrid"]["path"] = str((input_root / "mgrid_res2p5cm_180pln.nc").resolve())
    side["artifact"]["path"] = str((input_root / "prototype_runs/geometry/hsx_fci_64x64x64").resolve())
    side["metric_query_batch_size"] = 4096
    local = output / "localized_sidecar.json"
    if local.exists():
        if json.loads(local.read_text()) != side:
            raise ValueError("localized reference sidecar changed")
    else:
        write(local, side)
    return local


def verify(*, input_root: Path, output: Path, oracle_root: Path | None) -> dict:
    """Verify the immutable inputs, this campaign's own sources, and the
    oracle files -- never rebuilding the oracle manifest from local file
    state (bug fix: the checkout is not guaranteed to sit inside the same
    workspace the oracle files live under, and a remote run's ``--input
    -root`` holds none of the oracle files at all -- see
    ``committed_oracle_manifest``'s docstring). ``ORACLE_ROOT`` (the root
    every oracle file's hash is checked under) is ``oracle_root`` if given,
    else ``input_root`` -- which suits local use, where the workspace root
    is both. Refuses (raises) if ``ORACLE_ROOT`` or the committed manifest's
    own identity changed relative to a previous run recorded at the same
    ``output``."""
    input_root = Path(input_root); output = Path(output)
    cfg = config()
    im = _input_manifest()
    for rec in im["files"]:
        path = input_root / rec["path"]
        if not path.is_file() or path.stat().st_size != rec["bytes"] or sha(path) != rec["sha256"]:
            raise ValueError(f"missing or changed immutable input: {path}")
    sources = source_hashes()
    manifest = committed_oracle_manifest()
    resolved_oracle_root = Path(oracle_root) if oracle_root is not None else input_root
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = None
    identity = digest({"configuration": cfg, "input_manifest": im, "sources": sources, "commit": commit,
                       "oracle_manifest": manifest})
    manifest_path = output / "campaign_manifest.json"
    local = localize_sidecar(input_root, output)
    oracle_errors = om.verify_manifest(manifest, resolved_oracle_root)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity:
            raise ValueError("campaign identity changed; use a new output folder")
        if saved["localized_sidecar_sha256"] != sha(local):
            raise ValueError("localized sidecar hash changed")
        if saved.get("oracle_root") != str(resolved_oracle_root.resolve()):
            raise ValueError("oracle root changed; use a new output folder")
    if oracle_errors:
        raise ValueError("oracle verification failed:\n" + "\n".join(oracle_errors[:50]) +
                         (f"\n(+{len(oracle_errors) - 50} more)" if len(oracle_errors) > 50 else ""))
    if not manifest_path.exists():
        import jax
        if jax.default_backend() != "cpu":
            raise ValueError("JAX CPU backend required")
        write(manifest_path, {"identity": identity, "configuration": cfg, "input_manifest": im,
                              "source_hashes": sources, "commit": commit,
                              "input_root": str(input_root.resolve()),
                              "oracle_root": str(resolved_oracle_root.resolve()),
                              "oracle_manifest": manifest, "oracle_verified": True,
                              "localized_sidecar_sha256": sha(local),
                              "python": sys.version, "platform": platform.platform(),
                              "jax_backend": jax.default_backend()})
    write(output / "oracle_manifest.json", manifest)
    for name in ("logs", "invocations", "executions"):
        (output / name).mkdir(exist_ok=True)
    return identity


def _manifest_campaigns_slice(manifest: dict, *, campaigns, grids) -> dict:
    """The subset of ``manifest["campaigns"]`` covering exactly
    ``campaigns``/``grids`` (as ``str(n)`` keys) -- used to compare a
    freshly built manifest against the committed one over only the slice
    actually requested (the CLI's ``pack-oracles --resolutions`` can ask for
    fewer than all three grids; ``workspace_root`` is deliberately excluded
    from the comparison, since it legitimately differs between the
    committed manifest's original build workspace and whatever local
    ``input_root`` this call was given)."""
    grid_keys = [str(n) for n in grids]
    return {c: {n: manifest["campaigns"][c][n] for n in grid_keys if n in manifest["campaigns"].get(c, {})}
           for c in campaigns if c in manifest["campaigns"]}


def pack_oracles(*, input_root: Path, output: Path, tar_path: Path, campaigns=None, grids=None) -> dict:
    """``pack-oracles``: tar exactly the oracle manifest's files (plus the
    manifest itself) for delivery to a remote run (see the coordinator note
    in this package's README, "Oracle delivery"). Always reads from the
    *local* default oracle paths (``--oracle-root`` is where a remote run
    points *after* extracting this tarball, never where packing itself
    reads from).

    Before packing, asserts that a manifest freshly built against
    ``input_root`` (never rebuilt at ``verify()`` time any more -- see
    ``committed_oracle_manifest``) still equals the committed
    ``scripts/p08_step1_global/oracle_manifest.json`` over exactly the
    requested ``campaigns``/``grids`` slice -- refuses to pack (and deliver)
    a tarball the committed manifest, and therefore ``verify()``'s later
    hash checks, would not recognize."""
    cfg = config()
    campaigns = list(campaigns) if campaigns is not None else list(cfg["campaigns"])
    grids = list(grids) if grids is not None else list(GRIDS)
    fresh = oracle_manifest_for(input_root=input_root, grids=grids, campaigns=campaigns)
    committed = committed_oracle_manifest()
    fresh_slice = _manifest_campaigns_slice(fresh, campaigns=campaigns, grids=grids)
    committed_slice = _manifest_campaigns_slice(committed, campaigns=campaigns, grids=grids)
    if fresh_slice != committed_slice:
        raise ValueError("freshly built oracle manifest no longer matches the committed "
                         "scripts/p08_step1_global/oracle_manifest.json for the requested "
                         "campaigns/grids; regenerate and review before packing")
    result = om.pack_oracle_tar(fresh, input_root, tar_path, campaigns=campaigns, grids=grids)
    write(output / "pack_oracles_receipt.json", {**result, "tar_path": str(tar_path)})
    return result


# ---------------------------------------------------------------------------
# Build adapter: the one place this campaign calls into
# ``p_shared.build_artifact``'s entry point. Update only this function if
# that module's signature changes.
# ---------------------------------------------------------------------------
def _build_artifact(*, n: int, input_root: Path, sidecar_path: Path, output: Path, workers: int,
                    memory_budget_gib: float | None, worker_memory_gib: float | None,
                    memory_reserve_gib: float, max_tasks_per_worker: int | None = None,
                    max_units: int | None = None) -> dict:
    cfg = config()
    if max_units is not None:
        # Local testing only (see the CLI's --max-units help): a bounded
        # build via bb.build_units, exactly like preflight_grid's own real
        # -artifact branch -- NOT build_artifact.run_full_build's own
        # max_units. That parameter is for a resume-across-many-calls
        # pattern: run_full_build's _assemble/_aggregate_receipts always
        # require a receipt for EVERY unit of the FULL grid's plan,
        # regardless of max_units, so a single truncated call always raises
        # FileNotFoundError there (confirmed empirically -- see the task
        # report) -- it can only ever succeed once enough resumed calls have
        # covered the whole grid, which is exactly the "no full-grid
        # computation" case this local-testing flag exists to avoid.
        # bb.build_units instead assembles only the units it was actually
        # given, so a single bounded call completes cleanly. Geometry must
        # already exist (this never builds a full grid's geometry -- the
        # same hard local constraint preflight's own light-subset path
        # documents).
        identity = build_artifact_mod.build_identity(n=n, input_root=input_root, sidecar_path=sidecar_path)
        grid_dir = Path(output) / f"N{n}"
        if not ((grid_dir / "geometry.npz").is_file() and (grid_dir / "census.npz").is_file()):
            raise ValueError(f"--max-units local-testing run requires pre-existing geometry.npz/census.npz "
                             f"under {grid_dir} (it never builds a full grid's geometry itself)")
        all_units = bb.stage_units(n=n, output=output, cell_chunk_size=cfg["cell_chunk_size"],
                                   face_chunk_size=cfg["face_chunk_size"], p07_chunk_size=cfg["p07_chunk_size"])
        bounded = {stage: units[:max_units] for stage, units in all_units.items()}
        return bb.build_units(n=n, input_root=input_root, sidecar_path=sidecar_path, output=output,
                              identity=identity, workers=workers, units_by_stage=bounded,
                              max_tasks_per_worker=max_tasks_per_worker)
    return build_artifact_mod.run_full_build(
        n=n, input_root=input_root, sidecar_path=sidecar_path, output=output, workers=workers,
        memory_budget_gib=memory_budget_gib, worker_memory_gib=worker_memory_gib,
        memory_reserve_gib=memory_reserve_gib,
        cell_chunk_size=cfg["cell_chunk_size"], face_chunk_size=cfg["face_chunk_size"],
        p07_chunk_size=cfg["p07_chunk_size"], geometry_raw_chunk_size=cfg["geometry_raw_chunk_size"],
        geometry_face_chunk_size=cfg["geometry_face_chunk_size"], max_tasks_per_worker=max_tasks_per_worker)


# ---------------------------------------------------------------------------
# Replay unit stages + reduction.
# ---------------------------------------------------------------------------
def _effective_workers(workers, memory_budget_gib, worker_memory_gib, memory_reserve_gib) -> int:
    if workers is None or workers < 1:
        raise ValueError("--workers must be positive")
    if memory_budget_gib is None or worker_memory_gib is None:
        return workers
    count = min(workers, int((memory_budget_gib - memory_reserve_gib) // worker_memory_gib))
    if count < 1:
        raise ValueError("memory budget cannot fit one worker plus reserve")
    return count


def run_replay_stage(*, artifact_root: Path, replay_output: Path, n: int, stage: str, input_root: Path,
                     sidecar_path: Path, paths: dict, campaigns: tuple, workers: int, identity: dict,
                     max_tasks_per_worker=None, max_units=None) -> dict:
    """One replay stage (``cells``/``faces``/``p07``) over every build unit
    ``p_shared.replay_units.artifact_plan_units`` reports for grid ``n`` --
    a thin, named wrapper around :func:`_run_stage_with_kwargs_init` for
    callers outside this module (e.g. tests)."""
    units = ru.artifact_plan_units(artifact_root, n)[stage]
    compute = {"cells": ru.compute_cells_unit, "faces": ru.compute_faces_unit, "p07": ru.compute_p07_unit}[stage]
    initargs = dict(artifact_root=str(artifact_root), n=n, input_root=str(input_root),
                    sidecar_path=str(sidecar_path), paths=paths, campaigns=campaigns,
                    output=str(replay_output), campaign_identity=identity)
    return _run_stage_with_kwargs_init(replay_output, stage, units, identity, compute, initargs,
                                       workers, max_tasks_per_worker, max_units)


def _run_stage_with_kwargs_init(output, stage, units, identity, compute, initargs: dict, workers,
                                max_tasks_per_worker, max_units) -> dict:
    """``runner.run_stage`` calls ``initializer(*initargs)`` (positional);
    ``replay_units.init_worker`` takes keyword-only arguments, so this
    passes one positional tuple containing the kwargs dict and a tiny
    module-level trampoline unpacks it in the worker."""
    return runner.run_stage(output, stage, units, identity, compute=compute, initializer=_init_worker_trampoline,
                            initargs=(initargs,), workers=workers, parts=("chunk",),
                            max_tasks_per_worker=max_tasks_per_worker, max_units=max_units)


def _init_worker_trampoline(initargs: dict) -> None:
    ru.init_worker(**initargs)


def reduce_replay(*, artifact_root: Path, replay_output: Path, n: int, input_root: Path, sidecar_path: Path,
                  paths: dict, campaigns: tuple, max_units: int | None = None) -> dict:
    """Reduce every replay unit ``p_shared.replay_units.artifact_plan_units``
    reports for grid ``n``. ``max_units``, when given (local testing only --
    see the CLI's ``--max-units`` help), restricts the plan handed to
    ``reduce_grid`` to the same per-stage prefix the bounded replay stages
    above were run over (``runner.run_stage``'s own ``max_units`` truncates
    ``todo`` to a prefix of not-yet-valid units in plan order) -- so a
    bounded ``run`` still reduces/validates cleanly over exactly what it
    built, instead of ``reduce_grid`` trying to open a chunk file for a
    planned-but-never-built unit."""
    plan = ru.artifact_plan_units(artifact_root, n)
    if max_units is not None:
        plan = {stage: units[:max_units] for stage, units in plan.items()}
    return ru.reduce_grid(output=replay_output, artifact_root=artifact_root, n=n, input_root=input_root,
                         sidecar_path=sidecar_path, paths=paths, campaigns=campaigns, plan=plan)


# ---------------------------------------------------------------------------
# Preflight (bounded; N32-style real artifact when geometry already exists,
# else the lightweight in-memory subset path -- "preflight must never
# require a full geometry stage").
# ---------------------------------------------------------------------------
def preflight_grid(*, n: int, input_root: Path, sidecar_path: Path, output: Path, paths: dict, workers: int) -> dict:
    from drbx.stencils.census import FaceCensus

    cfg = config()
    grid_dir = output / f"N{n}"
    geometry_exists = (grid_dir / "geometry.npz").is_file() and (grid_dir / "census.npz").is_file()
    started = time.time()

    if geometry_exists:
        identity = build_artifact_mod.build_identity(n=n, input_root=input_root, sidecar_path=sidecar_path)
        census = FaceCensus.load(grid_dir / "census.npz")
        face_row_indices = build_artifact_mod.face_row_selection(census)
        p07_row_indices = build_artifact_mod.builder.p07_row_selection(census)
        cell_units_all = runner.chunk_units("cells", n, n ** 3, cfg["cell_chunk_size"])
        face_units_all = runner.chunk_units("faces", n, len(face_row_indices), cfg["face_chunk_size"])
        p07_units_all = runner.chunk_units("p07", n, len(p07_row_indices), cfg["p07_chunk_size"])
        stage_units_map = {"cells": cell_units_all, "faces": face_units_all, "p07": p07_units_all}
        selected = bb.boundary_and_family_units(n=n, output=output, stage_units_map=stage_units_map,
                                                tail_count=cfg["preflight_face_tail_units"])
        selected["cells"] = cell_units_all[-cfg["preflight_cells_tail_units"]:]
        receipt = bb.build_units(n=n, input_root=input_root, sidecar_path=sidecar_path, output=output,
                                 identity=identity, workers=workers, units_by_stage=selected)
        # `bb.build_units` writes `<output>/N{n}/plan.json` from `selected`
        # (`runner.chunk_units`-shaped `{stage, n, start, stop}` dicts, no
        # `chunk_index`); a replay unit's own `compute_cells_unit`/
        # `compute_faces_unit`/`compute_p07_unit` requires `chunk_index` (to
        # find its own build chunk file in the manifest -- see
        # `load_point_chunk_for_unit`). Re-read the plan through
        # `ru.artifact_plan_units` (which tags each unit with its position in
        # the just-written plan, exactly `bb.build_units`'s own insertion
        # order) instead of reusing `selected` directly -- feeding
        # chunk_index-less units to a replay unit raised `KeyError
        # ('chunk_index')` here (see the task report), never exercised
        # end-to-end before this fix.
        plan = ru.artifact_plan_units(output, n)
        replay_output = output / "preflight_replay" / f"N{n}"
        for stage, units in plan.items():
            initargs = dict(artifact_root=str(output), n=n, input_root=str(input_root),
                            sidecar_path=str(sidecar_path), paths=paths, campaigns=tuple(cfg["campaigns"]),
                            output=str(replay_output), campaign_identity=identity)
            compute = {"cells": ru.compute_cells_unit, "faces": ru.compute_faces_unit,
                      "p07": ru.compute_p07_unit}[stage]
            _run_stage_with_kwargs_init(replay_output, stage, units, identity, compute, initargs, workers,
                                        None, None)
        replay_result = ru.reduce_grid(output=replay_output, artifact_root=output, n=n, input_root=input_root,
                                       sidecar_path=sidecar_path, paths=paths, campaigns=tuple(cfg["campaigns"]),
                                       plan=plan)
        neumann_check = ru.neumann_rebuild_compare_check(artifact_root=output, n=n, input_root=input_root,
                                                         sidecar_path=sidecar_path, stage="faces",
                                                         chunk_index=plan["faces"][0]["chunk_index"],
                                                         sample=cfg["neumann_rebuild_compare_sample"])
        campaigns_tuple = tuple(cfg["campaigns"])
        owner_closure_result = oc.run_owner_closure_check(
            n=n, input_root=input_root, sidecar_path=sidecar_path, paths=paths, campaigns=campaigns_tuple,
            compare=oc.oracle_available(paths, campaigns_tuple, n=n))
        write(grid_dir / "owner_closure_selection.json", owner_closure_result["selection"])
        all_pass = (neumann_check["pass"]
                   and all(r.get("status") == "ok" for r in replay_result["campaigns"].values())
                   and owner_closure_result.get("all_pass", True))
        return {"n": n, "mode": "real_bounded_build", "build_receipt_wall_seconds": receipt.get("wall_seconds"),
               "replay": {k: {"status": v.get("status")} for k, v in replay_result["campaigns"].items()},
               "neumann_rebuild_compare": neumann_check,
               "owner_closure": {k: v for k, v in owner_closure_result.items() if k != "table"},
               "owner_closure_failures": [r for r in owner_closure_result.get("table", []) if not r["pass"]],
               "all_pass": bool(all_pass), "seconds": time.time() - started}

    # No existing geometry: the lightweight, subset-only path (never a
    # full-grid geometry stage -- hard local constraint).
    t = _light_geometry_context(n, input_root)
    census = _build_census_light(n, t)
    face_row_indices_full = build_artifact_mod.face_row_selection(census)
    p07_row_indices_full = build_artifact_mod.builder.p07_row_selection(census)
    raw_count = min(cfg["preflight_light_raw_count"], n ** 3)
    face_count = min(cfg["preflight_light_face_count"], len(face_row_indices_full))
    p07_count = min(cfg["preflight_light_p07_count"], len(p07_row_indices_full))
    # Bias the sample toward the wall/boundary end (radial_block ordering
    # puts i>=n-2 there) plus a handful from the front (interior) --
    # cheap, bounded, no chunked artifact at all.
    raw_ids = np.concatenate([np.arange(raw_count // 2), np.arange(n ** 3 - raw_count // 2, n ** 3)])
    face_sel = np.concatenate([np.arange(face_count // 2),
                               np.arange(len(face_row_indices_full) - face_count // 2, len(face_row_indices_full))])
    face_row_indices = face_row_indices_full[face_sel]
    p07_mask = np.isin(p07_row_indices_full, face_row_indices)
    p07_row_indices = p07_row_indices_full[p07_mask][:p07_count]
    if len(p07_row_indices) == 0:
        p07_row_indices = face_row_indices[:1]  # keep the P07 stage non-empty

    point_requests, neumann_requests, integrated_requests, geometry, t, context, ref = bb.light_subset_requests(
        n=n, input_root=input_root, sidecar_path=sidecar_path, census=census, raw_ids=raw_ids,
        face_row_indices=face_row_indices, p07_row_indices=p07_row_indices)

    families = {}
    for req in integrated_requests:
        families[str(int(req.row.family))] = families.get(str(int(req.row.family)), 0) + 1
    max_point_residual = max((float(r.row.diagnostics.get("max_residual", 0.0) or 0.0) for r in point_requests),
                             default=0.0)
    max_condition = max((float(r.row.condition) for r in neumann_requests), default=0.0)
    max_neumann_residual = max((float(r.row.constraint_residual) for r in neumann_requests), default=0.0)
    condition_ok = max_condition <= cfg["neumann_max_condition"]
    residual_ok = max_point_residual <= cfg["residual_tolerance"] * 1e3  # loose local sanity band

    # Bounded rebuild-and-compare of a few Neumann rows: rebuild each
    # sampled row's own request independently (a fresh ``patch_cache``, no
    # reuse of whatever memo the original build populated) and compare
    # bitwise. This path never has an on-disk artifact to cross-check
    # against (no full geometry stage was built), so "rebuild and compare"
    # here verifies the field-independent Neumann builder is
    # deterministic/cache-independent on its own freshly-built requests,
    # rather than cross-checking an artifact's stored chunk (that check is
    # :func:`p_shared.replay_units.neumann_rebuild_compare_check`, used by
    # the real-artifact ``preflight_grid`` branch above).
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
    from p07n_field_derived_global.fields import normal as _p07n_normal

    def normal_coefficients(q):
        return _p07n_normal(ref, np.asarray(q, dtype=np.float64))

    # Root-cause fix (see the task report): a Neumann row's own true query
    # point is NOT `row.boundary_points[0]` (the first of its 28 wall
    # -lattice SVD-reconstruction nodes -- always u==1, never the row's own
    # anchor) -- this is exactly the "wrong query point" bug class
    # `p_shared.replay_units`'s own module docstring documents finding and
    # fixing elsewhere (`PointRows.trace_target_points` there). The
    # true point is derived the same way `replay_units._true_query_point`
    # derives it for a real artifact's stored rows, from the request's own
    # `source`/`entity_id`/`quad_node` tag -- reused here (not re-derived)
    # via a minimal ``t``/``census`` shim, since this branch's `t`/`census`
    # are plain local variables, not a `p_shared.replay_support.Environment`.
    from types import SimpleNamespace
    _env_shim = SimpleNamespace(t=t, census=census)
    _stage_for_source = {"R1": "cells", "R2": "faces", "R3": "faces", "R4": "p07"}

    sample = neumann_requests[:cfg["neumann_rebuild_compare_sample"]]
    max_value_diff = 0.0
    max_gradient_diff = 0.0
    for req in sample:
        point = ru._true_query_point(_env_shim, _stage_for_source[req.source], req.source, req.entity_id,
                                     req.quad_node)
        again = prepare_neumann_point_rows(context, point, normal_coefficients=normal_coefficients,
                                           radial_degree=req.radial_degree, patch_cache={})[0]
        max_value_diff = max(max_value_diff, float(np.max(np.abs(again.value - req.row.value))))
        max_gradient_diff = max(max_gradient_diff, float(np.max(np.abs(again.gradient - req.row.gradient))))
    neumann_rebuild_ok = max_value_diff == 0.0 and max_gradient_diff == 0.0

    # Owner-closure selection + oracle comparison (task report): both the
    # selection and the oracle comparison are grid-generic (every frozen
    # campaign saved its oracle arrays at N32, N48 *and* N64 -- see
    # p_shared.owner_closure's module docstring), and the comparison never
    # depends on this grid's own production geometry.npz existing --
    # ``run_owner_closure_check``'s own ``build_owner_rows`` computes
    # geometry only for the selected owners' raw cells/incident faces
    # directly from the provider (batched at the production geometry
    # stage's own 4096-row chunk size -- see
    # ``p_shared.owner_closure._owner_geometry_arrays``), independent of the
    # separate light (structural-only) subset built above. So this runs the
    # real oracle comparison here too, not just the selection.
    campaigns_tuple = tuple(cfg["campaigns"])
    owner_closure_result = oc.run_owner_closure_check(
        n=n, input_root=input_root, sidecar_path=sidecar_path, paths=paths, campaigns=campaigns_tuple,
        compare=oc.oracle_available(paths, campaigns_tuple, n=n))
    write(grid_dir / "owner_closure_selection.json", owner_closure_result["selection"])

    all_pass = (condition_ok and residual_ok and neumann_rebuild_ok
               and owner_closure_result.get("all_pass", True))
    return {
        "n": n, "mode": "light_subset", "raw_count": len(raw_ids), "face_count": len(face_row_indices),
        "p07_count": len(p07_row_indices), "point_row_count": len(point_requests),
        "neumann_row_count": len(neumann_requests), "integrated_row_count": len(integrated_requests),
        "p07_family_histogram": families, "max_point_row_residual": max_point_residual,
        "max_neumann_condition": max_condition, "max_neumann_constraint_residual": max_neumann_residual,
        "neumann_rebuild_compare": {"checked": len(sample), "max_value_diff": max_value_diff,
                                    "max_gradient_diff": max_gradient_diff, "pass": neumann_rebuild_ok},
        "structural_checks": {"condition_ok": condition_ok, "residual_ok": residual_ok},
        "owner_closure": {k: v for k, v in owner_closure_result.items() if k != "table"},
        "owner_closure_failures": [r for r in owner_closure_result.get("table", []) if not r["pass"]],
        "all_pass": bool(all_pass),
        "note": "no full geometry stage was built for this grid (hard local constraint); the owner-closure "
                "oracle comparison above never needs one (it builds geometry only for the selected owners' "
                "raw cells/incident faces directly), but every other structural check in this branch "
                "(condition/residual bounds, Neumann rebuild) is still restricted to the separate light "
                "subset sampled above.",
        "seconds": time.time() - started,
    }


def _light_geometry_context(n, input_root):
    from perpendicular_structured.reconstruction import load_context
    return load_context(n, str(input_root))


def _build_census_light(n, t):
    from drbx.stencils.census import FaceCensus
    return FaceCensus.build(n, t.ro)


def preflight(*, args, identity) -> dict:
    cfg = config()
    output = args.output
    cases = {}
    for n in args.resolutions:
        result = preflight_grid(n=n, input_root=args.input_root, sidecar_path=_sidecar_path(args),
                                output=output, paths=oracle_paths(args.oracle_root, args.input_root),
                                workers=_effective_workers(args.workers, args.memory_budget_gib,
                                                          args.worker_memory_gib, args.memory_reserve_gib))
        cases[str(n)] = result
    all_pass = all(c.get("all_pass") for c in cases.values())
    payload = {"identity": identity, "cases": cases, "all_pass": all_pass}
    write(output / "preflight.json", payload)
    return payload


def _sidecar_path(args) -> Path:
    """The localized sidecar :func:`verify`/:func:`localize_sidecar` wrote
    under this run's own ``--output`` (never a local ``work/`` scratch
    path) -- every caller (``preflight``, ``run``, ``validate``,
    ``run-stage``) resolves the sidecar through this one function, and
    ``main()`` always calls :func:`verify` (which localizes/checks it)
    before any of them runs."""
    return Path(args.output) / "localized_sidecar.json"


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------
def validate(*, args, identity) -> dict:
    output = args.output
    preflight_path = output / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["identity"] != identity:
        raise ValueError("a matching preflight is required before validate")
    preflight_payload = json.loads(preflight_path.read_text())
    if not preflight_payload.get("all_pass"):
        raise ValueError("preflight did not pass; refusing to validate")
    replay_summaries = {}
    for n in args.resolutions:
        path = output / "replay" / f"N{n}" / "replay.json"
        if not path.is_file():
            raise ValueError(f"no replay.json for N{n}; run replay/reduce first")
        replay_summaries[str(n)] = json.loads(path.read_text())
    payload = {"identity": identity, "operational_complete": True,
              "resolutions": list(args.resolutions),
              "campaign_status": {n: {name: c.get("status") for name, c in r["campaigns"].items()}
                                  for n, r in replay_summaries.items()}}
    write(output / "validation.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("verify-inputs", "preflight", "run", "validate", "run-stage",
                                       "pack-oracles"))
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", choices=(32, 48, 64), default=[32, 48, 64])
    p.add_argument("--stage", choices=("artifact",) + STAGES[1:])
    p.add_argument("--n", type=int, choices=(32, 48, 64))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--memory-budget-gib", type=float, default=None)
    p.add_argument("--worker-memory-gib", type=float, default=None)
    p.add_argument("--memory-reserve-gib", type=float, default=1.0)
    p.add_argument("--max-tasks-per-worker", type=int, default=None)
    p.add_argument("--max-units", type=int, default=None,
                   help="Local testing only: cap every stage (the artifact build's geometry_raw/geometry_face "
                        "and cells/faces/p07, plus the matching replay/reduction units) to at most this many "
                        "not-yet-built units per stage, per resolution. Applies to both `run-stage` and `run`. "
                        "A `run` whose --max-units truncates the geometry stage itself stops after that "
                        "resolution's build (geometry_complete: false in its receipt) rather than running row "
                        "stages against an incomplete geometry -- resume with a later, unbounded (or larger "
                        "--max-units) call over the same --output. Never used for a real, unbounded campaign "
                        "run.")
    p.add_argument("--oracle-root", type=Path, default=None)
    p.add_argument("--pack-output", type=Path, default=None, help="pack-oracles: tar destination")
    return p.parse_args()


def main():
    args = parse()
    args.output = args.output.resolve()
    args.input_root = args.input_root.resolve()
    if args.resolutions != sorted(set(args.resolutions)):
        raise ValueError("resolutions must be unique and ascending")
    cfg = config()
    with lock(args.output):
        for name in ("logs", "invocations"):
            (args.output / name).mkdir(exist_ok=True, parents=True)
        identity = verify(input_root=args.input_root, output=args.output, oracle_root=args.oracle_root)
        write(args.output / "invocations" / f"{time.time_ns()}_{args.command}.json",
             {"argv": sys.argv, "identity": identity, "time": time.time()})
        if args.command == "verify-inputs":
            print(json.dumps({"command": "verify-inputs", "status": "complete", "identity": identity}))
            return
        if args.command == "pack-oracles":
            if args.pack_output is None:
                raise ValueError("--pack-output is required for pack-oracles")
            result = pack_oracles(input_root=args.input_root, output=args.output, tar_path=args.pack_output,
                                  grids=args.resolutions)
            print(json.dumps({"command": "pack-oracles", **result}, default=str))
            return
        paths = oracle_paths(args.oracle_root, args.input_root)
        sidecar_path = _sidecar_path(args)
        if args.command == "preflight":
            payload = preflight(args=args, identity=identity)
            print(json.dumps({"command": "preflight", "all_pass": payload["all_pass"]}))
            return
        if args.command == "run-stage":
            if args.stage is None or args.n is None:
                raise ValueError("--stage and --n are required for run-stage")
            effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                           args.memory_reserve_gib)
            artifact_root = args.output / "artifact"
            if args.stage == "artifact":
                receipt = _build_artifact(n=args.n, input_root=args.input_root, sidecar_path=sidecar_path,
                                          output=artifact_root, workers=effective,
                                          memory_budget_gib=args.memory_budget_gib,
                                          worker_memory_gib=args.worker_memory_gib,
                                          memory_reserve_gib=args.memory_reserve_gib,
                                          max_tasks_per_worker=args.max_tasks_per_worker)
                print(json.dumps({"command": "run-stage", "stage": "artifact", "n": args.n,
                                  "wall_seconds": receipt.get("wall_seconds")}))
                return
            build_identity = build_artifact_mod.build_identity(n=args.n, input_root=args.input_root,
                                                                sidecar_path=sidecar_path)
            replay_output = args.output / "replay" / f"N{args.n}"
            summary = _run_stage_with_kwargs_init(
                replay_output, args.stage, ru.artifact_plan_units(artifact_root, args.n)[args.stage],
                build_identity, {"cells": ru.compute_cells_unit, "faces": ru.compute_faces_unit,
                                 "p07": ru.compute_p07_unit}[args.stage],
                dict(artifact_root=str(artifact_root), n=args.n, input_root=str(args.input_root),
                    sidecar_path=str(sidecar_path), paths=paths, campaigns=tuple(cfg["campaigns"]),
                    output=str(replay_output), campaign_identity=build_identity),
                effective, args.max_tasks_per_worker, args.max_units)
            print(json.dumps({"command": "run-stage", "stage": args.stage, "n": args.n, **summary}))
            return
        if args.command in ("run", "validate"):
            if args.command == "run":
                preflight_path = args.output / "preflight.json"
                if not preflight_path.is_file() or json.loads(preflight_path.read_text())["identity"] != identity:
                    raise ValueError("run requires a matching preflight first")
                effective = _effective_workers(args.workers, args.memory_budget_gib, args.worker_memory_gib,
                                               args.memory_reserve_gib)
                artifact_root = args.output / "artifact"
                for n in args.resolutions:
                    build_receipt = _build_artifact(n=n, input_root=args.input_root, sidecar_path=sidecar_path,
                                                    output=artifact_root, workers=effective,
                                                    memory_budget_gib=args.memory_budget_gib,
                                                    worker_memory_gib=args.worker_memory_gib,
                                                    memory_reserve_gib=args.memory_reserve_gib,
                                                    max_tasks_per_worker=args.max_tasks_per_worker,
                                                    max_units=args.max_units)
                    build_identity = build_receipt["identity"]
                    if not build_receipt.get("geometry_complete", True):
                        # `--max-units` (local testing only -- see the CLI help)
                        # truncated the geometry stage itself; row stages/replay
                        # cannot run yet (mirrors run_full_build's own
                        # documented resume note). Skip straight to the next
                        # resolution rather than failing the whole `run`.
                        continue
                    replay_output = args.output / "replay" / f"N{n}"
                    for stage in ("cells", "faces", "p07"):
                        _run_stage_with_kwargs_init(
                            replay_output, stage, ru.artifact_plan_units(artifact_root, n)[stage],
                            build_identity, {"cells": ru.compute_cells_unit, "faces": ru.compute_faces_unit,
                                            "p07": ru.compute_p07_unit}[stage],
                            dict(artifact_root=str(artifact_root), n=n, input_root=str(args.input_root),
                                sidecar_path=str(sidecar_path), paths=paths, campaigns=tuple(cfg["campaigns"]),
                                output=str(replay_output), campaign_identity=build_identity),
                            effective, args.max_tasks_per_worker, args.max_units)
                    reduce_replay(artifact_root=artifact_root, replay_output=replay_output, n=n,
                                 input_root=args.input_root, sidecar_path=sidecar_path, paths=paths,
                                 campaigns=tuple(cfg["campaigns"]), max_units=args.max_units)
            payload = validate(args=args, identity=identity)
            print(json.dumps({"command": args.command, "status": "complete",
                              "operational_complete": payload["operational_complete"]}))
            return
    raise ValueError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        a = parse()
        write(a.output / "last_exit.json", {"command": a.command, "status": "failed", "error": repr(exc),
                                            "time": time.time()})
        raise
