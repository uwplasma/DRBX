"""Run matrix, manifest, scheduling rules and run-directory bookkeeping of the P10 evolved-MMS campaign (chunk C7).

Pure Python / NumPy (no JAX import): the launcher ``run_all.py`` imports it, and the tests exercise it without compute.

* :func:`build_matrix`   the 69 frozen :class:`Unit` records (60 main + 9 variant) of the settled configuration;
* :func:`build_manifest` / :func:`write_manifest` ``ROOT/manifest.json``: the matrix, the effective configuration and its sha256,
  the git commit (read from ``.git``, like ``bundle.git_commit``), the input-manifest sha256;
* scheduling: :func:`unit_cost`, :func:`priority_key`, :func:`order_units`, :func:`can_start`, :func:`simulate_schedule`,
  memory estimates :func:`parse_memory_table` / :func:`unit_memory_gb`;
* :func:`classify_run`   state of a unit's run directory against the unit and the expected bundle identity;
* :func:`variant_comparisons` the variant comparisons of ``ROOT/analysis/variants.json`` (machine output, no verdicts).

Run layout (``reduce.py`` discovers ``ROOT/main`` as a campaign tree and ``ROOT/variants/<variant>`` likewise)::

    ROOT/main/{arm}/n{N}/{mode}/{pattern}/{source}/
    ROOT/variants/{variant}/{arm}/n{N}/{mode}/{pattern}/{source}/
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent

SCHEMA = "drbx.p10-campaign-manifest-v1"
CONFIG_PATH = HERE / "configuration.json"
#: configuration keys a ``--config-override`` file may change (test hook: a tiny matrix); everything else comes from the file
ALLOWED_OVERRIDES = ("resolutions", "arms", "gated_arms", "nsteps", "T")
DEFAULT_CHUNK = 25
FIELD_NAMES = ("n", "Te", "Ti", "Omega")
#: relative cost of one step per mode (scheduling heuristic only: diffusion < bracket + curvature < coupled with the potential solve)
MODE_COST = {"diffusion": 1.0, "hyperbolic": 2.0, "coupled": 4.0}
#: default per-unit memory estimates in GiB by N (override with ``--memory-table``; the launcher records the measured peak RSS)
DEFAULT_MEMORY_GB = {32: 1.5, 48: 3.0, 64: 4.5}
#: preflight gate: ``lambda_max * dt <= 1.3`` (RK4 stability radius 2.6, safety 0.5)
PREFLIGHT_GATE = 1.3
#: tolerance of the variant ``rtol_tight`` budget (reference value reported next to the measured ratio; no verdict is made)
RTOL_BUDGET = 0.01


# ---------------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------------
def canonical_sha(obj) -> str:
    """sha256 of the canonical JSON of ``obj`` (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(override_path=None) -> dict:
    """``configuration.json`` with the optional test override file merged in (only :data:`ALLOWED_OVERRIDES`)."""
    cfg = json.loads(CONFIG_PATH.read_text())
    if override_path is not None:
        ov = json.loads(Path(override_path).read_text())
        spec_keys = {"main", "variants", "chunk"}
        bad = sorted(set(ov) - set(ALLOWED_OVERRIDES) - spec_keys)
        if bad:
            raise ValueError(f"config override may only set {ALLOWED_OVERRIDES} (and the matrix spec {sorted(spec_keys)}), got {bad}")
        cfg.update({k: v for k, v in ov.items() if k in ALLOWED_OVERRIDES})
    return cfg


def load_spec_override(override_path=None) -> dict | None:
    """The matrix-spec part (``main`` / ``variants`` / ``chunk``) of a config override file, or ``None``."""
    if override_path is None:
        return None
    ov = json.loads(Path(override_path).read_text())
    spec = {k: ov[k] for k in ("main", "variants", "chunk") if k in ov}
    return spec or None


def read_git_commit(repo: Path = REPO) -> str:
    """The HEAD commit read from the ``.git`` directory without running git (the logic of ``bundle.git_commit``); ``"unknown"``
    when it cannot be found (for example in a ``git archive`` export)."""
    try:
        git = repo / ".git"
        if git.is_file():
            git = (repo / git.read_text().split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        common = git
        if (git / "commondir").exists():
            common = (git / (git / "commondir").read_text().strip()).resolve()
        for base in (git, common):
            if (base / ref).exists():
                return (base / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except Exception:                                                               # noqa: BLE001
        pass
    return "unknown"


# ---------------------------------------------------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------------------------------------------------
def default_spec() -> dict:
    """The settled run matrix: main experiments E1-E3 and the four reported variants (all filtered arm, E3 coupled NNN-D,
    continuum source)."""
    return {
        "chunk": DEFAULT_CHUNK,
        "main": [{"experiment": "E1", "mode": "diffusion", "patterns": ["NNN-D", "DDDD"]},
                 {"experiment": "E2", "mode": "hyperbolic", "patterns": ["NNN-D"]},
                 {"experiment": "E3", "mode": "coupled", "patterns": ["NNN-D", "DDDD"]}],
        "variants": {
            "dt_half": {"ns": [32, 64], "nsteps_factor": 2, "comparison": "dt_half"},
            "rtol_tight": {"ns": [32, 64], "opts_override": {"phi_rtol": 1e-13}, "comparison": "rtol_tight"},
            "e3d": {"ns": [32, 48, 64], "opts_override": {"curvature_jump_dissipation": True, "curvature_c_kappa": 1.0},
                    "comparison": "own"},
            "w1": {"ns": [32, 64], "params_override": {"w1": 1.0}, "comparison": "own"},
        },
        "variant_defaults": {"mode": "coupled", "pattern": "NNN-D", "source": "continuum"},
    }


def _merged_spec(spec: dict | None) -> dict:
    out = default_spec()
    for key, value in (spec or {}).items():
        out[key] = copy.deepcopy(value)
    return out


@dataclass(frozen=True)
class Unit:
    """One evolution run: a frozen record with a stable id and its run directory (relative to the campaign root)."""

    id: str
    stage: str                 # "main" | "variant"
    experiment: str            # "E1" | "E2" | "E3" | the variant name
    variant: str | None
    arm: str
    n: int
    mode: str
    pattern: str
    source: str
    params_override: dict = field(default_factory=dict)      # overrides of the MmsParams of the configuration (``w1`` ...)
    opts_override: dict = field(default_factory=dict)        # overrides of the RHS options (``phi_rtol`` ...)
    nsteps: int = 0
    dt: float = 0.0
    T: float = 0.0
    chunk: int = DEFAULT_CHUNK
    out_dir: str = ""

    @property
    def cost(self) -> float:
        """Relative cost (steps x (N / 32)^3 x per-step mode weight); a scheduling heuristic."""
        return unit_cost(self)

    @property
    def group(self) -> tuple:
        return (self.arm, self.n, self.mode, self.pattern)

    def path(self, root) -> Path:
        return Path(root) / self.out_dir

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Unit":
        return cls(**d)


def unit_id(stage: str, variant: str | None, arm: str, n: int, mode: str, pattern: str, source: str) -> str:
    head = "main" if variant is None else f"var.{variant}"
    return f"{head}.{arm}.n{n}.{mode}.{pattern}.{source}"


def _out_dir(variant: str | None, arm: str, n: int, mode: str, pattern: str, source: str) -> str:
    base = "main" if variant is None else f"variants/{variant}"
    return f"{base}/{arm}/n{n}/{mode}/{pattern}/{source}"


def _nsteps(config: dict, n: int) -> int:
    try:
        return int(config["nsteps"][str(n)])
    except KeyError:
        raise ValueError(f"configuration has no step count for N = {n} (nsteps keys {sorted(config['nsteps'])})") from None


def build_matrix(config: dict, spec: dict | None = None) -> list[Unit]:
    """The run matrix of ``config`` (the effective configuration) as a list of :class:`Unit` (main units first, then variants,
    each in a fixed order). ``dt = T / nsteps`` with ``nsteps`` from ``config["nsteps"]`` (variant ``dt_half``: twice as many)."""
    spec = _merged_spec(spec)
    T = float(config["T"])
    chunk = int(spec["chunk"])
    sources = ("continuum", "discrete")
    units: list[Unit] = []
    for arm in config["arms"]:
        for exp in spec["main"]:
            for pattern in exp["patterns"]:
                for n in config["resolutions"]:
                    for source in sources:
                        ns_ = _nsteps(config, n)
                        units.append(Unit(
                            id=unit_id("main", None, arm, n, exp["mode"], pattern, source), stage="main",
                            experiment=exp["experiment"], variant=None, arm=arm, n=int(n), mode=exp["mode"], pattern=pattern,
                            source=source, nsteps=ns_, dt=T / ns_, T=T, chunk=chunk,
                            out_dir=_out_dir(None, arm, n, exp["mode"], pattern, source)))
    defaults = spec["variant_defaults"]
    for name, v in spec["variants"].items():
        mode, pattern, source = (v.get(k, defaults[k]) for k in ("mode", "pattern", "source"))
        for arm in config["gated_arms"]:
            for n in v["ns"]:
                ns_ = _nsteps(config, n) * int(v.get("nsteps_factor", 1))
                units.append(Unit(
                    id=unit_id("variant", name, arm, n, mode, pattern, source), stage="variant", experiment=name, variant=name,
                    arm=arm, n=int(n), mode=mode, pattern=pattern, source=source,
                    params_override=copy.deepcopy(v.get("params_override", {})),
                    opts_override=copy.deepcopy(v.get("opts_override", {})), nsteps=ns_, dt=T / ns_, T=T, chunk=chunk,
                    out_dir=_out_dir(name, arm, n, mode, pattern, source)))
    ids = [u.id for u in units]
    dirs = [u.out_dir for u in units]
    if len(set(ids)) != len(ids) or len(set(dirs)) != len(dirs):
        raise ValueError("the run matrix has duplicate unit ids or run directories")
    return units


def matrix_counts(units) -> dict:
    """Counts of ``units`` by stage, experiment, arm, N and variant."""
    out: dict = {"total": len(units)}
    for key in ("stage", "experiment", "arm", "n", "variant", "mode", "source"):
        c: dict = {}
        for u in units:
            k = str(getattr(u, key))
            c[k] = c.get(k, 0) + 1
        out["by_" + key] = c
    return out


def select_units(units, only: str | None):
    """``units`` whose id matches the regular expression ``only`` (``re.search``); all when ``only`` is empty."""
    if not only:
        return list(units)
    rx = re.compile(only)
    return [u for u in units if rx.search(u.id)]


def params_kwargs(config: dict, params_override: dict | None = None) -> dict:
    """The ``fields.MmsParams`` keyword arguments of a unit (floats; ``D`` a list) from the configuration."""
    base = dict(rho_star=float(config["rho_star"]), tau=float(config["tau"]), D=[float(x) for x in config["D"]],
                a_phi=float(config["a_phi"]), time_scale=float(config["time_scale"]), w1=float(config["w1"]),
                a_omega=float(config["a_omega"]))
    base.update({k: ([float(x) for x in v] if isinstance(v, (list, tuple)) else float(v)) for k, v in (params_override or {}).items()})
    return base


def groups_of(units) -> list[tuple]:
    """The ``(arm, n, mode, pattern)`` groups of ``units`` (sorted)."""
    return sorted({u.group for u in units})


# ---------------------------------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------------------------------
def build_manifest(config: dict, units, *, spec: dict | None, commit: str, inputs_manifest_sha256: str | None, synthetic: bool,
                   config_file_sha256: str | None = None, created: str | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "created": created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "synthetic": bool(synthetic),
        "configuration": config,
        "configuration_sha256": canonical_sha(config),
        "configuration_file_sha256": config_file_sha256,
        "inputs_manifest_sha256": inputs_manifest_sha256,
        "spec": _merged_spec(spec),
        "counts": matrix_counts(units),
        "units": [u.to_json() for u in units],
    }


#: manifest entries that identify a campaign (a root is never reused for a different one)
IDENTITY_KEYS = ("schema", "synthetic", "configuration_sha256", "inputs_manifest_sha256", "units")


def manifest_difference(old: dict, new: dict) -> list[str]:
    return [k for k in IDENTITY_KEYS if old.get(k) != new.get(k)]


def load_manifest(root) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def units_from_manifest(manifest: dict) -> list[Unit]:
    return [Unit.from_json(d) for d in manifest["units"]]


# ---------------------------------------------------------------------------------------------------------------------
# scheduling (pure functions)
# ---------------------------------------------------------------------------------------------------------------------
def unit_cost(item) -> float:
    """``nsteps (N/32)^3 w_mode`` for a unit; items of other kinds (bundle builds, preflight calls) carry their own ``cost``."""
    if hasattr(item, "nsteps"):
        return float(item.nsteps) * (item.n / 32.0) ** 3 * MODE_COST.get(item.mode, 1.0)
    return float(item.cost)


def priority_key(item) -> tuple:
    """Largest first: descending N (so that the memory estimate is non-increasing along the order: no head-of-line blocking),
    then descending cost (coupled before hyperbolic before diffusion, more steps first), then the id."""
    return (-int(item.n), -unit_cost(item), item.id)


def order_units(items) -> list:
    return sorted(items, key=priority_key)


def parse_memory_table(text: str | None) -> dict:
    """``"32=1.5,48=3,64=4.5"`` (GiB) merged over :data:`DEFAULT_MEMORY_GB`."""
    table = dict(DEFAULT_MEMORY_GB)
    if text:
        for part in text.split(","):
            k, v = part.split("=")
            table[int(k)] = float(v)
    return table


def unit_memory_gb(n: int, table: dict | None = None) -> float:
    """Memory estimate (GiB) for resolution ``n``: the table value, else that of the smallest tabulated N above it (or the
    largest entry)."""
    table = DEFAULT_MEMORY_GB if table is None else table
    if n in table:
        return float(table[n])
    above = sorted(k for k in table if k >= n)
    return float(table[above[0]] if above else table[max(table)])


def can_start(in_use_gb: float, n_running: int, need_gb: float, workers: int, budget_gb: float | None) -> bool:
    """Whether a task needing ``need_gb`` may start: fewer than ``workers`` running and (with a budget) the sum of the estimates
    within it. A task alone (nothing running) always starts, even if it exceeds the budget (the launcher warns)."""
    if n_running >= workers:
        return False
    if budget_gb is None or n_running == 0:
        return True
    return in_use_gb + need_gb <= budget_gb + 1e-12


def simulate_schedule(items, workers: int, budget_gb: float | None, mem_of, dur_of=None) -> list[dict]:
    """Event simulation of the launcher's policy on ``items`` (already in priority order; head-of-line start rule of
    :func:`can_start`): ``[{"id", "start", "end", "mem_gb", "n_running", "in_use_gb"}]`` in start order (``n_running`` and
    ``in_use_gb`` after the start). Durations default to :func:`unit_cost`."""
    dur_of = dur_of or unit_cost
    pending, running, events, t = list(items), [], [], 0.0
    while pending or running:
        while pending and can_start(sum(r[2] for r in running), len(running), mem_of(pending[0]), workers, budget_gb):
            it = pending.pop(0)
            m = float(mem_of(it))
            running.append((t + float(dur_of(it)), it.id, m))
            events.append({"id": it.id, "start": t, "end": t + float(dur_of(it)), "mem_gb": m, "n_running": len(running),
                           "in_use_gb": sum(r[2] for r in running)})
        t = min(r[0] for r in running)
        running = [r for r in running if r[0] > t]
    return events


# ---------------------------------------------------------------------------------------------------------------------
# run directories
# ---------------------------------------------------------------------------------------------------------------------
def expected_header(unit: Unit, config: dict, bundle_sha256: str) -> dict:
    """The entries of ``run.json`` a run of ``unit`` must carry (``evolve.run`` header; ``params`` as ``evolve._jsonable_params``)."""
    return {"mode": unit.mode, "source": unit.source, "pattern": unit.pattern, "dt": unit.dt, "T": unit.T, "nsteps": unit.nsteps,
            "chunk": unit.chunk, "params": params_kwargs(config, unit.params_override), "bundle_sha256": bundle_sha256,
            "arm": unit.arm, "n": unit.n}


def classify_run(root, unit: Unit, config: dict, bundle_sha256: str) -> dict:
    """State of the run directory of ``unit``: ``{"status", "reason", "run_status", "step"}`` with ``status`` one of

    * ``pending``    no ``run.json`` (never started): to be run;
    * ``resumable``  ``run.json`` with status ``running`` (interrupted): to be resumed from its checkpoint through ``evolve.run``;
    * ``complete``   status ``complete`` with all outputs: nothing to do;
    * ``stopped``    status ``stopped`` (the stop rule of ``evolve``: a result, not rerun);
    * ``mismatch``   the stored run differs from the unit (identity, parameters, dt, ...): reported, never rerun or relabelled;
    * ``damaged``    unreadable ``run.json`` / missing outputs of a complete run / unknown status: reported, never rerun.
    """
    d = unit.path(root)
    rj = d / "run.json"
    if not rj.is_file():
        return {"status": "pending", "reason": "no run.json", "run_status": None, "step": 0}
    try:
        rec = json.loads(rj.read_text())
    except (OSError, ValueError) as exc:
        return {"status": "damaged", "reason": f"unreadable run.json: {exc}", "run_status": None, "step": None}
    want = expected_header(unit, config, bundle_sha256)
    ident = rec.get("identity") or {}
    have = {k: rec.get(k) for k in ("mode", "source", "pattern", "dt", "T", "nsteps", "chunk", "params")}
    have.update(bundle_sha256=ident.get("bundle_sha256"), arm=ident.get("arm"), n=ident.get("n"))
    bad = {k: {"stored": have[k], "expected": want[k]} for k in want if have[k] != want[k]}
    if rec.get("schema") != "drbx.p10-run-v1":
        bad["schema"] = {"stored": rec.get("schema"), "expected": "drbx.p10-run-v1"}
    status = rec.get("status")
    chunks = rec.get("chunks") or []
    step = chunks[-1]["step_end"] if chunks else 0
    if bad:
        return {"status": "mismatch", "reason": "stored run differs from the unit: " + json.dumps(bad), "run_status": status,
                "step": step}
    if status == "complete":
        snaps = rec.get("snapshot_steps") or []
        missing = [name for name in ["geometry.npz"] + [f"snapshots/snap_{k:02d}.npz" for k in range(len(snaps) + 1)]
                   if not (d / name).is_file()]
        if missing or step != unit.nsteps:
            return {"status": "damaged", "reason": f"complete run with missing outputs {missing} / last step {step}",
                    "run_status": status, "step": step}
        return {"status": "complete", "reason": None, "run_status": status, "step": step}
    if status == "stopped":
        return {"status": "stopped", "reason": rec.get("stop_reason"), "run_status": status, "step": step,
                "stop_step": rec.get("stop_step")}
    if status == "running":
        note = None if (d / "checkpoint.npz").is_file() else "no checkpoint: evolve restarts from step 0"
        return {"status": "resumable", "reason": note, "run_status": status, "step": step}
    return {"status": "damaged", "reason": f"unknown run status {status!r}", "run_status": status, "step": step}


# ---------------------------------------------------------------------------------------------------------------------
# variant comparisons
# ---------------------------------------------------------------------------------------------------------------------
def _final_snapshot(run_dir: Path):
    """``(q, q_exact, t, H, regions)`` of the last snapshot of a complete run, or ``(None, reason)``."""
    try:
        rec = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError) as exc:
        return None, f"no readable run.json ({type(exc).__name__})"
    if rec.get("status") != "complete":
        return None, f"run status {rec.get('status')!r}"
    snaps = sorted((int(m.group(1)), p) for p in (run_dir / "snapshots").glob("snap_*.npz")
                   if (m := re.fullmatch(r"snap_(\d+)\.npz", p.name)))
    if not snaps:
        return None, "no snapshots"
    with np.load(snaps[-1][1]) as z:
        q, qe, t = np.asarray(z["q"], dtype=np.float64), np.asarray(z["q_exact"], dtype=np.float64), float(z["t"])
    with np.load(run_dir / "geometry.npz") as g:
        H = np.asarray(g["H"], dtype=np.float64)
        regions = [("global", None)] + [(str(nm), np.asarray(g["region_masks"][i], dtype=bool))
                                        for i, nm in enumerate(g["region_names"]) if str(nm) != "global"]
    return (q, qe, t, H, regions), None


def _norms(d, H, regions) -> dict:
    """``{field: {region: ||d_field||_region}}`` of a difference ``d (E, P, 4)`` in the H norm."""
    out = {}
    for i, f in enumerate(FIELD_NAMES):
        a2 = H * d[..., i] ** 2
        out[f] = {name: float(np.sqrt(a2.sum() if m is None else a2[m].sum())) for name, m in regions}
    return out


def _ratio(a, b):
    return None if (a is None or b is None or not b > 0.0) else a / b


def _combine(**tables) -> dict:
    """``{field: {region: {name: value}}}`` from ``{name: {field: {region: value}}}`` (the first table fixes fields / regions)."""
    first = next(iter(tables.values()))
    return {f: {r: {name: (t[f][r] if t is not None else None) for name, t in tables.items()} for r in first[f]} for f in first}


def variant_comparisons(root, units, spec: dict | None = None) -> dict:
    """The variant comparisons at the final time ``T`` (H norms per field and region; machine output, no verdicts):

    * ``dt_half``   ``||q_N(dt) - q_N(dt/2)||`` next to the main run's ``O - R`` and ``N - R`` (same arm, N, mode, pattern) and
      the ratios to both;
    * ``rtol_tight`` ``||q_N(rtol) - q_N(tight rtol)||`` next to the main ``N - R`` and their ratio (``budget_reference`` 1%);
    * ``own`` (``e3d``, ``w1``, any other variant) the variant's own ``N - R`` next to the main ``N - R`` and their ratio.

    Which comparison a variant gets is ``spec["variants"][name]["comparison"]`` (default: the name when it is ``dt_half`` or
    ``rtol_tight``, else ``own``). A main or variant run that is not complete gives an entry with ``"status"`` explaining why."""
    root = Path(root)
    spec = _merged_spec(spec)
    main = {(u.arm, u.n, u.mode, u.pattern, u.source): u for u in units if u.stage == "main"}
    out: dict = {"schema": "drbx.p10-variants-v1", "norm": "H-weighted: ||a||_R = sqrt(sum_{nodes in R} H a^2), region 'global' = "
                 "the whole (E, P) domain; fields (n, Te, Ti, Omega) at the final snapshot time T",
                 "rtol_budget_reference": RTOL_BUDGET, "variants": {}}
    for name in sorted({u.variant for u in units if u.variant}):
        kind = spec["variants"].get(name, {}).get("comparison", name if name in ("dt_half", "rtol_tight") else "own")
        entries = []
        for u in sorted((u for u in units if u.variant == name), key=lambda x: (x.arm, x.n)):
            ent = {"unit": u.id, "arm": u.arm, "n": u.n, "mode": u.mode, "pattern": u.pattern, "source": u.source,
                   "comparison": kind, "status": "ok"}
            entries.append(ent)
            snap, why = _final_snapshot(u.path(root))
            if snap is None:
                ent["status"] = f"variant run unusable: {why}"
                continue
            q_v, qe_v, t_v, H, regions = snap
            mc = main.get((u.arm, u.n, u.mode, u.pattern, "continuum"))
            mo = main.get((u.arm, u.n, u.mode, u.pattern, "discrete"))
            sc = _final_snapshot(mc.path(root)) if mc is not None else (None, "no such main unit")
            so = _final_snapshot(mo.path(root)) if mo is not None else (None, "no such main unit")
            nr = _norms(sc[0][0] - sc[0][1], H, regions) if sc[0] is not None and sc[0][0].shape == q_v.shape else None
            orr = _norms(so[0][0] - so[0][1], H, regions) if so[0] is not None and so[0][0].shape == q_v.shape else None
            ent["main_N-R_status"] = "ok" if nr is not None else f"unavailable: {sc[1] or 'shape mismatch'}"
            ent["t_variant"] = t_v
            if sc[0] is not None:
                ent["t_main"] = sc[0][2]
            if kind in ("dt_half", "rtol_tight"):
                if sc[0] is None or nr is None:
                    ent["status"] = f"main continuum run unusable: {sc[1] or 'shape mismatch'}"
                    continue
                diff = _norms(sc[0][0] - q_v, H, regions)
                key = "dt_diff" if kind == "dt_half" else "rtol_diff"
                tables = {key: diff, "N-R": nr}
                if kind == "dt_half":
                    tables["O-R"] = orr
                comb = _combine(**tables)
                for f in comb:
                    for r, cell in comb[f].items():
                        cell[f"{key}/N-R"] = _ratio(cell[key], cell["N-R"])
                        if kind == "dt_half":
                            cell[f"{key}/O-R"] = _ratio(cell[key], cell.get("O-R"))
                ent["fields"] = comb
            else:
                own = _norms(q_v - qe_v, H, regions)
                comb = _combine(**{"variant_N-R": own, "main_N-R": nr})
                for f in comb:
                    for r, cell in comb[f].items():
                        cell["variant/main"] = _ratio(cell["variant_N-R"], cell["main_N-R"])
                ent["fields"] = comb
        out["variants"][name] = {"comparison": kind, "entries": entries}
    return out


def tail(path, n: int = 15) -> str:
    """The last ``n`` lines of a text file (empty string if unreadable)."""
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ceil_nsteps(T: float, lam: float, gate: float = PREFLIGHT_GATE) -> int:
    """Smallest step count with ``lam * T / nsteps <= gate``."""
    return int(math.ceil(T * lam / gate))
