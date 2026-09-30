"""JAX owner-closure harness and gates G1/G2 for the P08 step-2b operators (task E6).

Design: ``work/p08_step2b_operator_assembly_design_20260929/design.md`` sections 2.4, 3 and 8.

``jax_assemble_owner_terms(env, built, campaigns, oracle)`` returns the same ``out`` dict as
:func:`p_shared.owner_closure.assemble_owner_terms` (``{"cells": ..., "faces": ..., "p07": ...}``; sparse
``(uniq, numerator)`` pairs, per-variant lists, per-field dicts, ``p05_live_jump_p07ids`` / ``values``,
``q1_evolution_volume``, ...), computed with the JAX operators instead of the host replay:

1. the bounded owner rows of ``owner_closure.build_owner_rows`` are lowered once into a
   :class:`~drbx.stencils.operator_plan.PerpendicularPlan`
   (``lower_perpendicular_plan_from_rows``);
2. the boundary data of each campaign is evaluated **once** at the plan's ``dirichlet_points`` /
   ``neumann_points`` through the :mod:`p_shared.campaign_fields` adapters
   (``boundary_data_from_callables``), respecting each adapter's role tables and ``field_kinds``;
3. ``p05_terms`` / ``p05n_action`` (P05, P05N frozen and upwind), ``p06_action`` with ``p06n_layout`` (P06N,
   14 variants over shared reconstructions) and the legacy ``face_multiplier`` (P06-legacy, 4 fields),
   ``p07_face_flux`` + owner scatter (P07, P07N N and D) run on the plan;
4. the results are converted back to the host format: an owner numerator ``(n_owners, ...)`` becomes the pair
   ``(uniq, numerator[uniq])`` where ``uniq`` is exactly the set of owners the host's ``_sparse_scatter``
   touches for the same rows/faces (unique owners of the cells / of the face lower+upper owners >= 0 in the
   term's domain), so ``uniq`` arrays equal the host's.

**Host-only pieces** (not operators; computed with the host code exactly as ``replay_units`` does, or taken
from an already-computed host ``out`` via ``host_only_from``): the MMS reference terms P05N ``raw_R``, P06N
``raw_R_*`` and P07N ``O_q3``. Nothing else is host: the P05 antisymmetry diagnostic ``p05_antisymmetry_max``
is the JAX ``pair_actions`` diagnostic of the same centered bracket (the host value is reported next to it).

Boundary data is evaluated live on the plan's point tables (one batch per campaign). ``wall_cache=True``
routes the callbacks through ``env.wall_cache`` with the adapters' ``wall_keys`` for the ``"faces"`` /
``"p07"`` roles instead (the host's caching detail); see the E6 report for the measured effect (the two
routes differ from the host, and from each other, only at the ~1e-16 level of the metric evaluator's batch
shape).

Uniform tolerance policy (design section 8)
-------------------------------------------
Every JAX term is compared with the host term **in owner space** at the closure owners, in the form
``compare_to_oracle`` divides it into (numerators / owner volume; P06 terms / the q1 evolution volume; the
P05 per-face live-jump values raw, matched by p07 id). A *term* includes all its variants / fields / pairs;
its **scale** is ``max |host|`` over all of them (as E4 did: the constant-field control variants have no scale
of their own). With ``d = max |JAX - host|``:

* **non-cancellation terms**: ``d / scale <= NONCANCELLATION_REL_TOL = 1e-11``;
* **cancellation terms** (:data:`CANCELLATION_TERMS`: P05 live jump, P05N ``face_N`` / ``face_D``, the P06N and
  P06-legacy q3 corrections; they are jumps of near-equal states, so their scale is below the roundoff the
  operator itself amplifies from its inputs): ``d <= CANCELLATION_FLOOR_FACTOR * floor`` with
  ``CANCELLATION_FLOOR_FACTOR = 10``. No fraction-of-scale clause applies: these jumps shrink with refinement, so
  any fixed fraction of their scale eventually falls below one ulp of input noise (P05N-frozen ``face_D`` at N64:
  floor 7e-8 of scale). The *conditioning
  floor* of a term is the largest change of the JAX term (same owner-space form) under a random one-ulp
  relative perturbation (``ULP * N(0, 1)``, ``ULP = 2.2e-16``) of all owner fields and boundary data, maximum
  over the seeds ``FLOOR_SEEDS = (0, 1)`` (the idea of E5's ``_conditioning_floor``);
  ``max_rel_to_scale`` and ``floor_limited`` (the floor itself exceeds ``1e-8`` of scale) are reported for
  information only;
* host-only terms are reported (they are the host arrays, so ``d = 0``) but are not JAX terms.

The frozen-oracle gate is ``owner_closure.compare_to_oracle`` with the JAX terms, unchanged.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import jax                                                             # noqa: E402
jax.config.update("jax_enable_x64", True)

from drbx.native.fci_perpendicular_integrated_rows import scatter_integrated_face_flux  # noqa: E402
from drbx.native.fci_perpendicular_p05_operator import p05_terms, p05n_action           # noqa: E402
from drbx.native.fci_perpendicular_p06_operator import bc_columns, p06_action, p06n_layout  # noqa: E402
from drbx.native.fci_perpendicular_p07_operator import p07_face_flux                     # noqa: E402
from drbx.native.fci_perpendicular_reconstruction_state import (                         # noqa: E402
    BoundaryData, boundary_data_from_callables, normalize_kinds)
from drbx.stencils.loader import LoaderGrid                                              # noqa: E402
from drbx.stencils.operator_plan import lower_perpendicular_plan_from_rows              # noqa: E402

from p_shared import apply as pshared_apply                                              # noqa: E402
from p_shared import campaign_fields as cf                                               # noqa: E402
from p_shared import owner_closure as oc                                                 # noqa: E402
from p_shared import replay_units as ru                                                  # noqa: E402
from p_shared.replay_support import CAMPAIGN_FUNCS, Environment, build_environment      # noqa: E402
from p_shared.curvature_reference import DEFAULT_CURVATURE  # noqa: E402
from p_shared.face_quadrature import DEFAULT_FACE_QUADRATURE, FACE_QUADRATURE_CHOICES  # noqa: E402

__all__ = [
    "SCHEMA", "NONCANCELLATION_REL_TOL", "CANCELLATION_FLOOR_FACTOR", "CANCELLATION_REL_TOL", "ULP",
    "FLOOR_SEEDS", "CANCELLATION_TERMS", "JaxOwnerClosure", "jax_assemble_owner_terms",
    "run_jax_owner_closure_check", "normalized_terms", "classify_term", "evaluate_policy", "compare_terms",
    "conditioning_floors", "pairs_mismatches", "iter_pairs", "host_only_terms", "perturb_boundary",
    "batched_callable", "greedy_blocks", "blocked_p05_terms", "blocked_p05n_action", "blocked_p06n_action",
]

SCHEMA = "drbx.p08-step2b-jax-owner-closure.v1"

#: non-cancellation gate: ``max |JAX - host| / max |host|``
NONCANCELLATION_REL_TOL = 1.0e-11
#: cancellation gate: ``max |JAX - host| <= FACTOR * floor``
CANCELLATION_FLOOR_FACTOR = 10.0
#: informational only: a cancellation row whose floor exceeds this fraction of its scale is ``floor_limited``
CANCELLATION_REL_TOL = 1.0e-8
#: the one-ulp relative input perturbation used to measure a conditioning floor, and its seeds
ULP = 2.2e-16
FLOOR_SEEDS = (0, 1)

#: ``out`` keys of the cancellation terms (the root key of a term name ``section.key[.sub]``)
CANCELLATION_TERMS = frozenset({
    "p05_live_jump_owner_num", "p05_live_jump_values",
    "p05n_frozen_face_N", "p05n_frozen_face_D", "p05n_upwind_face_N", "p05n_upwind_face_D",
    "p06n_faces_correction", "p06legacy_faces_correction",
})
_HOST_ONLY_ROOTS = frozenset({"p05n_frozen_raw_R", "p05n_upwind_raw_R", "p07n_global_O_q3"})
_P06N_LABELS = ("material", "remainder", "total", "R_material", "R_remainder", "R_total")
_LEGACY_TERMS = ("material", "remainder", "total")
_EV_FLOOR = 1.0e-300


def _peak_rss_gib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss) / (1024.0 ** 3 if sys.platform == "darwin" else 1024.0 ** 2)


# ---------------------------------------------------------------------------
# Host-only pieces: exactly the host lines of ``replay_units._cells_unit_core`` / ``_p07_unit_core``.
# ---------------------------------------------------------------------------
def _cells_geometry(env: Environment, raw_ids):
    t = env.t
    return t.pts[raw_ids], t.ro[raw_ids], t.rv[raw_ids]


def host_p05n_raw_R(env: Environment, adapter, raw_ids):
    """P05N ``raw_R`` (MMS reference bracket of the analytic gradients) as the host: a ``(uniq, num)`` pair."""
    points, owner_ids, volume = _cells_geometry(env, raw_ids)
    metric = env.ref._metric(points)
    h = metric["bcov"] / metric["B"][:, None]
    jac = np.abs(metric["J"])
    exact_grad = adapter.exact_gradient(points)
    action_R = np.empty((len(raw_ids), len(adapter.pair_names)))
    for col, (a, b) in enumerate(adapter.r_pair_index):
        action_R[:, col] = pshared_apply.p05_bracket(h, jac, exact_grad[:, :, a], exact_grad[:, :, b])
    return ru._sparse_scatter(action_R, volume, owner_ids)


def host_p06n_reference(env: Environment, adapter, raw_ids) -> dict:
    """P06N ``R_material`` / ``R_remainder`` / ``R_total`` per variant as the host: ``{label: [(uniq, num)]}``."""
    import p06_structured_global.numerics as p06numerics
    from perpendicular_structured.reference_geometry import curvature_geometry
    from p07_diffusion_global.numerics import quadrature as p07_quadrature

    t, n = env.t, env.n
    points, owner_ids, _volume = _cells_geometry(env, raw_ids)
    prepared = curvature_geometry(env.ref, points)
    raw_keys = np.array(np.unravel_index(raw_ids, (n,) * 3)).T
    _q1_points, q1_weight = p07_quadrature(t.faces, raw_keys, 1, face=False)
    evolution_weight = q1_weight.reshape(-1) * np.asarray(prepared.J) / np.maximum(np.asarray(prepared.B), 1.0e-30)
    out = {f"R_{label}": [] for label in ("material", "remainder", "total")}
    for name in adapter.variant_names:
        spec = adapter.variant_spec[name]
        exact_values = np.empty((len(spec), len(raw_ids)))
        exact_gradients = np.empty((len(spec), len(raw_ids), 3))
        for j, (field, _bc) in enumerate(spec):
            ev, eg, _ = adapter.evaluate_exact(points, field)
            exact_values[j] = ev
            exact_gradients[j] = eg
        exact = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
        for label, arr in zip(("R_material", "R_remainder", "R_total"), exact):
            out[label].append(ru._sparse_scatter(arr, evolution_weight, owner_ids))
    return out


def host_p07n_o_q3(env: Environment, adapter, p07_rows):
    """P07N ``O_q3`` (exact gradients against the q3 integrand) as the host: a ``(uniq, num)`` pair."""
    from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor

    census = env.census
    family = census.family[p07_rows]
    keys = census.keys()[p07_rows]
    lower_owner, upper_owner = census.owner_lo[p07_rows], census.owner_hi[p07_rows]
    names = adapter.names
    non_collapsed = np.flatnonzero(family != 0)
    face_O = np.zeros((len(keys), len(names)))
    if non_collapsed.size:
        sel_keys = keys[non_collapsed]
        points, weight = ru.pshared_provider._quadrature(env.t.faces, sel_keys, 3, face=True)
        tensor = env.ref._perpendicular_flux_tensor(points.reshape(-1, 3)).reshape(len(non_collapsed), 9, 3, 3)
        integrand = contract_face_tensor(weight, tensor, sel_keys[:, 0])
        grads = adapter.exact_gradients(points.reshape(-1, 3)).reshape(len(non_collapsed), 9, len(names), 3)
        face_O[non_collapsed] = np.einsum("fqa,fqka->fk", integrand, grads)
    return ru._sparse_scatter_signed(face_O, lower_owner, upper_owner, -1.0, +1.0)


def host_only_terms(env: Environment, built: dict, campaigns, oracle: dict, adapters: dict) -> dict:
    """``{"cells": {...}, "p07": {...}}`` of the host-only MMS reference terms of ``campaigns``."""
    cells: dict = {}
    p07: dict = {}
    raw_ids = built["raw_ids"]
    for name in ("p05n_frozen", "p05n_upwind"):
        if name in campaigns:
            cells[f"{name}_raw_R"] = host_p05n_raw_R(env, adapters[name], raw_ids)
    if "p06n" in campaigns:
        for label, value in host_p06n_reference(env, adapters["p06n"], raw_ids).items():
            cells[f"p06n_raw_{label}"] = value
    if "p07n" in campaigns:
        p07["p07n_global_O_q3"] = host_p07n_o_q3(env, adapters["p07n"], built["p07_row_indices"])
    return {"cells": cells, "p07": p07}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def perturb_array(x, rng: np.random.Generator, scale: float = ULP) -> np.ndarray:
    """``x * (1 + scale * N(0, 1))``: a random one-ulp relative perturbation."""
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 + scale * rng.standard_normal(x.shape))


def perturb_boundary(bc: BoundaryData, rng: np.random.Generator, scale: float = ULP) -> BoundaryData:
    """The same one-ulp perturbation applied to every present array of a ``BoundaryData``."""
    return BoundaryData(*(None if part is None else perturb_array(part, rng, scale) for part in bc))


def _pair(numerator, uniq: np.ndarray):
    """Host-format sparse pair ``(uniq, numerator[uniq])`` from a dense ``(n_owners, ...)`` numerator."""
    return uniq, np.asarray(numerator, dtype=np.float64)[uniq]


def _touched(*owner_arrays) -> np.ndarray:
    """Sorted unique owners ``>= 0`` (the host's ``_sparse_scatter`` ``uniq``) of the given owner-id arrays."""
    owners = np.concatenate([np.asarray(a, dtype=np.int64).reshape(-1) for a in owner_arrays])
    return np.unique(owners[owners >= 0])


def iter_pairs(obj, path: tuple = ()) -> Iterator[tuple]:
    """Yield ``(path, (uniq, values))`` for every sparse pair in a (nested) ``out`` section."""
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], np.ndarray):
        yield path, obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from iter_pairs(item, path + (i,))
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield from iter_pairs(item, path + (key,))


def pairs_mismatches(host: dict, jax_out: dict) -> list:
    """Structural differences of two ``out`` dicts: missing/extra keys per section and pairs whose ``uniq``
    arrays differ (or whose value shapes differ). Empty when the formats agree."""
    problems = []
    for section in ("cells", "faces", "p07"):
        h, j = host.get(section, {}), jax_out.get(section, {})
        for key in sorted(set(h) - set(j)):
            problems.append(f"{section}.{key}: missing in JAX out")
        for key in sorted(set(j) - set(h)):
            problems.append(f"{section}.{key}: extra in JAX out")
        hp = dict(iter_pairs({k: h[k] for k in h if k in j}))
        jp = dict(iter_pairs({k: j[k] for k in j if k in h}))
        if set(hp) != set(jp):
            problems.append(f"{section}: pair structure differs")
            continue
        for path, (uh, vh) in hp.items():
            uj, vj = jp[path]
            if not np.array_equal(uh, uj):
                problems.append(f"{section}.{'.'.join(map(str, path))}: uniq differs")
            elif np.shape(vh) != np.shape(vj):
                problems.append(f"{section}.{'.'.join(map(str, path))}: value shape {np.shape(vj)} != {np.shape(vh)}")
    return problems


# ---------------------------------------------------------------------------
# Batching / blocking (G3, full-grid plans): bound the memory of a replay without changing its arithmetic
# ---------------------------------------------------------------------------
def batched_callable(fn, batch, progress=None):
    """``fn`` evaluated on consecutive ``batch``-point slices of its ``(Q, 3)`` argument and concatenated along
    the point axis (a callback returning a tuple, like the Dirichlet trace ``(value, gradient)``, is
    concatenated component-wise). ``batch=None`` (or a table not longer than ``batch``) is ``fn`` itself.
    Use a multiple of 4096: the frozen metric evaluator re-batches at 4096 points internally.
    ``progress(done, total)`` is called after every slice (long host callbacks: P05's trace costs ~6 ms/point)."""
    if fn is None or not batch:
        return fn
    batch = int(batch)

    def wrapped(points):
        points = np.asarray(points)
        if len(points) <= batch:
            return fn(points)
        parts = []
        for i in range(0, len(points), batch):
            parts.append(fn(points[i:i + batch]))
            if progress is not None:
                progress(min(i + batch, len(points)), len(points))
        if isinstance(parts[0], tuple):
            return tuple(np.concatenate([p[k] for p in parts], axis=0) for k in range(len(parts[0])))
        return np.concatenate(parts, axis=0)
    return wrapped


def greedy_blocks(needs, cap: int, limit: Optional[int] = None) -> list:
    """Split items ``0..len(needs)-1`` into consecutive blocks whose *union* of needed columns (``needs[i]``, a
    set of hashables) has at most ``cap`` members and which hold at most ``limit`` items; an item that alone
    needs more than ``cap`` columns gets a block of its own."""
    blocks: list = []
    current: list = []
    used: set = set()
    for i, need in enumerate(needs):
        need = set(need)
        if current and (len(used | need) > cap or (limit is not None and len(current) >= limit)):
            blocks.append(current)
            current, used = [], set()
        current.append(i)
        used |= need
    if current:
        blocks.append(current)
    return blocks


def _host_arrays(result, names) -> dict:
    return {name: np.asarray(getattr(result, name)) for name in names}


def blocked_p05_terms(plan, fields, bc, field_kinds, pairs, *, column_block: int) -> SimpleNamespace:
    """:func:`p05_terms` over blocks of pairs whose columns fit ``column_block`` (fields are contracted
    independently, so the blocks reproduce the unblocked terms). Returns host NumPy ``centered_numerator``,
    ``jump_numerator`` ``(n_owners, P)``, ``face_jump`` ``(Fc, P)`` and the maximum ``antisymmetry``."""
    fields = np.asarray(fields)
    kinds = normalize_kinds(field_kinds, fields.shape[1])
    pairs = [(int(a), int(b)) for a, b in pairs]
    parts = []
    for block in greedy_blocks(pairs, column_block):
        cols = sorted({c for i in block for c in pairs[i]})
        pos = {c: j for j, c in enumerate(cols)}
        sub = tuple((pos[pairs[i][0]], pos[pairs[i][1]]) for i in block)
        r = p05_terms(plan, fields[:, cols], bc_columns(bc, cols), tuple(kinds[c] for c in cols), sub)
        parts.append(_host_arrays(r, ("centered_numerator", "jump_numerator", "face_jump", "antisymmetry")))
    cat = lambda name: np.concatenate([p[name] for p in parts], axis=1)
    return SimpleNamespace(centered_numerator=cat("centered_numerator"), jump_numerator=cat("jump_numerator"),
                           face_jump=cat("face_jump"), antisymmetry=max(float(p["antisymmetry"]) for p in parts))


def blocked_p05n_action(plan, fields, bc, role_kinds, n_pair_index, d_pair_index, *, columns,
                        column_block: int) -> SimpleNamespace:
    """:func:`p05n_action` over blocks of pairs whose role columns (N and D pairs together) fit ``column_block``.
    ``columns`` is the role -> physical column table (``Reconstruction("role").columns``); ``fields`` / ``bc``
    stay in the physical layout. Returns host NumPy ``raw_N_numerator`` / ``raw_D_numerator`` /
    ``face_N_numerator`` / ``face_D_numerator`` ``(n_owners, P)`` and the maximum ``antisymmetry``."""
    columns = [int(c) for c in np.asarray(columns).tolist()]
    kinds = normalize_kinds(role_kinds, len(columns))
    n_pairs = [(int(a), int(b)) for a, b in n_pair_index]
    d_pairs = [(int(a), int(b)) for a, b in d_pair_index]
    needs = [set(n) | set(d) for n, d in zip(n_pairs, d_pairs)]
    names = ("raw_N_numerator", "raw_D_numerator", "face_N_numerator", "face_D_numerator", "antisymmetry")
    parts = []
    for block in greedy_blocks(needs, column_block):
        roles = sorted(set().union(*(needs[i] for i in block)))
        pos = {r: j for j, r in enumerate(roles)}
        remap = lambda pair: (pos[pair[0]], pos[pair[1]])
        r = p05n_action(plan, fields, bc, tuple(kinds[k] for k in roles), [remap(n_pairs[i]) for i in block],
                        [remap(d_pairs[i]) for i in block], columns=[columns[k] for k in roles])
        parts.append(_host_arrays(r, names))
    ns = {name: np.concatenate([p[name] for p in parts], axis=1) for name in names[:-1]}
    return SimpleNamespace(**ns, antisymmetry=max(float(p["antisymmetry"]) for p in parts))


def blocked_p06n_action(plan, fields, bc, reconstructions, *, column_block: int,
                        variant_block: Optional[int] = None) -> SimpleNamespace:
    """:func:`p06_action` over blocks of P06N variants (``reconstructions``: the adapter ``Reconstruction`` of
    each variant, in order) whose unique ``(column, kind)`` pairs fit ``column_block`` and which hold at most
    ``variant_block`` variants (the q3 eigen-solve is per variant and face node). Returns host NumPy
    ``material_numerator`` / ``remainder_numerator`` / ``correction_numerator`` ``(V, n_owners, 4)``."""
    fields = np.asarray(fields)
    needs = [set(zip(np.asarray(rec.columns).tolist(), rec.field_kinds)) for rec in reconstructions]
    names = ("material_numerator", "remainder_numerator", "correction_numerator")
    parts = []
    for block in greedy_blocks(needs, column_block, variant_block):
        columns, kinds, groups = p06n_layout([reconstructions[i] for i in block])
        act = p06_action(plan, fields[:, columns], bc_columns(bc, columns), kinds, groups)
        parts.append(_host_arrays(act, names))
    return SimpleNamespace(**{name: np.concatenate([p[name] for p in parts], axis=0) for name in names})


# ---------------------------------------------------------------------------
# The JAX closure
# ---------------------------------------------------------------------------
class JaxOwnerClosure:
    """Plan + boundary data of a bounded owner closure; :meth:`evaluate` runs the JAX operators.

    ``env`` / ``built`` / ``campaigns`` / ``oracle`` as for ``owner_closure.assemble_owner_terms``. The plan is
    lowered and the boundary data evaluated once at construction (timings in ``self.timings``);
    :meth:`evaluate` may then be called repeatedly (one-ulp perturbations for the conditioning floors).

    G3 (full-grid replay, ``scripts/p08_step2_global``) options, all off by default (the default path is
    byte-for-byte the E6 closure):

    * ``plan``: an already lowered plan (e.g. streamed from a row artifact); ``built`` may then be ``None``
      (the host-only MMS references need it and are skipped without it);
    * ``column_block`` / ``variant_block``: evaluate the operators in blocks of at most that many field columns
      (P06N: variants), see :func:`blocked_p05_terms` and friends;
    * ``boundary_batch``: evaluate the boundary-data callbacks on slices of that many points, optionally
      reporting ``boundary_progress(label, done, total)`` after every slice.
    """

    def __init__(self, env: Environment, built: Optional[dict], campaigns, oracle: dict, *,
                 wall_cache: bool = False, plan=None, column_block: Optional[int] = None,
                 variant_block: Optional[int] = None, boundary_batch: Optional[int] = None,
                 boundary_progress=None):
        self.env, self.built, self.oracle = env, built, oracle
        # one flag switches the operator geometry (``built``) and the reference (``env.ref``) together
        self.curvature = getattr(env, "curvature", "fd")
        if built is not None and built.get("curvature", self.curvature) != self.curvature:
            raise ValueError(f"owner rows were built with curvature {built['curvature']!r} but the environment "
                             f"uses {self.curvature!r}")
        self.face_quadrature = getattr(env, "face_quadrature", DEFAULT_FACE_QUADRATURE)
        if built is not None and built.get("face_quadrature", self.face_quadrature) != self.face_quadrature:
            raise ValueError(f"owner rows were built with face_quadrature {built['face_quadrature']!r} but the "
                             f"environment uses {self.face_quadrature!r}")
        self.boundary_progress = boundary_progress
        self.campaigns = tuple(campaigns)
        unknown = set(self.campaigns) - set(CAMPAIGN_FUNCS)
        if unknown:
            raise KeyError(f"unknown campaigns {sorted(unknown)}")
        self.wall_cache = bool(wall_cache)
        self.column_block = None if column_block is None else int(column_block)
        self.variant_block = None if variant_block is None else int(variant_block)
        self.boundary_batch = None if boundary_batch is None else int(boundary_batch)
        if self.column_block is not None and self.column_block < 5:
            raise ValueError("column_block must be at least 5 (a P06 state has five fields)")
        self.timings: dict = {}
        t = env.t

        started = time.perf_counter()
        if plan is not None:
            self.plan = plan
        else:
            if built is None:
                raise ValueError("either a lowered plan or the built owner rows are required")
            grid = LoaderGrid.from_arrays(n=env.n, raw_to_owner=t.ro, eta_centers=t.centers[2])
            self.plan = lower_perpendicular_plan_from_rows(
                built["row_index"], built["neumann_index"], grid=grid, census=env.census,
                geometry=built["geometry"], raw_volume=t.rv, owner_volume=t.vol, raw_ids=built["raw_ids"],
                face_rows=built["face_row_indices"], p07_rows=built["p07_row_indices"])
        self.timings["lower_plan"] = time.perf_counter() - started

        self.adapters = self._build_adapters()
        started = time.perf_counter()
        self.bc = self._boundary_data()
        self.timings["boundary_data"] = time.perf_counter() - started

        plan = self.plan
        self.uniq_cells = _touched(np.asarray(plan.cells.raw_owner))
        self.uniq_faces = _touched(plan.faces.lower_owner, plan.faces.upper_owner)
        self.jump_mask = np.asarray(plan.faces.p07_valid, dtype=bool)
        self.uniq_jump = _touched(np.asarray(plan.faces.lower_owner)[self.jump_mask],
                                  np.asarray(plan.faces.upper_owner)[self.jump_mask])
        self.uniq_p07 = _touched(plan.p07.rows.lower_owner, plan.p07.rows.upper_owner)
        self.legacy_multiplier = (cf.legacy_seam_multiplier(env.census.keys()[np.asarray(plan.faces.census_row)])
                                  if "p06_legacy" in self.campaigns else None)

    # -- setup ---------------------------------------------------------------------------------------------
    def _build_adapters(self) -> dict:
        env, oracle, campaigns = self.env, self.oracle, self.campaigns
        period = env.t.g.eta_period
        adapters: dict = {}
        if "p05" in campaigns:
            adapters["p05"] = cf.P05Adapter(env.ref, oracle["p05"]["owner_values"])
        for name in ("p05n_frozen", "p05n_upwind"):
            if name in campaigns:
                adapters[name] = cf.P05NAdapter(name, env.ref, period, oracle[name]["owner_values"])
        if "p06n" in campaigns:
            adapters["p06n"] = cf.P06NAdapter(env.ref, period, oracle["p06n"]["owner_values"])
        if "p06_legacy" in campaigns:
            adapters["p06_legacy"] = cf.P06LegacyAdapter(env.ref, oracle["p06_legacy"]["owner_values"])
        if "p07" in campaigns:
            adapters["p07"] = cf.P07Adapter(env.ref, oracle["p07"]["owner_values"])
        if "p07n" in campaigns:
            adapters["p07n"] = cf.P07NAdapter(env.ref, period, oracle["p07n"]["owner_values"])
        return adapters

    def _callbacks(self, adapter, role: str, *, normal: bool):
        """``(dirichlet, normal)`` callables of ``adapter`` (optionally through ``env.wall_cache``)."""
        dirichlet = adapter.dirichlet
        normal_fn = adapter.normal if normal else None
        if not self.wall_cache:
            label = getattr(adapter, "campaign", "?") + (f"/{adapter.field_name}" if hasattr(adapter, "field_name") else "")

            def progress(kind):
                if self.boundary_progress is None:
                    return None
                return lambda done, total: self.boundary_progress(f"{label}.{kind}", done, total)

            return (batched_callable(dirichlet, self.boundary_batch, progress("dirichlet")),
                    batched_callable(normal_fn, self.boundary_batch, progress("normal")))
        cache, keys = self.env.wall_cache, adapter.wall_keys[role]
        d = dirichlet
        if keys.trace is not None:
            def d(q, _f=dirichlet, _k=keys.trace):
                return cache.lookup_or_compute(_k, _f, q)
        n = normal_fn
        if normal_fn is not None and keys.normal is not None:
            def n(q, _f=normal_fn, _k=keys.normal):
                return cache.lookup_or_compute(_k, _f, q)
        return d, n

    def _boundary_data(self) -> dict:
        plan, bc = self.plan, {}
        for name, adapter in self.adapters.items():
            if name == "p06_legacy":
                bc[name] = {}
                for field_name, field in adapter:
                    d, _ = self._callbacks(field, "faces", normal=False)
                    bc[name][field_name] = boundary_data_from_callables(plan, d)
                continue
            role = "p07" if name in ("p07", "p07n") else "faces"
            # plain P07 is all-Dirichlet (replay fix 2): the physical-normal data is never used
            need_normal = adapter.has_normal and name != "p07"
            d, n = self._callbacks(adapter, role, normal=need_normal)
            bc[name] = boundary_data_from_callables(plan, d, n)
        return bc

    # -- the operators -------------------------------------------------------------------------------------
    def _p07_numerator(self, fields, bc, kinds) -> np.ndarray:
        """Owner numerator ``(n_owners, F)`` of the P07 face flux: lower minus, upper plus (divided by one)."""
        p07 = self.plan.p07
        payload = SimpleNamespace(lower_owner=p07.rows.lower_owner, upper_owner=p07.rows.upper_owner,
                                  owner_volume=np.ones(len(p07.owner_volume)))

        def numerator(f, b, k):
            return np.asarray(scatter_integrated_face_flux(payload, p07_face_flux(self.plan, f, b, k)))

        cap = getattr(self, "column_block", None)
        if cap is None:
            return numerator(fields, bc, kinds)
        fields = np.asarray(fields)
        kinds = normalize_kinds(kinds, fields.shape[1])
        return np.concatenate([numerator(fields[:, i:i + cap], bc_columns(bc, np.arange(i, min(i + cap, fields.shape[1]))),
                                         kinds[i:i + cap]) for i in range(0, fields.shape[1], cap)], axis=1)

    def evaluate(self, *, perturb_seed: Optional[int] = None, host_only_from: Optional[dict] = None,
                 include_host_only: bool = True) -> dict:
        """Run the JAX operators and return the host-format ``out`` dict.

        ``perturb_seed``: apply a random one-ulp relative perturbation to every owner-field array and every
        boundary-data array (conditioning floors); the host-only terms are then never included. Otherwise the
        host-only MMS references are computed with the host code, or copied from ``host_only_from`` (a host
        ``out``), unless ``include_host_only`` is false.
        """
        plan, campaigns, adapters = self.plan, self.campaigns, self.adapters
        rng = None if perturb_seed is None else np.random.default_rng(perturb_seed)
        fields_of = (lambda x: np.asarray(x, dtype=np.float64)) if rng is None else (lambda x: perturb_array(x, rng))
        bc_of = (lambda bc: bc) if rng is None else (lambda bc: perturb_boundary(bc, rng))
        uc, uf, uj, up = self.uniq_cells, self.uniq_faces, self.uniq_jump, self.uniq_p07
        cells: dict = {}
        faces: dict = {}
        p07: dict = {}
        timings: dict = {}

        def timed(label, started):
            timings[label] = time.perf_counter() - started

        if "p05" in campaigns:
            started = time.perf_counter()
            a = adapters["p05"]
            if self.column_block is None:
                r = p05_terms(plan, fields_of(a.owner_values), bc_of(self.bc["p05"]), a.field_kinds, a.pairs)
            else:
                r = blocked_p05_terms(plan, fields_of(a.owner_values), bc_of(self.bc["p05"]), a.field_kinds,
                                      a.pairs, column_block=self.column_block)
            cells["p05_centered"] = _pair(r.centered_numerator, uc)
            cells["p05_antisymmetry_max"] = float(r.antisymmetry)
            mask = self.jump_mask
            faces["p05_live_jump_p07ids"] = np.asarray(plan.faces.p07_id)[mask].astype(np.int64)
            faces["p05_live_jump_values"] = np.asarray(r.face_jump)[mask]
            faces["p05_live_jump_owner_num"] = _pair(r.jump_numerator, uj)
            timed("p05", started)

        for name in ("p05n_frozen", "p05n_upwind"):
            if name not in campaigns:
                continue
            started = time.perf_counter()
            a = adapters[name]
            role = a.reconstructions["role"]
            if self.column_block is None:
                r = p05n_action(plan, fields_of(a.owner_values), bc_of(self.bc[name]), role.field_kinds,
                                a.n_pair_index, a.d_pair_index, columns=role.columns)
            else:
                r = blocked_p05n_action(plan, fields_of(a.owner_values), bc_of(self.bc[name]), role.field_kinds,
                                        a.n_pair_index, a.d_pair_index, columns=role.columns,
                                        column_block=self.column_block)
            cells[f"{name}_raw_N"] = _pair(r.raw_N_numerator, uc)
            cells[f"{name}_raw_D"] = _pair(r.raw_D_numerator, uc)
            faces[f"{name}_face_N"] = _pair(r.face_N_numerator, uf)
            faces[f"{name}_face_D"] = _pair(r.face_D_numerator, uf)
            timed(name, started)

        if "p06n" in campaigns or "p06_legacy" in campaigns:
            cells["q1_evolution_volume"] = _pair(plan.cells.evolution_volume, uc)

        if "p06n" in campaigns:
            started = time.perf_counter()
            a = adapters["p06n"]
            if self.column_block is None:
                columns, kinds, groups = p06n_layout([a.reconstructions[v] for v in a.variant_names])
                fields = fields_of(a.owner_values)[:, columns]
                act = p06_action(plan, fields, bc_columns(bc_of(self.bc["p06n"]), columns), kinds, groups)
            else:
                act = blocked_p06n_action(plan, fields_of(a.owner_values), bc_of(self.bc["p06n"]),
                                          [a.reconstructions[v] for v in a.variant_names],
                                          column_block=self.column_block, variant_block=self.variant_block)
            material, remainder = np.asarray(act.material_numerator), np.asarray(act.remainder_numerator)
            correction = np.asarray(act.correction_numerator)
            V = material.shape[0]
            cells["p06n_raw_material"] = [_pair(material[v], uc) for v in range(V)]
            cells["p06n_raw_remainder"] = [_pair(remainder[v], uc) for v in range(V)]
            cells["p06n_raw_total"] = [_pair(material[v] + remainder[v], uc) for v in range(V)]
            faces["p06n_faces_correction"] = [_pair(correction[v], uf) for v in range(V)]
            timed("p06n", started)

        if "p06_legacy" in campaigns:
            started = time.perf_counter()
            legacy = adapters["p06_legacy"]
            cells["p06legacy_raw_centered"] = {}
            faces["p06legacy_faces_correction"] = {}
            for field_name, field in legacy:
                act = p06_action(plan, fields_of(field.owner_values), bc_of(self.bc["p06_legacy"][field_name]),
                                 field.field_kinds, face_multiplier=self.legacy_multiplier)
                material, remainder = np.asarray(act.material_numerator), np.asarray(act.remainder_numerator)
                cells["p06legacy_raw_centered"][field_name] = {
                    "material": _pair(material, uc), "remainder": _pair(remainder, uc),
                    "total": _pair(material + remainder, uc)}
                faces["p06legacy_faces_correction"][field_name] = _pair(act.correction_numerator, uf)
            timed("p06_legacy", started)

        if "p07" in campaigns:
            started = time.perf_counter()
            a = adapters["p07"]
            p07["p07_global_N"] = _pair(self._p07_numerator(fields_of(a.owner_values), bc_of(self.bc["p07"]),
                                                            a.field_kinds), up)
            timed("p07", started)

        if "p07n" in campaigns:
            started = time.perf_counter()
            a = adapters["p07n"]
            fields, bc = fields_of(a.owner_values), bc_of(self.bc["p07n"])
            p07["p07n_global_N"] = _pair(self._p07_numerator(fields, bc, a.reconstructions["N"].field_kinds), up)
            p07["p07n_global_D"] = _pair(self._p07_numerator(fields, bc, a.reconstructions["D"].field_kinds), up)
            timed("p07n", started)

        if rng is None and include_host_only and self.built is not None:
            started = time.perf_counter()
            if host_only_from is not None:
                host_part = {"cells": {}, "p07": {}}
                for section in ("cells", "p07"):
                    for key, value in host_only_from.get(section, {}).items():
                        root = key.split(".")[0]
                        if root in _HOST_ONLY_ROOTS or key.startswith("p06n_raw_R_"):
                            host_part[section][key] = value
            else:
                host_part = host_only_terms(self.env, self.built, campaigns, self.oracle, adapters)
            cells.update(host_part["cells"])
            p07.update(host_part["p07"])
            timed("host_only", started)

        self.timings[f"operators_{'nominal' if rng is None else f'perturb{perturb_seed}'}"] = timings
        return {"cells": cells, "faces": faces, "p07": p07}


def jax_assemble_owner_terms(env: Environment, built: dict, campaigns: tuple, oracle: dict, *,
                             wall_cache: bool = False, host_only_from: Optional[dict] = None) -> dict:
    """Same inputs and output format as :func:`p_shared.owner_closure.assemble_owner_terms`, computed with the
    JAX operators of the owner plan (module docstring). ``host_only_from``: an already-computed host ``out``
    to take the host-only MMS references from instead of recomputing them."""
    return JaxOwnerClosure(env, built, campaigns, oracle, wall_cache=wall_cache).evaluate(
        host_only_from=host_only_from)


# ---------------------------------------------------------------------------
# Owner-space terms, policy and comparison
# ---------------------------------------------------------------------------
def normalized_terms(out: dict, *, vol, owners) -> dict:
    """``{term name: [array, ...]}``: every term of an ``out`` dict in owner space at ``owners`` (module docstring).

    Names are ``section.key`` (``cells.p05_centered``, ``faces.p05n_frozen_face_N``, ``p07.p07_global_N``, ...);
    a term with variants / fields lists all their arrays (P06N: 14 variants; P06-legacy: ``section.key.term``
    over the 4 fields). Missing terms are simply absent."""
    vol = np.asarray(vol, dtype=np.float64)
    owners = np.asarray(owners, dtype=np.int64)
    n_total = len(vol)
    dense = lambda pair: oc.owner_values_from_pairs(pair, owners, n_total)
    vol_o = vol[owners][:, None]
    cells, faces, p07 = out.get("cells", {}), out.get("faces", {}), out.get("p07", {})
    terms: dict = {}
    ev = None
    if "q1_evolution_volume" in cells:
        ev_raw = dense(cells["q1_evolution_volume"])
        terms["cells.q1_evolution_volume"] = [ev_raw]
        ev = np.maximum(ev_raw, _EV_FLOOR)[:, None]

    for key, value in cells.items():
        if key in ("q1_evolution_volume", "p05_antisymmetry_max"):
            continue
        if key.startswith("p06n_raw_"):
            terms[f"cells.{key}"] = [dense(pair) / ev for pair in value]
        elif key == "p06legacy_raw_centered":
            for term in _LEGACY_TERMS:
                terms[f"cells.{key}.{term}"] = [dense(value[f][term]) / ev for f in value]
        else:
            terms[f"cells.{key}"] = [dense(value) / vol_o]
    for key, value in faces.items():
        if key == "p05_live_jump_p07ids":
            continue
        if key == "p05_live_jump_values":
            order = np.argsort(np.asarray(faces["p05_live_jump_p07ids"]), kind="stable")
            terms[f"faces.{key}"] = [np.asarray(value)[order]]
        elif key == "p06n_faces_correction":
            terms[f"faces.{key}"] = [dense(pair) / ev for pair in value]
        elif key == "p06legacy_faces_correction":
            terms[f"faces.{key}"] = [dense(value[f]) / ev for f in value]
        else:
            terms[f"faces.{key}"] = [dense(value) / vol_o]
    for key, value in p07.items():
        terms[f"p07.{key}"] = [dense(value) / vol_o]
    return terms


def _root_key(name: str) -> str:
    """``cells.p06legacy_raw_centered.material`` -> ``p06legacy_raw_centered``."""
    return name.split(".")[1]


def classify_term(name: str) -> str:
    """``"cancellation"``, ``"host_only"`` or ``"operator"`` (non-cancellation JAX term)."""
    root = _root_key(name)
    if root in CANCELLATION_TERMS:
        return "cancellation"
    if root in _HOST_ONLY_ROOTS or root.startswith("p06n_raw_R_"):
        return "host_only"
    return "operator"


def term_campaign(name: str) -> str:
    root = _root_key(name)
    for prefix, campaign in (("p05n_frozen", "p05n_frozen"), ("p05n_upwind", "p05n_upwind"), ("p05_", "p05"),
                             ("p06n", "p06n"), ("p06legacy", "p06_legacy"), ("p07n", "p07n"), ("p07_", "p07"),
                             ("q1_evolution_volume", "p06 (shared)")):
        if root.startswith(prefix):
            return campaign
    return "?"


def evaluate_policy(name: str, max_abs: float, scale: float, floor: Optional[float]) -> dict:
    """One row of the uniform tolerance policy (module docstring) for a term with ``max |JAX - host| =
    max_abs``, ``scale = max |host|`` and (cancellation terms) the measured conditioning ``floor``."""
    kind = classify_term(name)
    if scale > 0.0:
        rel = max_abs / scale
    else:
        rel = 0.0 if max_abs == 0.0 else 1.0e18       # finite sentinel: this payload is written as strict JSON
    row = {"term": name, "campaign": term_campaign(name), "kind": kind, "max_abs": float(max_abs),
           "scale": float(scale), "max_rel_to_scale": float(rel)}
    row["floor"] = None if floor is None else float(floor)
    if kind == "cancellation":
        row["over_floor"] = (None if floor is None else
                             float(max_abs / floor) if floor > 0.0 else (0.0 if max_abs == 0.0 else 1.0e18))
        row["pass_floor_clause"] = bool(floor is not None and max_abs <= CANCELLATION_FLOOR_FACTOR * floor)
        row["pass_rel_clause"] = bool(rel < CANCELLATION_REL_TOL)
        # informational: the term's own conditioning floor already exceeds 1e-8 of its scale
        row["floor_limited"] = bool(floor is not None and floor >= CANCELLATION_REL_TOL * scale)
        row["pass"] = row["pass_floor_clause"]
    else:
        row["over_floor"] = None
        row["pass"] = bool(rel <= NONCANCELLATION_REL_TOL)
    return row


def compare_terms(host_terms: dict, jax_terms: dict, floors: Optional[dict] = None) -> list:
    """Policy rows (:func:`evaluate_policy`) for every host term; a term missing on the JAX side fails."""
    floors = floors or {}
    rows = []
    for name, host_arrays in host_terms.items():
        jax_arrays = jax_terms.get(name)
        scale = max(float(np.max(np.abs(h))) if h.size else 0.0 for h in host_arrays)
        if jax_arrays is None or len(jax_arrays) != len(host_arrays):
            row = evaluate_policy(name, 1.0e18, scale, floors.get(name))
            row["pass"] = False
            row["note"] = "missing in JAX terms"
            rows.append(row)
            continue
        max_abs = max(float(np.max(np.abs(np.asarray(j) - h))) if h.size else 0.0
                      for j, h in zip(jax_arrays, host_arrays))
        rows.append(evaluate_policy(name, max_abs, scale, floors.get(name)))
    return rows


def conditioning_floors(nominal_terms: dict, perturbed_terms: list) -> dict:
    """Per term: ``max`` over the perturbed evaluations (seeds) of ``max |perturbed - nominal|`` (owner space)."""
    floors: dict = {}
    for terms in perturbed_terms:
        for name, nominal in nominal_terms.items():
            if name not in terms:
                continue
            d = max(float(np.max(np.abs(np.asarray(p) - q))) if q.size else 0.0
                    for p, q in zip(terms[name], nominal))
            floors[name] = max(floors.get(name, 0.0), d)
    return floors


# ---------------------------------------------------------------------------
# The closure check
# ---------------------------------------------------------------------------
def _oracle_table(rows) -> list:
    return [{"campaign": r["campaign"], "term": r["term"], "max_abs": r["max_abs"], "max_rel": r["max_rel"],
             "ratio_to_oracle_NR": r["ratio_to_oracle_NR"], "pass": bool(r["pass"]), "note": r.get("note", "")}
            for r in rows]


def run_jax_owner_closure_check(*, n: int, input_root, sidecar_path, paths: dict, campaigns: tuple = CAMPAIGN_FUNCS,
                                floor_seeds=FLOOR_SEEDS, wall_cache: bool = False, output=None,
                                column_block: Optional[int] = None, variant_block: Optional[int] = None,
                                boundary_batch: Optional[int] = None, curvature: str = DEFAULT_CURVATURE,
                                face_quadrature: str = DEFAULT_FACE_QUADRATURE) -> dict:
    """The G1 check at grid ``n`` (mirrors ``owner_closure.run_owner_closure_check``): build the owner rows once,
    run the host ``assemble_owner_terms`` and the JAX assembly on them, and return

    * ``diff_table``: per campaign/term JAX-vs-host rows under the uniform tolerance policy (max abs diff,
      relative to scale, conditioning floor for cancellation terms, pass) and ``all_diff_pass``;
    * ``oracle_jax`` / ``oracle_host``: ``compare_to_oracle`` rows for the JAX terms and the host terms;
    * structure check (``uniq_mismatches``), timings and peak RSS.

    ``output`` (a path) additionally writes the payload as strict JSON. ``column_block`` / ``variant_block`` /
    ``boundary_batch`` run the JAX side through the blocked path of the G3 full-grid replay (see
    :class:`JaxOwnerClosure`); all ``None`` is the plain E6 check.

    ``curvature`` (``"fd"`` default, or ``"autodiff"``) switches the curvature ``K`` of the operator geometry
    (owner rows) and of the host references (``env.ref``) together; the frozen oracles still hold finite-difference
    values, so the oracle rows may move by the operator change.

    ``face_quadrature`` (``"q3"`` default, or ``"q2"``) is the P05/P06 face-node rule of the owner rows, the plan
    and the host references (P07 stays q3); it is recorded in the payload only when it is not ``"q3"``."""
    campaigns = tuple(campaigns)
    wall_started = time.perf_counter()
    marks: dict = {}

    def mark(label, since):
        marks[label] = {"seconds": time.perf_counter() - since, "peak_rss_gib": _peak_rss_gib()}

    started = time.perf_counter()
    env = build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path), curvature=curvature,
                            face_quadrature=face_quadrature)
    mark("environment", started)
    t = env.t
    fixture = oc.selection_fixture(t, env.census)
    owners = np.asarray(fixture["owners"], dtype=np.int64)

    started = time.perf_counter()
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(
        sidecar_path, curvature=curvature, face_quadrature=face_quadrature))
    mark("build_owner_rows", started)
    oracle = ru._load_oracle_owner_values(env, dict(paths), campaigns)

    started = time.perf_counter()
    host_out = oc.assemble_owner_terms(env, built, campaigns, oracle)
    mark("host_assemble", started)

    started = time.perf_counter()
    closure = JaxOwnerClosure(env, built, campaigns, oracle, wall_cache=wall_cache, column_block=column_block,
                              variant_block=variant_block, boundary_batch=boundary_batch)
    mark("jax_plan_and_boundary", started)
    started = time.perf_counter()
    jax_out = closure.evaluate(host_only_from=host_out)
    mark("jax_operators", started)

    host_terms = normalized_terms(host_out, vol=t.vol, owners=owners)
    jax_terms = normalized_terms(jax_out, vol=t.vol, owners=owners)

    started = time.perf_counter()
    perturbed = [normalized_terms(closure.evaluate(perturb_seed=seed), vol=t.vol, owners=owners)
                 for seed in floor_seeds]
    floors = conditioning_floors(jax_terms, perturbed)
    mark("conditioning_floors", started)

    rows = compare_terms(host_terms, jax_terms, floors)
    mismatches = pairs_mismatches(host_out, jax_out)

    started = time.perf_counter()
    oracle_jax = oc.compare_to_oracle(env, jax_out, owners, dict(paths), campaigns)
    oracle_host = oc.compare_to_oracle(env, host_out, owners, dict(paths), campaigns)
    mark("compare_to_oracle", started)

    plan = closure.plan
    payload = {
        "schema": SCHEMA, "n": int(n), "curvature": curvature, "campaigns": list(campaigns),
        "wall_cache": bool(wall_cache),
        "blocking": {"column_block": column_block, "variant_block": variant_block, "boundary_batch": boundary_batch},
        "selection": fixture,
        "plan": {"cells": int(len(plan.cells.raw_ids)), "faces": int(len(plan.faces.census_row)),
                 "p07_faces": int(len(plan.p07.p07_id)), "dirichlet_points": int(len(plan.dirichlet_points)),
                 "neumann_points": int(len(plan.neumann_points))},
        "policy": {"noncancellation_rel_tol": NONCANCELLATION_REL_TOL,
                   "cancellation_floor_factor": CANCELLATION_FLOOR_FACTOR,
                   "cancellation_rel_tol": CANCELLATION_REL_TOL, "ulp": ULP, "floor_seeds": list(floor_seeds)},
        "diff_table": rows, "all_diff_pass": bool(all(r["pass"] for r in rows)),
        "diff_failures": [r["term"] for r in rows if not r["pass"]],
        "diff_failures_all_floor_limited": bool(all(
            r["kind"] == "cancellation" and r["pass_floor_clause"] and r["floor_limited"]
            for r in rows if not r["pass"])),
        "uniq_mismatches": mismatches,
        "oracle_jax": _oracle_table(oracle_jax), "oracle_host": _oracle_table(oracle_host),
        "oracle_jax_all_pass": bool(all(r["pass"] for r in oracle_jax)),
        "oracle_host_all_pass": bool(all(r["pass"] for r in oracle_host)),
        "antisymmetry_max": {"jax": jax_out["cells"].get("p05_antisymmetry_max"),
                             "host": host_out["cells"].get("p05_antisymmetry_max")},
        "phases": marks, "operator_timings": closure.timings,
        "wall_seconds": time.perf_counter() - wall_started, "peak_rss_gib": _peak_rss_gib(),
    }
    if face_quadrature != DEFAULT_FACE_QUADRATURE:
        payload["face_quadrature"] = face_quadrature
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default))
    return payload


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def _print_summary(payload: dict) -> None:
    print(f"N{payload['n']}: owners={len(payload['selection']['owners'])} plan={payload['plan']}")
    print(f"  diff table: {sum(r['pass'] for r in payload['diff_table'])}/{len(payload['diff_table'])} pass; "
          f"uniq mismatches: {payload['uniq_mismatches']}")
    worst = sorted(payload["diff_table"], key=lambda r: -r["max_rel_to_scale"])[:8]
    for r in worst:
        print(f"    {r['term']:50s} {r['kind']:12s} abs={r['max_abs']:.3e} rel={r['max_rel_to_scale']:.3e} "
              f"floor={r['floor']} pass={r['pass']}")
    print(f"  oracle rows JAX {sum(r['pass'] for r in payload['oracle_jax'])}/{len(payload['oracle_jax'])} pass, "
          f"host {sum(r['pass'] for r in payload['oracle_host'])}/{len(payload['oracle_host'])} pass")
    print(f"  wall {payload['wall_seconds']:.1f}s peak RSS {payload['peak_rss_gib']:.2f} GiB")


def main(argv=None) -> int:
    from p_shared.replay_support import DEFAULT_PATHS

    workspace = _SCRIPTS.parents[1]      # .../HSX drbx
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--input-root", default=str(workspace))
    parser.add_argument("--sidecar", default=str(workspace / "work/p07n_extraction_hotspot_audit_20260926/"
                                                             "localized_sidecar.json"))
    parser.add_argument("--campaigns", default=",".join(CAMPAIGN_FUNCS))
    parser.add_argument("--wall-cache", action="store_true")
    parser.add_argument("--curvature", choices=("fd", "autodiff"), default=DEFAULT_CURVATURE)
    parser.add_argument("--face-quadrature", choices=FACE_QUADRATURE_CHOICES, default=DEFAULT_FACE_QUADRATURE,
                        help="P05/P06 face-node rule (P07 stays q3)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    payload = run_jax_owner_closure_check(
        n=args.n, input_root=args.input_root, sidecar_path=args.sidecar, paths=dict(DEFAULT_PATHS),
        campaigns=tuple(args.campaigns.split(",")), wall_cache=args.wall_cache, output=args.output,
        curvature=args.curvature, face_quadrature=args.face_quadrature)
    _print_summary(payload)
    return 0 if (payload["all_diff_pass"] and payload["oracle_jax_all_pass"] and not payload["uniq_mismatches"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
