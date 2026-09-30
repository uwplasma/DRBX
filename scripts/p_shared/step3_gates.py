"""Gates G3.2 and G3.3 of P08 step 3: the combined perpendicular RHS on the bounded owner closure.

Design: ``work/p08_step3_combined_rhs_design_20260930/design.md`` (sections 4, 7 and 8).  Everything runs the combined
call :func:`drbx.native.fci_perpendicular_rhs.perpendicular_rhs` on the plan and the campaign boundary data of the
E6 closure (:class:`p_shared.jax_replay.JaxOwnerClosure`); bounded, single process (11-12 owners).

**G3.2 -- accuracy against the frozen references through the combined path** (:func:`run_g32`).  For each campaign
key (``p05``, ``p05n_frozen``, ``p05n_upwind``, ``p06n``, ``p06_legacy``, ``p07``, ``p07n``) the campaign's own owner
values / kinds / boundary data are fed through ``perpendicular_rhs`` with only that campaign's term enabled and
coefficients that isolate the qualified operator:

* bracket (P05 / P05N): ``rho_star = 1`` and the campaign's own ``(generator, transported)`` pairs as ``raw_pairs``
  (P05N: the N and D role catalogues in one pair list; each *role* is a named column with its own kind).  The first
  pair is named ``(phi, density)`` so the ``bracket`` term itself runs too and is checked against the first raw pair;
  the campaign columns are otherwise named ``c<i>`` (:func:`pair_layout`).  P05N has the whole face domain, P05 the
  default ``p07_valid`` domain, as the E6 closure.
* curvature (P06N: the 14 catalogue variants, one combined call each on its 5 columns / kinds; P06-legacy: 4 fields x 5
  Dirichlet columns with the seam ``face_multiplier``): ``tau`` = the campaign's (1), terms ``curvature`` only.  The
  combined output is per-owner ``material / remainder / q1 / correction`` (divided by the q1 evolution volume); the
  host-format numerators are ``value * max(evolution volume, 1e-300)`` (a conversion exact to one ulp, no more).
* diffusion (P07 / P07N N and D): ``D = 1`` for every field and the sign ``perpendicular_diffusion = -D * P07``, so the
  frozen ``+P07`` is ``-term`` (exact).  The combined call has four field names (``FIELDS``): P07N's five columns run
  in two blocks, each with a dummy ``phi`` (a copy of its first column); the numerators are ``P07 * owner volume``.

The result is converted to the host ``out`` format (the sparse ``(uniq, numerator)`` pairs of
``jax_replay.jax_assemble_owner_terms``, ``uniq`` arrays those of the closure) and checked (a) against the E6
``closure.evaluate()`` terms (the separately gated operators) under the uniform tolerance policy of ``jax_replay``
(design section 8 of the step-2b design; bitwise or roundoff is expected) and (b) against the frozen oracles with
``owner_closure.compare_to_oracle`` (every row; the host-only MMS reference terms are those of the E6 closure).

**G3.3 -- convergence against the published-form continuum reference** (:func:`run_g33`).  The common state is
the P06N ``main_phi_dirichlet`` / ``main_phi_neumann`` variant (n, Te, Ti, omega, phi with the variant's kinds and boundary
data).  The full combined RHS (bracket, curvature, diffusion, total) of the four fields at fixed parameters
(:data:`G33_PARAMS`) is compared with :func:`p_shared.perpendicular_reference_rhs.reference_rhs` at the closure owners:
owner-volume-weighted L2 and max of ``N - R`` per field per term, relative to the term's own reference scale, and the
least-squares scale ``<N, R> / <R, R>`` (a sign or coefficient error shows up as a fit far from 1 and an O(1) relative
error).  :func:`observed_orders` turns the reports of several grids into observed orders and flags.  The closure has
only 11-12 owners, so orders are indicative.

Import as ``from p_shared import step3_gates`` with ``DRBX/scripts`` on ``sys.path`` (never a directory holding a
stray ``operator.py``).  CLI: ``python -m p_shared.step3_gates --grids 32 [--gates g32,g33]`` from ``DRBX/scripts``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import jax                                                                      # noqa: E402
jax.config.update("jax_enable_x64", True)

from drbx.native.fci_perpendicular_p06_operator import bc_columns                # noqa: E402
from drbx.native.fci_perpendicular_rhs import (                                  # noqa: E402
    FIELDS, PHI, PerpendicularParams, perpendicular_columns, perpendicular_rhs)

from p_shared import jax_replay as jr                                            # noqa: E402
from p_shared import owner_closure as oc                                         # noqa: E402
from p_shared import perpendicular_reference_rhs as prr                          # noqa: E402
from p_shared import replay_units as ru                                          # noqa: E402
from p_shared.curvature_reference import DEFAULT_CURVATURE  # noqa: E402
from p_shared.face_quadrature import DEFAULT_FACE_QUADRATURE, FACE_QUADRATURE_CHOICES  # noqa: E402
from p_shared.replay_support import (                                            # noqa: E402
    CAMPAIGN_FUNCS, DEFAULT_PATHS, DEFAULT_SIDECAR, owner_weighted_l2)

__all__ = [
    "SCHEMA_G32", "SCHEMA_G33", "WORKSPACE", "GATES_DIR", "CAMPAIGNS", "G33_VARIANTS", "G33_PARAMS", "PairLayout",
    "pair_layout", "layout_feed", "sparse_pair", "Step3Setup", "build_setup", "combined_out", "host_only_part",
    "term_metrics", "observed_order", "observed_orders", "flag_orders", "run_g32", "run_g33", "orders_from_dir"]

SCHEMA_G32 = "drbx.p08-step3-g32.v1"
SCHEMA_G33 = "drbx.p08-step3-g33.v1"
WORKSPACE = _SCRIPTS.parents[1]
GATES_DIR = WORKSPACE / "work/p08_step3_combined_rhs_design_20260930/gates"
CAMPAIGNS = tuple(CAMPAIGN_FUNCS)

#: G3.3: the common manufactured state and the (fixed, physically sensible) parameters
G33_VARIANTS = ("main_phi_dirichlet", "main_phi_neumann")
G33_PARAMS = {"rho_star": 0.05, "tau": 1.0,
              "diffusion": {"density": 1.0e-2, "Te": 1.0e-2, "Ti": 1.0e-2, "vorticity": 1.0e-2}}
G33_TERMS = ("poisson_bracket", "curvature", "perpendicular_diffusion", "total")
#: an error that grows by more than this factor between two grids is flagged (and asserted by the slow test)
GROWTH_FLAG = 2.0
#: a term whose reference L2 is below this fraction of its field's largest term is degenerate (see ``term_metrics``)
DEGENERATE_FRACTION = 1.0e-8

#: ``compare_to_oracle`` campaign labels -> campaign keys
_ORACLE_LABEL = {"P05": "p05", "p05n_frozen": "p05n_frozen", "p05n_upwind": "p05n_upwind", "P06N": "p06n",
                 "P06_legacy": "p06_legacy", "P07": "p07", "P07N": "p07n"}
_EV_FLOOR = jr._EV_FLOOR
_HOST_ONLY_ROOTS = jr._HOST_ONLY_ROOTS


# ---------------------------------------------------------------------------
# Pure helpers: layout of a campaign's columns / pairs in the combined call's named columns
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PairLayout:
    """``names``: campaign column -> combined column name; ``raw_pairs``: the campaign pairs by name; ``columns``:
    the combined layout (``perpendicular_columns``, ``phi`` last); ``source``: the campaign column of each layout
    column."""

    names: Mapping
    raw_pairs: tuple
    columns: tuple
    source: tuple


def pair_layout(pairs: Sequence[tuple[int, int]]) -> PairLayout:
    """Name the campaign columns of ``pairs`` (``(generator, transported)`` column indices) for the combined call.

    The first pair becomes ``(phi, density)`` (so the ``bracket`` term of the single field ``density`` is exactly the
    first raw pair), every other column is ``c<index>``.  The layout holds exactly the columns the pairs use."""
    pairs = tuple((int(a), int(b)) for a, b in pairs)
    if not pairs:
        raise ValueError("at least one pair is required")
    a0, b0 = pairs[0]
    if a0 == b0:
        raise ValueError("the first pair must have distinct generator and transported columns")
    used = sorted({c for pair in pairs for c in pair})
    names = {c: f"c{c}" for c in used}
    names[a0], names[b0] = PHI, "density"
    raw = tuple((names[a], names[b]) for a, b in pairs)
    columns = perpendicular_columns(("density",), ("bracket",), raw)
    inverse = {name: c for c, name in names.items()}
    return PairLayout(names, raw, columns, tuple(inverse[name] for name in columns))


def layout_feed(columns: Sequence[str], source: Sequence[int], values, bc, kinds):
    """``(state, phi, bc, kinds)`` of the combined call for the layout ``columns`` (``phi`` last) fed from the campaign
    arrays: ``values (n_owners, F)``, ``bc`` with trailing axis ``F`` and ``kinds`` per campaign column."""
    values = np.asarray(values, dtype=np.float64)
    state = {name: values[:, c] for name, c in zip(columns[:-1], source[:-1])}
    phi = values[:, source[-1]]
    kinds_map = {name: str(kinds[c]) for name, c in zip(columns, source)}
    return state, phi, bc_columns(bc, list(source)), kinds_map


def sparse_pair(numerator, uniq: np.ndarray):
    """The host sparse pair ``(uniq, numerator[uniq])`` of a dense ``(n_owners, ...)`` numerator."""
    return uniq, np.asarray(numerator, dtype=np.float64)[uniq]


def host_only_part(out: dict) -> dict:
    """The host-only MMS reference terms (``raw_R``, ``raw_R_*``, ``O_q3``) of an ``out`` dict."""
    part: dict = {"cells": {}, "p07": {}}
    for section in ("cells", "p07"):
        for key, value in out.get(section, {}).items():
            if key.split(".")[0] in _HOST_ONLY_ROOTS or key.startswith("p06n_raw_R_"):
                part[section][key] = value
    return part


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Step3Setup:
    """The bounded closure of one grid: environment, owner rows, oracle inputs and the E6 :class:`JaxOwnerClosure`."""

    n: int
    env: object
    built: dict
    owners: np.ndarray
    oracle: dict
    closure: object
    campaigns: tuple
    paths: dict
    seconds: float
    curvature: str = DEFAULT_CURVATURE
    face_quadrature: str = DEFAULT_FACE_QUADRATURE


def build_setup(n: int, campaigns: Sequence[str] = CAMPAIGNS, *, input_root=WORKSPACE, sidecar_path=DEFAULT_SIDECAR,
                paths: Optional[dict] = None, curvature: str = DEFAULT_CURVATURE,
                face_quadrature: str = DEFAULT_FACE_QUADRATURE) -> Step3Setup:
    """Build the owner-closure rows, the oracle inputs and the JAX closure of ``campaigns`` at grid ``n``.
    ``curvature`` (``"autodiff"`` default | ``"fd"``) switches the operator geometry and the host references
    (``env.ref``, hence the G3.3 continuum reference) together.  ``face_quadrature`` (``"q3"`` default | ``"q2"``)
    is the P05/P06 face-node rule of the rows, the plan and the host references (P07 stays q3)."""
    started = time.perf_counter()
    campaigns = tuple(campaigns)
    paths = dict(DEFAULT_PATHS if paths is None else paths)
    env = jr.build_environment(n=n, input_root=Path(input_root), sidecar_path=Path(sidecar_path),
                               curvature=curvature, face_quadrature=face_quadrature)
    owners = np.asarray(oc.selection_fixture(env.t, env.census)["owners"], dtype=np.int64)
    built = oc.build_owner_rows(env, owners, provider=oc.load_provider_for_env(
        sidecar_path, curvature=curvature, face_quadrature=face_quadrature))
    oracle = ru._load_oracle_owner_values(env, paths, campaigns)
    closure = jr.JaxOwnerClosure(env, built, campaigns, oracle)
    return Step3Setup(n=int(n), env=env, built=built, owners=owners, oracle=oracle, closure=closure,
                      campaigns=campaigns, paths=paths, seconds=time.perf_counter() - started, curvature=curvature,
                      face_quadrature=face_quadrature)


# ---------------------------------------------------------------------------
# The combined path per campaign -> host-format ``out``
# ---------------------------------------------------------------------------
def _bracket(closure, values, bc, kinds, pairs, jump_mask):
    """P05 / P05N through the combined call: ``(P05Terms of the raw pairs, bracket-term mismatch)``."""
    layout = pair_layout(pairs)
    state, phi, bc_l, kinds_l = layout_feed(layout.columns, layout.source, values, bc, kinds)
    res = perpendicular_rhs(closure.plan, state, phi, bc_l, kinds_l, PerpendicularParams(rho_star=1.0),
                            fields=("density",), terms=("bracket",), raw_pairs=layout.raw_pairs, jump_mask=jump_mask)
    raw = res.raw_pairs
    first = np.asarray(raw.centered_owner)[:, 0] + np.asarray(raw.jump_owner)[:, 0]
    mismatch = float(np.max(np.abs(np.asarray(res.terms["density"]["poisson_bracket"]) - first)))
    return raw, mismatch, res.diagnostics


def _curvature(closure, values5, bc5, kinds5, *, face_multiplier=None):
    """The combined curvature term on a five-column state ``(n, Te, Ti, omega, phi)``: per-field owner arrays
    ``material, remainder, q1, correction`` ``(n_owners, 4)`` (divided by the q1 evolution volume) and the counters."""
    kinds = dict(zip((*FIELDS, PHI), (str(k) for k in kinds5)))
    values5 = np.asarray(values5, dtype=np.float64)
    state = {f: values5[:, i] for i, f in enumerate(FIELDS)}
    res = perpendicular_rhs(closure.plan, state, values5[:, 4], bc5, kinds, PerpendicularParams(tau=1.0),
                            fields=FIELDS, terms=("curvature",), face_multiplier=face_multiplier)
    stack = lambda key: np.stack([np.asarray(res.detail[f][key]) for f in FIELDS], axis=1)
    arrays = {"material": stack("curvature_material"), "remainder": stack("curvature_remainder"),
              "q1": stack("curvature_q1"), "correction": stack("curvature_correction")}
    curvature = np.stack([np.asarray(res.terms[f]["curvature"]) for f in FIELDS], axis=1)
    return arrays, curvature, {k: int(v) for k, v in res.diagnostics.items()}


def _diffusion_action(closure, values, bc, kinds) -> np.ndarray:
    """``+P07 action`` ``(n_owners, F)`` through the combined diffusion term with ``D = 1`` (``-term``, exact), the
    columns in blocks of the four field names with a dummy ``phi`` (a copy of the block's first column)."""
    values = np.asarray(values, dtype=np.float64)
    F = values.shape[1]
    out = np.empty(values.shape)
    for start in range(0, F, len(FIELDS)):
        cols = list(range(start, min(start + len(FIELDS), F)))
        names = FIELDS[:len(cols)]
        source = cols + [cols[0]]
        state, phi, bc_b, kinds_b = layout_feed((*names, PHI), source, values, bc, kinds)
        res = perpendicular_rhs(closure.plan, state, phi, bc_b, kinds_b,
                                PerpendicularParams(diffusion={f: 1.0 for f in names}), fields=names,
                                terms=("diffusion",))
        for f, c in zip(names, cols):
            out[:, c] = -np.asarray(res.terms[f]["perpendicular_diffusion"])
    return out


def combined_out(closure, campaigns: Sequence[str], *, host_part: Optional[dict] = None) -> tuple[dict, dict]:
    """The host-format ``out`` dict of ``campaigns`` computed through the combined RHS and a per-campaign
    diagnostics dict (``bracket_term_mismatch``, P06 counters, ...).  ``host_part`` (:func:`host_only_part` of an E6
    ``out``) supplies the host-only MMS reference terms."""
    plan, adapters, t = closure.plan, closure.adapters, closure.env.t
    campaigns = tuple(campaigns)
    uc, uf, uj, up = closure.uniq_cells, closure.uniq_faces, closure.uniq_jump, closure.uniq_p07
    vol = np.asarray(t.vol, dtype=np.float64)[:, None]
    ev = np.maximum(np.asarray(plan.cells.evolution_volume, dtype=np.float64), _EV_FLOOR)[:, None]
    cells: dict = {}
    faces: dict = {}
    p07: dict = {}
    diagnostics: dict = {}

    if "p05" in campaigns:
        a = adapters["p05"]
        raw, mismatch, diag = _bracket(closure, a.owner_values, closure.bc["p05"], a.field_kinds, a.pairs, None)
        mask = closure.jump_mask
        cells["p05_centered"] = sparse_pair(raw.centered_numerator, uc)
        cells["p05_antisymmetry_max"] = float(raw.antisymmetry)
        faces["p05_live_jump_p07ids"] = np.asarray(plan.faces.p07_id)[mask].astype(np.int64)
        faces["p05_live_jump_values"] = np.asarray(raw.face_jump)[mask]
        faces["p05_live_jump_owner_num"] = sparse_pair(raw.jump_numerator, uj)
        diagnostics["p05"] = {"bracket_term_mismatch": mismatch, "antisymmetry": float(raw.antisymmetry)}

    for name in ("p05n_frozen", "p05n_upwind"):
        if name not in campaigns:
            continue
        a = adapters[name]
        role = a.reconstructions["role"]
        columns = np.asarray(role.columns)
        pairs = list(a.n_pair_index) + list(a.d_pair_index)
        P = len(a.n_pair_index)
        values = np.asarray(a.owner_values)[:, columns]
        bc = bc_columns(closure.bc[name], columns)
        raw, mismatch, _diag = _bracket(closure, values, bc, role.field_kinds, pairs,
                                        np.ones(len(plan.faces.census_row), dtype=bool))
        cn, jn = np.asarray(raw.centered_numerator), np.asarray(raw.jump_numerator)
        cells[f"{name}_raw_N"], cells[f"{name}_raw_D"] = sparse_pair(cn[:, :P], uc), sparse_pair(cn[:, P:], uc)
        faces[f"{name}_face_N"], faces[f"{name}_face_D"] = sparse_pair(jn[:, :P], uf), sparse_pair(jn[:, P:], uf)
        diagnostics[name] = {"bracket_term_mismatch": mismatch, "antisymmetry": float(raw.antisymmetry)}

    if "p06n" in campaigns or "p06_legacy" in campaigns:
        cells["q1_evolution_volume"] = sparse_pair(plan.cells.evolution_volume, uc)

    if "p06n" in campaigns:
        a = adapters["p06n"]
        material, remainder, total, correction, counters = [], [], [], [], {}
        for variant in a.variant_names:
            rec = a.reconstructions[variant]
            columns = np.asarray(rec.columns)
            arrays, _curv, diag = _curvature(closure, np.asarray(a.owner_values)[:, columns],
                                             bc_columns(closure.bc["p06n"], columns), rec.field_kinds)
            material.append(sparse_pair(arrays["material"] * ev, uc))
            remainder.append(sparse_pair(arrays["remainder"] * ev, uc))
            total.append(sparse_pair(arrays["q1"] * ev, uc))
            correction.append(sparse_pair(arrays["correction"] * ev, uf))
            counters[variant] = diag
        cells["p06n_raw_material"], cells["p06n_raw_remainder"], cells["p06n_raw_total"] = material, remainder, total
        faces["p06n_faces_correction"] = correction
        diagnostics["p06n"] = {"q3_counters": counters}

    if "p06_legacy" in campaigns:
        cells["p06legacy_raw_centered"] = {}
        faces["p06legacy_faces_correction"] = {}
        counters = {}
        for field_name, field in adapters["p06_legacy"]:
            arrays, _curv, diag = _curvature(closure, field.owner_values, closure.bc["p06_legacy"][field_name],
                                             field.field_kinds, face_multiplier=closure.legacy_multiplier)
            cells["p06legacy_raw_centered"][field_name] = {
                "material": sparse_pair(arrays["material"] * ev, uc),
                "remainder": sparse_pair(arrays["remainder"] * ev, uc), "total": sparse_pair(arrays["q1"] * ev, uc)}
            faces["p06legacy_faces_correction"][field_name] = sparse_pair(arrays["correction"] * ev, uf)
            counters[field_name] = diag
        diagnostics["p06_legacy"] = {"q3_counters": counters}

    if "p07" in campaigns:
        a = adapters["p07"]
        action = _diffusion_action(closure, a.owner_values, closure.bc["p07"], a.field_kinds)
        p07["p07_global_N"] = sparse_pair(action * vol, up)

    if "p07n" in campaigns:
        a = adapters["p07n"]
        values, bc = a.owner_values, closure.bc["p07n"]
        p07["p07n_global_N"] = sparse_pair(
            _diffusion_action(closure, values, bc, a.reconstructions["N"].field_kinds) * vol, up)
        p07["p07n_global_D"] = sparse_pair(
            _diffusion_action(closure, values, bc, a.reconstructions["D"].field_kinds) * vol, up)

    if host_part is not None:
        cells.update(host_part.get("cells", {}))
        p07.update(host_part.get("p07", {}))
    return {"cells": cells, "faces": faces, "p07": p07}, diagnostics


# ---------------------------------------------------------------------------
# G3.2
# ---------------------------------------------------------------------------
def _campaign_of_row(row: dict) -> str:
    campaign = jr.term_campaign(row["term"])
    return "p06n" if campaign == "p06 (shared)" else campaign


def _summarize_diff(rows: list) -> dict:
    ops = [r for r in rows if r["kind"] != "host_only"]
    return {
        "terms": len(ops), "bitwise_terms": sum(r["max_abs"] == 0.0 for r in ops),
        "max_abs": max((r["max_abs"] for r in ops), default=0.0),
        "max_rel_to_scale": max((r["max_rel_to_scale"] for r in ops), default=0.0),
        "cancellation_terms": sum(r["kind"] == "cancellation" for r in ops),
        "pass": all(r["pass"] for r in rows),
        "failures": [r["term"] for r in rows if not r["pass"]]}


def _summarize_oracle(rows: list) -> dict:
    ratios = [r["ratio_to_oracle_NR"] for r in rows if r["ratio_to_oracle_NR"] is not None]
    return {"rows": len(rows), "passed": sum(bool(r["pass"]) for r in rows),
            "max_abs": max((r["max_abs"] for r in rows), default=0.0),
            "worst_ratio_to_oracle_NR": max(ratios, default=0.0),
            "pass": all(r["pass"] for r in rows), "failures": [r["term"] for r in rows if not r["pass"]]}


def run_g32(n: int, *, setup: Optional[Step3Setup] = None, campaigns: Sequence[str] = CAMPAIGNS,
            output_dir=None, curvature: str = DEFAULT_CURVATURE,
            face_quadrature: str = DEFAULT_FACE_QUADRATURE) -> dict:
    """Gate G3.2 at grid ``n`` (module docstring): JSON-able report; ``output_dir`` additionally writes
    ``N{n}_g32.json`` there.  ``curvature`` and ``face_quadrature`` apply only when no ``setup`` is given (a
    setup carries its own)."""
    started = time.perf_counter()
    campaigns = tuple(campaigns)
    setup = setup if setup is not None else build_setup(n, campaigns, curvature=curvature,
                                                        face_quadrature=face_quadrature)
    missing = set(campaigns) - set(setup.campaigns)
    if missing:
        raise ValueError(f"the setup was built without {sorted(missing)}")
    closure, env, owners = setup.closure, setup.env, setup.owners
    vol = env.t.vol

    e6_out = closure.evaluate()
    comb_out, diagnostics = combined_out(closure, campaigns, host_part=host_only_part(e6_out))
    mismatches = jr.pairs_mismatches(e6_out, comb_out)

    terms_e6 = jr.normalized_terms(e6_out, vol=vol, owners=owners)
    terms_c = jr.normalized_terms(comb_out, vol=vol, owners=owners)
    rows = jr.compare_terms(terms_e6, terms_c)
    floors_measured = False
    if any(r["kind"] == "cancellation" and r["max_abs"] > 0.0 for r in rows):
        perturbed = [jr.normalized_terms(closure.evaluate(perturb_seed=s), vol=vol, owners=owners)
                     for s in jr.FLOOR_SEEDS]
        rows = jr.compare_terms(terms_e6, terms_c, jr.conditioning_floors(terms_e6, perturbed))
        floors_measured = True
    oracle_rows = oc.compare_to_oracle(env, comb_out, owners, setup.paths, campaigns)

    per_campaign: dict = {}
    for campaign in campaigns:
        diff_rows = [r for r in rows if _campaign_of_row(r) == campaign]
        oracle = [r for r in oracle_rows if _ORACLE_LABEL.get(r["campaign"]) == campaign]
        entry = {"vs_e6": _summarize_diff(diff_rows), "vs_oracle": _summarize_oracle(oracle),
                 "diagnostics": diagnostics.get(campaign, {})}
        entry["pass"] = bool(entry["vs_e6"]["pass"] and entry["vs_oracle"]["pass"])
        per_campaign[campaign] = entry
    bracket_mismatch = max((d.get("bracket_term_mismatch", 0.0) for d in diagnostics.values()), default=0.0)
    accounted = sum(e["vs_oracle"]["rows"] for e in per_campaign.values())
    report = {
        "schema": SCHEMA_G32, "n": int(n), "curvature": setup.curvature, "owners": [int(o) for o in owners], "campaigns": list(campaigns),
        "coefficients": {"rho_star": 1.0, "tau": 1.0, "diffusion": 1.0, "diffusion_sign": "frozen +P07 = -(combined term)"},
        "policy": {"noncancellation_rel_tol": jr.NONCANCELLATION_REL_TOL,
                   "cancellation_floor_factor": jr.CANCELLATION_FLOOR_FACTOR, "floors_measured": floors_measured},
        "per_campaign": per_campaign,
        "oracle_rows": len(oracle_rows), "oracle_rows_passed": sum(bool(r["pass"]) for r in oracle_rows),
        "oracle_rows_accounted": accounted,
        "diff_terms": len(rows), "bitwise_terms": sum(r["max_abs"] == 0.0 for r in rows if r["kind"] != "host_only"),
        "uniq_mismatches": mismatches, "bracket_term_mismatch_max": bracket_mismatch,
        "diff_table": rows,
        "oracle_table": [{k: r[k] for k in ("campaign", "term", "max_abs", "max_rel", "ratio_to_oracle_NR", "pass", "note")}
                         for r in oracle_rows],
        "pass": bool(all(e["pass"] for e in per_campaign.values()) and not mismatches and bracket_mismatch == 0.0
                     and accounted == len(oracle_rows)),
        "seconds": time.perf_counter() - started, "setup_seconds": setup.seconds,
    }
    if setup.face_quadrature != "q3":
        report["face_quadrature"] = setup.face_quadrature
    _write(report, output_dir, f"N{n}_g32.json")
    return report


# ---------------------------------------------------------------------------
# G3.3
# ---------------------------------------------------------------------------
def _finite(x) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else float(x)


def term_metrics(combined, reference, owner_volume, *, fallback_l2: Optional[float] = None,
                 fallback_max: Optional[float] = None, degenerate: float = DEGENERATE_FRACTION) -> dict:
    """Owner-volume-weighted L2 and max of ``combined - reference`` (absolute and relative to the reference's own L2 /
    max), and the volume-weighted least-squares scale ``<N, R> / <R, R>`` of the two ``(n_owners,)`` arrays.

    A term whose continuum reference vanishes identically (e.g. the bracket ``[phi, g]`` when ``phi`` is a function of
    ``g``: its reference is roundoff) has no scale of its own: when ``ref_l2 < degenerate * fallback_l2`` the relative
    errors use ``fallback_l2`` / ``fallback_max`` (the largest term of the same field), ``degenerate_reference`` is set
    and ``fit_scale`` is ``None``."""
    combined = np.asarray(combined, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    volume = np.asarray(owner_volume, dtype=np.float64)
    diff = combined - reference
    l2 = owner_weighted_l2(diff, volume)
    ref_l2 = owner_weighted_l2(reference, volume)
    max_abs, ref_max = float(np.max(np.abs(diff))), float(np.max(np.abs(reference)))
    is_degenerate = bool(fallback_l2 is not None and ref_l2 < degenerate * fallback_l2)
    scale_l2, scale_max = (fallback_l2, fallback_max) if is_degenerate else (ref_l2, ref_max)
    rr = float(np.sum(volume * reference ** 2))
    ratio = lambda a, b: None if b is None or b == 0.0 or not math.isfinite(a) else float(a / b)
    return {"l2": _finite(l2), "max_abs": _finite(max_abs), "ref_l2": _finite(ref_l2), "ref_max": _finite(ref_max),
            "rel_l2": ratio(l2, scale_l2), "rel_max": ratio(max_abs, scale_max),
            "fit_scale": None if is_degenerate else ratio(float(np.sum(volume * combined * reference)), rr),
            "degenerate_reference": is_degenerate,
            "finite": bool(np.all(np.isfinite(combined)) and np.all(np.isfinite(reference)))}


def run_g33(n: int, *, setup: Optional[Step3Setup] = None, variants: Sequence[str] = G33_VARIANTS,
            params: Optional[Mapping] = None, output_dir=None, curvature: str = DEFAULT_CURVATURE,
            face_quadrature: str = DEFAULT_FACE_QUADRATURE) -> dict:
    """Gate G3.3 at grid ``n`` (module docstring): JSON-able report; ``output_dir`` additionally writes
    ``N{n}_g33.json`` there.  ``curvature`` and ``face_quadrature`` apply only when no ``setup`` is given (a
    setup carries its own);
    with ``"autodiff"`` the operator and the continuum reference use autodiff K together."""
    started = time.perf_counter()
    params = dict(G33_PARAMS if params is None else params)
    setup = setup if setup is not None else build_setup(n, ("p06n",), curvature=curvature,
                                                        face_quadrature=face_quadrature)
    if "p06n" not in setup.campaigns:
        raise ValueError("G3.3 needs the p06n campaign in the setup")
    closure, env, owners = setup.closure, setup.env, setup.owners
    adapter = closure.adapters["p06n"]
    support = prr.owner_support(env, setup.built)
    volume = np.asarray(env.t.vol, dtype=np.float64)[owners]
    layout = (*FIELDS, PHI)
    rhs_params = PerpendicularParams(rho_star=params["rho_star"], tau=params["tau"],
                                     diffusion=dict(params["diffusion"]))
    ref_params = prr.ReferenceParams(rho_star=params["rho_star"], tau=params["tau"],
                                     diffusion=dict(params["diffusion"]))

    report_variants: dict = {}
    for variant in variants:
        rec = adapter.reconstructions[variant]
        columns = np.asarray(rec.columns)
        values = np.asarray(adapter.owner_values, dtype=np.float64)[:, columns]
        state = {f: values[:, i] for i, f in enumerate(FIELDS)}
        res = perpendicular_rhs(closure.plan, state, values[:, 4], bc_columns(closure.bc["p06n"], columns),
                                dict(zip(layout, rec.field_kinds)), rhs_params)
        ref = prr.reference_rhs(env, owners, prr.p06n_state(env, variant), ref_params, support=support)
        fields: dict = {}
        for f in FIELDS:
            mine = {t: np.asarray(res.total[f] if t == "total" else res.terms[f][t])[owners] for t in G33_TERMS}
            fallback_l2 = max(owner_weighted_l2(ref[f][t], volume) for t in G33_TERMS)
            fallback_max = max(float(np.max(np.abs(ref[f][t]))) for t in G33_TERMS)
            fields[f] = {t: term_metrics(mine[t], ref[f][t], volume, fallback_l2=fallback_l2,
                                         fallback_max=fallback_max) for t in G33_TERMS}
        report_variants[variant] = {
            "columns": [int(c) for c in columns], "kinds": list(rec.field_kinds), "fields": fields,
            "q3_counters": {k: int(v) for k, v in res.diagnostics.items() if k != "antisymmetry"}}
    worst = max((m["rel_l2"] for v in report_variants.values() for e in v["fields"].values() for m in e.values()
                 if m["rel_l2"] is not None), default=None)
    report = {
        "schema": SCHEMA_G33, "n": int(n), "curvature": setup.curvature, "owners": [int(o) for o in owners], "params": params,
        "variants": list(variants), "terms": list(G33_TERMS), "n_owners": int(len(owners)),
        "owner_volume": [float(v) for v in volume], "results": report_variants, "max_rel_l2": worst,
        "seconds": time.perf_counter() - started, "setup_seconds": setup.seconds,
    }
    if setup.face_quadrature != "q3":
        report["face_quadrature"] = setup.face_quadrature
    _write(report, output_dir, f"N{n}_g33.json")
    return report


# ---------------------------------------------------------------------------
# Observed orders and flags
# ---------------------------------------------------------------------------
def observed_order(coarse: Optional[float], fine: Optional[float], n_coarse: int, n_fine: int) -> Optional[float]:
    """``log(e_coarse / e_fine) / log(n_fine / n_coarse)`` (``None`` unless both errors are positive and finite)."""
    if coarse is None or fine is None or not (coarse > 0.0 and fine > 0.0):
        return None
    return float(math.log(coarse / fine) / math.log(n_fine / n_coarse))


def flag_orders(entry: dict, *, growth: float = GROWTH_FLAG) -> list:
    """Flags of one ``observed_orders`` entry (``rel_l2`` per grid): ``non_finite``, ``O(1)`` (some ``rel_l2 >= 1``),
    ``not_decreasing`` (a finer grid is not smaller), ``grows_more_than_<growth>x``."""
    flags = []
    errors = entry["rel_l2"]
    if any(e is None or not math.isfinite(e) for e in errors) or not all(entry["finite"]):
        flags.append("non_finite_or_undefined")
    if any(e is not None and e >= 1.0 for e in errors):
        flags.append("O(1)")
    for (n0, e0), (n1, e1) in zip(zip(entry["n"], errors), zip(entry["n"][1:], errors[1:])):
        if e0 is None or e1 is None:
            continue
        if e1 >= e0:
            flags.append(f"not_decreasing_{n0}_to_{n1}")
        if e1 > growth * e0:
            flags.append(f"grows_more_than_{growth:g}x_{n0}_to_{n1}")
    return flags


def observed_orders(reports: Mapping[int, dict], *, growth: float = GROWTH_FLAG) -> dict:
    """``{variant: {field: {term: entry}}}`` from G3.3 reports ``{n: report}``: per grid ``rel_l2``, ``l2`` (absolute),
    ``max_abs``, ``fit_scale`` and, between consecutive grids, the observed orders of each, plus ``flags``."""
    ns = sorted(int(n) for n in reports)
    reports = {int(n): r for n, r in reports.items()}
    result: dict = {}
    for variant in reports[ns[0]]["results"]:
        result[variant] = {}
        for f in FIELDS:
            result[variant][f] = {}
            for term in G33_TERMS:
                series = [reports[n]["results"][variant]["fields"][f][term] for n in ns]
                entry = {"n": ns, "finite": [bool(s["finite"]) for s in series]}
                for key in ("rel_l2", "rel_max", "l2", "max_abs", "fit_scale"):
                    entry[key] = [s[key] for s in series]
                for key in ("rel_l2", "l2", "max_abs"):
                    entry[f"order_{key}"] = [observed_order(entry[key][i], entry[key][i + 1], ns[i], ns[i + 1])
                                             for i in range(len(ns) - 1)]
                entry["flags"] = flag_orders(entry, growth=growth)
                result[variant][f][term] = entry
    return result


def orders_from_dir(directory=GATES_DIR, grids: Optional[Sequence[int]] = None, *, write: bool = True) -> dict:
    """:func:`observed_orders` of the saved ``N{n}_g33.json`` files of ``directory`` (all, or ``grids``); written to
    ``g33_orders.json`` there when ``write``."""
    directory = Path(directory)
    if grids is None:
        grids = sorted(int(p.name[1:p.name.index("_")]) for p in directory.glob("N*_g33.json"))
    reports = {int(n): json.loads((directory / f"N{n}_g33.json").read_text()) for n in grids}
    orders = observed_orders(reports)
    if write:
        (directory / "g33_orders.json").write_text(json.dumps({"grids": sorted(reports), "orders": orders}, indent=2,
                                                              allow_nan=False, default=jr._json_default))
    return orders


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------
def _write(report: dict, output_dir, name: str) -> None:
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(json.dumps(report, indent=2, allow_nan=False, default=jr._json_default))


def _print_g32(report: dict) -> None:
    print(f"G3.2 N{report['n']}: pass={report['pass']} oracle rows {report['oracle_rows_passed']}/{report['oracle_rows']} "
          f"(accounted {report['oracle_rows_accounted']}), E6 terms {report['diff_terms']} "
          f"(bitwise {report['bitwise_terms']}), uniq mismatches {len(report['uniq_mismatches'])}, "
          f"bracket-term mismatch {report['bracket_term_mismatch_max']:.3g}, {report['seconds']:.0f}s")
    for campaign, e in report["per_campaign"].items():
        d, o = e["vs_e6"], e["vs_oracle"]
        print(f"  {campaign:12s} vs E6: {d['terms']:3d} terms, bitwise {d['bitwise_terms']:3d}, max abs {d['max_abs']:.2e}, "
              f"max rel {d['max_rel_to_scale']:.2e}, pass {d['pass']} | oracle {o['passed']}/{o['rows']} pass, "
              f"max abs {o['max_abs']:.2e}, worst ratio {o['worst_ratio_to_oracle_NR']:.2e}")


def _print_g33(report: dict) -> None:
    print(f"G3.3 N{report['n']} ({report['n_owners']} owners), params {report['params']}")
    for variant, v in report["results"].items():
        print(f"  {variant}")
        for f, entry in v["fields"].items():
            cells = "  ".join(f"{t[:5]}:{m['rel_l2']:.2e}/" + ("deg" if m["fit_scale"] is None else f"{m['fit_scale']:.3f}")
                              if m["rel_l2"] is not None else f"{t[:5]}:n/a" for t, m in entry.items())
            print(f"    {f:10s} rel_l2/fit  {cells}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grids", default="32")
    parser.add_argument("--gates", default="g32,g33")
    parser.add_argument("--curvature", choices=("fd", "autodiff"), default=DEFAULT_CURVATURE)
    parser.add_argument("--face-quadrature", choices=FACE_QUADRATURE_CHOICES, default=DEFAULT_FACE_QUADRATURE,
                        help="P05/P06 face-node rule (P07 stays q3)")
    parser.add_argument("--output-dir", default=None,
                        help="default: the saved-gates directory for fd; a sibling '<dir>_autodiff' for autodiff; "
                             "'<dir>_q2' for q2 (autodiff), '<dir>_fd_q2' for q2 with fd")
    parser.add_argument("--orders-only", action="store_true", help="only combine the saved N*_g33.json files")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.face_quadrature != "q3":
            suffix = "_q2" if args.curvature == "autodiff" else "_fd_q2"
            args.output_dir = str(GATES_DIR.with_name(GATES_DIR.name + suffix))
        else:
            args.output_dir = str(GATES_DIR if args.curvature == "fd" else GATES_DIR.with_name(GATES_DIR.name + "_autodiff"))
    gates = set(args.gates.split(","))
    ok = True
    if not args.orders_only:
        for n in (int(g) for g in args.grids.split(",")):
            campaigns = CAMPAIGNS if "g32" in gates else ("p06n",)
            setup = build_setup(n, campaigns, curvature=args.curvature, face_quadrature=args.face_quadrature)
            print(f"N{n}: setup {setup.seconds:.0f}s, owners {len(setup.owners)}", flush=True)
            if "g32" in gates:
                report = run_g32(n, setup=setup, output_dir=args.output_dir)
                _print_g32(report)
                ok = ok and report["pass"]
            if "g33" in gates:
                report = run_g33(n, setup=setup, output_dir=args.output_dir)
                _print_g33(report)
    if "g33" in gates:
        try:
            orders = orders_from_dir(args.output_dir)
        except (FileNotFoundError, ValueError):
            orders = {}
        for variant, fields in orders.items():
            for f, terms in fields.items():
                for term, e in terms.items():
                    if e["flags"]:
                        print(f"  FLAG {variant}/{f}/{term}: {e['flags']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
