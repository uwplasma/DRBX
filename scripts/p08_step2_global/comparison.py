"""Full-grid comparison of JAX operator terms with the frozen oracles (P08 step 2b, G3).

``compare_operator_terms`` is the finalize/compare half of ``p_shared.replay_units.reduce_grid`` applied
to the host-format ``out`` dict of the JAX replay (``p_shared.jax_replay.JaxOwnerClosure.evaluate``: sparse
``(uniq, numerator)`` pairs per campaign key, the same layout the step-1 replay units write). Everything is
formed exactly as ``reduce_grid`` forms it: the same accumulators (``replay_units.OwnerAccumulator``), the
same divisions (``t.vol`` for cell/face/P07 terms, ``max(evolution_volume, 1e-300)`` for P06 terms, P06N
corrections divided **once**), the same saved oracle arrays, region masks and ``archived_error``, and the
same ``compare_owner_term`` / ``compare_pointwise_only`` calls (Tier-B ratio, corrected pointwise cap,
the P05 ``cap_reference`` of the ``live_jump_vs_old_U_minus_A`` row).

**Step 3.0 rules** (the two structural effects G3 exposed; the step-1 ``reduce_grid`` is unchanged and keeps
the literal rules):

* P06N ``raw_*`` terms use ``tier_b="auto"``: a variant whose archived error is roundoff-level relative to the
  term's own magnitude (``archived_l2 <= ROUNDOFF_REL * max-over-variants owner-L2 of the saved term``) is gated by
  the absolute roundoff floor instead of the Tier-B ratio. The rule is generic; the catalogue's constant-field
  ``control_*`` variants (``p06n_core.CONTROL_CASES`` and their ``:D`` forms) are the expected set, and the
  classification is cross-checked against that list (a mismatch either way fails the variant).
* Cancellation terms use the conditioning-floor pointwise cap ``ULP_FACTOR * eps * constituent_scale`` instead of the
  entrywise ``1e-11 * scale`` cap (no measured floor exists at full grid): P05 per-face ``live_jump_vs_upwind``
  (constituent scale ``max |saved centered| * owner_volume``, the owner flux integrals whose face values the jump
  differences) and P05N ``face_N`` / ``face_D`` (``max |saved raw_N|, |saved raw_D|``, the owner-level densities of
  the same operator, over all columns). See ``p_shared.replay_support`` for the constants.

**Scope.** Only operator terms are compared. The MMS reference terms (P05N ``raw_R``, P06N
``raw_R_material/remainder/total``, P07N ``global_O_q3``) are host-only, need the analytic fields at
every raw cell / face node, and were already gated in step 1; they are omitted here (``OMITTED_TERMS``,
recorded in ``replay.json`` and the report). The oracle's own saved ``R`` arrays still enter as the
``archived_error`` reference exactly as in ``reduce_grid``.

``tests/test_p08_step2_campaign_real.py`` checks this function against ``reduce_grid`` itself (fake unit
outputs built from the saved oracle arrays) at N32.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from p_shared.replay_support import (                                        # noqa: E402
    _load_p05_upwind, P05_PREFIX_JUMP_RECORD_SIGN, _summarize_variants, compare_owner_term, compare_pointwise_only, owner_weighted_l2)
from p_shared.replay_units import OwnerAccumulator                            # noqa: E402

__all__ = ["OMITTED_TERMS", "compare_operator_terms"]

#: campaign -> the host-only MMS reference terms this comparison leaves out (see the module docstring)
OMITTED_TERMS = {
    "p05n_frozen": ["raw_R"],
    "p05n_upwind": ["raw_R"],
    "p06n": ["raw_R_material", "raw_R_remainder", "raw_R_total"],
    "p07n": ["global_O_q3"],
}

_P05N_ROOTS = (("p05n_frozen", "p05n_frozen"), ("p05n_upwind", "p05n_p06n_upwind"))


def _acc(owners: int, pair) -> OwnerAccumulator:
    """A dense accumulator holding one sparse pair (``reduce_grid``: ``OwnerAccumulator.add(uniq, values)``)."""
    uniq, values = pair
    acc = OwnerAccumulator(owners, tuple(np.shape(values)[1:]))
    acc.add(uniq, values)
    return acc


def _expected_roundoff_variants(variants, control_cases) -> frozenset:
    """The catalogue's explicit list of exactly-zero (constant-field control) variants, ``x`` and ``x:D``."""
    controls = set(control_cases)
    return frozenset(name for name in variants if name.split(":")[0] in controls)


def _check_roundoff_classification(per_variant: list, variants, controls: frozenset) -> None:
    """The roundoff-floor rule is generic; the catalogue's control list is a tripwire on it. A variant classified
    differently from the list fails (a real variant silently gated by the floor would weaken the gate; a control
    with a non-roundoff archived error is not the state this rule was written for)."""
    for result, name in zip(per_variant, variants):
        is_floor = result["tier_b_mode"] == "roundoff_floor"
        result["expected_roundoff_control"] = name in controls
        if is_floor != (name in controls):
            result["roundoff_classification_mismatch"] = True
            result["pass"] = False


def compare_operator_terms(*, env, out: dict, paths: dict, campaigns, n: int) -> dict:
    """``{campaign: {"campaign", "status": "ok", "terms": {...}}}`` (the ``reduce_grid`` ``results`` layout).

    ``out`` is the merged host-format term dict (``{"cells": ..., "faces": ..., "p07": ...}``) of the
    campaigns in ``campaigns`` at grid ``n`` over the *full* grid; ``env`` is the campaign
    ``Environment`` of that grid (``t.vol``, ``t.ro``, region masks) and ``paths`` the oracle folders."""
    t = env.t
    owners = len(t.vol)
    cells, faces, p07 = out["cells"], out["faces"], out["p07"]
    campaigns = tuple(campaigns)
    results: dict = {}

    evolution_volume_safe = None
    if "q1_evolution_volume" in cells:
        evolution_volume_safe = np.maximum(_acc(owners, cells["q1_evolution_volume"]).total, 1e-300)

    if "p05" in campaigns:
        from p05_direct_midpoint_global.direct_operator import regional_owner_masks

        with np.load(paths["p05"] / f"N{n}.owner_results.npz", allow_pickle=False) as z:
            saved_centered = z["centered"].copy(); saved_reference = z["reference"].copy()
            owner_volume = z["raw_owner_volume"].copy()
        with np.load(paths["p05"] / "reuse_inputs" / f"N{n}.reuse.npz", allow_pickle=False) as z:
            saved_old_u_minus_a = z["old_U_minus_A"].copy()
        replay_centered = _acc(owners, cells["p05_centered"]).total / t.vol[:, None]
        ijk = np.array(np.unravel_index(np.arange(n ** 3), (n, n, n))).T
        masks, _info = regional_owner_masks(t.ro, ijk, len(t.vol), n)
        archived_error = saved_centered - saved_reference
        terms = {"centered": compare_owner_term("p05.centered", replay_centered, saved_centered,
                                                owner_volume=owner_volume, archived_error=archived_error,
                                                region_masks=masks)}
        saved_upwind = _load_p05_upwind(paths["p05_upwind_chunks"], n)
        ids = np.asarray(faces["p05_live_jump_p07ids"], dtype=np.int64)
        values = P05_PREFIX_JUMP_RECORD_SIGN * np.asarray(faces["p05_live_jump_values"])
        max_pid = int(ids.max()) if len(ids) else -1
        dense_replay = np.full((max_pid + 1, saved_upwind.shape[1]), np.nan)
        dense_replay[ids] = values
        limit = min(dense_replay.shape[0], saved_upwind.shape[0])
        populated = np.flatnonzero(np.all(np.isfinite(dense_replay[:limit]), axis=1)
                                   & np.all(np.isfinite(saved_upwind[:limit]), axis=1))
        # cancellation term: the jump differences side values of the flux integrals whose owner sums are
        # ``centered * owner_volume``; its roundoff is set by them, over all columns (a constant-field column has a
        # tiny term but the same reconstruction noise), not by the jump (see the module docstring)
        flux_scale = float(np.max(np.abs(saved_centered) * owner_volume[:, None]))
        terms["live_jump_vs_upwind"] = compare_pointwise_only(
            "p05.live_jump_vs_upwind", dense_replay[:limit][populated], saved_upwind[:limit][populated],
            kind="cancellation", constituent_scale=flux_scale)
        owner_live_jump = P05_PREFIX_JUMP_RECORD_SIGN * _acc(owners, faces["p05_live_jump_owner_num"]).total / t.vol[:, None]
        terms["live_jump_vs_old_U_minus_A"] = compare_owner_term(
            "p05.live_jump_vs_old_U_minus_A", owner_live_jump, saved_old_u_minus_a,
            owner_volume=owner_volume, archived_error=archived_error, region_masks=masks,
            # U - A cancels terms of size |centered|; roundoff is set by those, not by the result.
            cap_reference=np.max(np.abs(saved_centered), axis=0, keepdims=True))
        results["p05"] = {"campaign": "p05", "status": "ok", "terms": terms,
                          "antisymmetry_max": float(cells.get("p05_antisymmetry_max", 0.0))}

    region_masks_p06n = None
    if any(name in campaigns for name in ("p05n_frozen", "p05n_upwind", "p06n")):
        import p06n_field_derived_global.core as p06n_core
        region_masks_p06n = p06n_core.regional_masks(t)

    for name, root_key in _P05N_ROOTS:
        if name not in campaigns:
            continue
        root = paths[root_key] if name == "p05n_frozen" else paths[root_key] / "p05n_upwind"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved_raw_N, saved_raw_D, saved_raw_R = z["N"].copy(), z["D"].copy(), z["R"].copy()
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_face_N, saved_face_D = z["N"].copy(), z["D"].copy()
        replay_raw = {suf: _acc(owners, cells[f"{name}_raw_{suf}"]).total / t.vol[:, None] for suf in ("N", "D")}
        archived_error_raw = saved_raw_N - saved_raw_R
        terms = {f"raw_{suf}": compare_owner_term(f"p05n.raw_{suf}", replay_raw[suf], saved, owner_volume=t.vol,
                                                  archived_error=archived_error_raw,
                                                  region_masks=region_masks_p06n)
                 for suf, saved in (("N", saved_raw_N), ("D", saved_raw_D))}
        replay_face_N = P05_PREFIX_JUMP_RECORD_SIGN * _acc(owners, faces[f"{name}_face_N"]).total / t.vol[:, None]
        replay_face_D = P05_PREFIX_JUMP_RECORD_SIGN * _acc(owners, faces[f"{name}_face_D"]).total / t.vol[:, None]
        # cancellation terms (jumps of near-equal side values): conditioning-floor pointwise cap from the
        # owner-level densities of the same operator (raw N / D), over all columns
        side_scale = float(max(np.max(np.abs(saved_raw_N)), np.max(np.abs(saved_raw_D))))
        terms["face_N"] = compare_owner_term("p05n.face_N", replay_face_N, saved_face_N, owner_volume=t.vol,
                                             archived_error=archived_error_raw, region_masks=region_masks_p06n,
                                             kind="cancellation", constituent_scale=side_scale)
        terms["face_D"] = compare_owner_term("p05n.face_D", replay_face_D, saved_face_D, owner_volume=t.vol,
                                             archived_error=archived_error_raw, region_masks=region_masks_p06n,
                                             kind="cancellation", constituent_scale=side_scale)
        results[name] = {"campaign": f"p05n[{'p05n_catalogue.json' if name == 'p05n_frozen' else 'p05n_upwind_catalogue.json'}]",
                         "status": "ok", "terms": terms}

    if "p06n" in campaigns:
        import p06n_field_derived_global.core as p06n_core

        variants = p06n_core.CATALOGUE_TABLES.variant_names
        root = paths["p05n_p06n_upwind"] / "p06n"
        with np.load(root / f"N{n}.raw.npz", allow_pickle=False) as z:
            saved = {label: z[label].copy() for label in ("material", "remainder", "total", "R_total")}
        with np.load(root / f"N{n}.faces.npz", allow_pickle=False) as z:
            saved_correction = z["correction"].copy()
        V = len(variants)
        replay = {label: np.stack([_acc(owners, cells[f"p06n_raw_{label}"][vi]).total / evolution_volume_safe[:, None]
                                   for vi in range(V)])
                  for label in ("material", "remainder", "total")}
        archived_error = saved["total"] - saved["R_total"]
        results_terms = {}
        # the constant-field control variants have an exactly-zero result: their archived error is roundoff and the
        # Tier-B ratio would compare roundoff with roundoff. ``tier_b="auto"`` gates such a variant by the absolute
        # roundoff floor; the scale is the term's own magnitude (largest owner-L2 over the variants).
        controls = _expected_roundoff_variants(variants, p06n_core.CONTROL_CASES)
        for label in ("material", "remainder", "total"):
            scale_l2 = max(owner_weighted_l2(saved[label][vi], t.vol) or 0.0 for vi in range(V))
            per_variant = [compare_owner_term(f"p06n.raw_{label}[{name_}]", replay[label][vi], saved[label][vi],
                                              owner_volume=t.vol, archived_error=archived_error[vi],
                                              region_masks=region_masks_p06n, cap_mode="flat", cap_flat_abs=1e-9,
                                              tier_b="auto", roundoff_scale_l2=scale_l2)
                           for vi, name_ in enumerate(variants)]
            _check_roundoff_classification(per_variant, variants, controls)
            results_terms[f"raw_{label}"] = _summarize_variants(per_variant)
        # the frozen P06N oracle stores the correction undivided (see reduce_grid): compare the raw owner sum
        replay_correction = np.stack([_acc(owners, faces["p06n_faces_correction"][vi]).total for vi in range(V)])
        variant_results = [compare_owner_term(f"p06n.faces_correction[{name_}]", replay_correction[vi],
                                              saved_correction[vi], owner_volume=t.vol,
                                              archived_error=archived_error[vi], region_masks=region_masks_p06n,
                                              cap_mode="flat", cap_flat_abs=1e-9)
                           for vi, name_ in enumerate(variants)]
        results_terms["faces_correction"] = _summarize_variants(variant_results)
        results["p06n"] = {"campaign": "p06n", "status": "ok", "terms": results_terms}

    if "p06_legacy" in campaigns:
        import p06_structured_global.numerics as p06numerics

        with np.load(paths["p06_legacy"] / f"N{n}.prepare.npz", allow_pickle=False) as z:
            region_masks_legacy = {name_[len("region:"):]: z[name_] for name_ in z.files if name_.startswith("region:")}
        with np.load(paths["p06_legacy"] / f"N{n}.npz", allow_pickle=False) as z:
            saved = {k_: z[k_].copy() for k_ in z.files if k_ != "metadata_json"}
        masks = {name_: np.asarray(arr, dtype=bool) for name_, arr in region_masks_legacy.items()}
        terms = {}
        for field_name in p06numerics.FIELD_NAMES:
            replay_centered = {term: _acc(owners, cells["p06legacy_raw_centered"][field_name][term]).total
                                     / evolution_volume_safe[:, None]
                               for term in ("material", "remainder", "total")}
            correction_owner = (_acc(owners, faces["p06legacy_faces_correction"][field_name]).total
                                / evolution_volume_safe[:, None])
            replay_u = {"remainder": replay_centered["remainder"]}
            replay_u["material"] = replay_centered["material"] + correction_owner
            replay_u["total"] = replay_u["material"] + replay_u["remainder"]
            for term in ("material", "remainder", "total"):
                saved_centered = saved[f"candidate:centered:{field_name}:{term}"]
                saved_u = saved[f"candidate:U:{field_name}:{term}"]
                saved_total_R = saved[f"reference_evolution:{field_name}:total"]
                archived_error = saved[f"candidate:centered:{field_name}:total"] - saved_total_R
                terms[f"centered:{field_name}:{term}"] = compare_owner_term(
                    f"p06.centered:{field_name}:{term}", replay_centered[term], saved_centered,
                    owner_volume=t.vol, archived_error=archived_error, region_masks=masks,
                    cap_mode="flat", cap_flat_abs=1e-9)
                terms[f"U:{field_name}:{term}"] = compare_owner_term(
                    f"p06.U:{field_name}:{term}", replay_u[term], saved_u,
                    owner_volume=t.vol, archived_error=archived_error, region_masks=masks,
                    cap_mode="flat", cap_flat_abs=1e-9)
        results["p06_legacy"] = {"campaign": "p06_legacy", "status": "ok", "terms": terms}

    if "p07" in campaigns:
        with np.load(paths["p07"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_action = z["action"].copy(); saved_reference_midpoint = z["reference_midpoint"].copy()
            region_masks_p07 = {name_[len("region_"):]: z[name_] for name_ in z.files if name_.startswith("region_")}
        replay_owner = _acc(owners, p07["p07_global_N"]).total / t.vol[:, None]
        archived_error = saved_action - saved_reference_midpoint
        terms = {"global_N": compare_owner_term("p07.global_N", replay_owner, saved_action, owner_volume=t.vol,
                                                archived_error=archived_error, region_masks=region_masks_p07)}
        results["p07"] = {"campaign": "p07", "status": "ok", "terms": terms}

    if "p07n" in campaigns:
        with np.load(paths["p07n"] / f"N{n}.global.npz", allow_pickle=False) as z:
            saved_N = z["N"].copy(); saved_D = z["D"].copy(); saved_N_minus_O = z["N_minus_O"].copy()
            region_masks_p07n = {name_[len("region_"):]: z[name_] for name_ in z.files if name_.startswith("region_")}
        replay_N = _acc(owners, p07["p07n_global_N"]).total / t.vol[:, None]
        replay_D = _acc(owners, p07["p07n_global_D"]).total / t.vol[:, None]
        archived_error = saved_N_minus_O
        terms = {
            "global_N": compare_owner_term("p07n.global_N", replay_N, saved_N, owner_volume=t.vol,
                                           archived_error=archived_error, region_masks=region_masks_p07n),
            "global_D": compare_owner_term("p07n.global_D", replay_D, saved_D, owner_volume=t.vol,
                                           archived_error=archived_error, region_masks=region_masks_p07n),
        }
        results["p07n"] = {"campaign": "p07n", "status": "ok", "terms": terms}

    return results
