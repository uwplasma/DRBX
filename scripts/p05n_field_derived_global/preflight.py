"""P05N preflight gates: small-scale (the same ~54-owner selection vendored
byte-identically in ``preflight_fixtures/`` -- see
``preflight_fixtures/extract.py`` for its provenance), one process per
resolution (the campaign runs the resolutions concurrently).

Gate (i), the slow per-owner oracle comparison, is a code-equivalence check:
it does not depend on where the campaign runs. It is therefore not part of the
campaign preflight but of the separate ``verify-equivalence`` command
(``run(..., oracle=True)``), run once per code change on the covering subset
``_oracle_subset``. The campaign preflight runs gates (ii) and (iii) only.

Gate (i): batched path == per-owner oracle (operator.owner_centered +
          operator.owner_jump), every frozen pairing, both BC variants,
          atol=rtol=1e-12 (isclose headroom convention).
Gate (ii): all-Dirichlet replay of the accepted P05 8-field catalogue through
           the BATCHED path against the vendored fixture's expected
           centered/jump rows (see ``preflight_fixtures/N{n}.accepted_p05_replay.npz``),
           same tolerance.
Gate (iii): row linearity/zero-data, constant, antisymmetry, zero jump on
            physical-wall/radial-n-1/transverse-last-two-layer faces, Neumann
            condition <= 1e8 -- batched equivalents of the bounded checks.

Reads only package-vendored files and ``input_root`` geometry/reference
inputs -- never any local scratch workspace path; see
``preflight_fixtures/fixtures_manifest.json`` for exactly what was vendored
and from which sources (``preflight_fixtures/extract.py`` is the one
sanctioned exception that reads those sources, and only when run manually).

Import as ``from p05n_field_derived_global import preflight`` (DRBX/scripts on
sys.path, never this package's own directory first -- see operator.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p05n_field_derived_global import core
from p05n_field_derived_global import operator as p05n_operator
from p05_structured_global import numerics as oldnum
from p05_direct_midpoint_global.direct_operator import direct_pair_actions
from drbx.native.fci_perpendicular_face_corrections import p05_scalar_face_jump

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "preflight_fixtures"

TOL = dict(replay_atol=1e-12, replay_rtol=1e-12, equivalence_atol=1e-12, equivalence_rtol=1e-12,
           linearity_atol=1e-9, constant_atol=1e-8, antisymmetry_atol=1e-12, exact_input_atol=1e-12,
           zero_jump_atol=1e-12, condition_max=1e8)


def _encode_faces(n, faces):
    face_ids = np.array([core.encode_face_id(n, key) for key, lo, hi in faces], dtype=np.int64)
    endpoints = np.array([[lo if lo is not None else -1, hi if hi is not None else -1] for _, lo, hi in faces],
                          dtype=np.int64)
    return face_ids, faces, endpoints


def _make_role(ref, period, name, bc, lattice):
    def evaluate(q, _n=name):
        v, g, _ = core.p05n_fields.evaluate(ref, q, _n, period)
        return v, g
    column = core.NAMES.index(name)
    normal_data = (lambda q, _c=column: lattice.normal_data(q)[:, _c]) if bc == "neumann" else None
    return p05n_operator.Role(name=name, bc=bc, evaluate=evaluate, normal_data=normal_data)


def _oracle_subset(selection):
    """Owners that also get the slow per-owner oracle comparison (gate i).

    Covers every radial layer of the wall-adjacent selection (its first
    angular position), a second angular position in the last two layers,
    all seam owners and both controls. The replay and structural gates still
    run on the whole selection through the batched path.
    """
    first_by_layer = {}
    second_by_layer = {}
    for rec in selection["wall_adjacent"]:
        layer = int(rec["labels"][0].split(":")[1][1:])
        if layer not in first_by_layer:
            first_by_layer[layer] = rec["owner"]
        elif layer not in second_by_layer:
            second_by_layer[layer] = rec["owner"]
    last_two = sorted(first_by_layer)[-2:]
    owners = list(first_by_layer.values()) + [second_by_layer[layer] for layer in last_two if layer in second_by_layer]
    owners += [rec["owner"] for rec in selection["seam"]] + [rec["owner"] for rec in selection["controls"]]
    return [int(o) for o in dict.fromkeys(owners)]


def _headroom(actual, expected, atol, rtol):
    return abs(actual - expected) / (atol + rtol * abs(expected))


def run(input_root, n, output_dir, oracle=False):
    input_root = Path(input_root); output_dir = Path(output_dir)
    # The localized reference sidecar is written by campaign.py's verify(),
    # which always runs (and is re-checked for staleness) before preflight_stage.
    t, ref = core.load(input_root, output_dir / "reference_sidecar.json", n)
    S = core.StructuredReconstruction(t)
    ctx = core.context(t)
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period)
    normal_coefficients = lattice.normal  # one bound object: patch caches key on its id

    values = np.column_stack([core.p05n_fields.evaluate(ref, t.pts, nm, period)[0] for nm in core.NAMES])
    owner_values = core.live_observations(t, values)
    live_values = {nm: owner_values[:, i] for i, nm in enumerate(core.NAMES)}

    roles = {r: _make_role(ref, period, *core.ROLES[r], lattice) for r in core.ROLE_NAMES}

    selection = json.loads((FIXTURES / f"N{n}.selection.json").read_text())
    all_owners = [int(rec["owner"]) for rec in selection["all_owners"]]
    oracle_owners = set(_oracle_subset(selection)) if oracle else set()

    patch_cache_oracle = {}
    patch_cache_batched = {}

    equivalence_max = 0.0
    condition_max = 0.0
    residual_max = 0.0
    antisymmetry_max = 0.0
    exact_input_defect_max = 0.0
    constant_action_max = 0.0
    zero_jump_max = {"physical_wall": 0.0, "radial_n_minus_1": 0.0, "transverse_last_two_layers": 0.0}
    per_owner = []

    for owner in all_owners:
        raw_ids, points, keys = p05n_operator.owner_raw(t, owner)
        rc = core.raw_chunk(t, S, ref, ctx, raw_ids, owner_values, normal_coefficients, patch_cache_batched, period,
                            normal_data_override=lattice.normal_data)
        vol = t.vol[owner]
        N_b = (rc["raw_volume"][:, None] * rc["N"]).sum(axis=0) / vol
        D_b = (rc["raw_volume"][:, None] * rc["D"]).sum(axis=0) / vol
        R_b = (rc["raw_volume"][:, None] * rc["R"]).sum(axis=0) / vol
        condition_max = max(condition_max, rc["condition_max"])
        residual_max = max(residual_max, rc["residual_max"])
        antisymmetry_max = max(antisymmetry_max, rc["antisymmetry"])
        exact_input_defect_max = max(exact_input_defect_max, float(np.max(rc["exact_input_defect"])))

        faces = p05n_operator.owner_boundary_faces(t, owner)
        fc = None
        if faces:
            face_ids, face_tuples, endpoints = _encode_faces(t.n, faces)
            fc = core.face_chunk(t, S, ref, ctx, face_ids, endpoints, owner_values, normal_coefficients,
                                  patch_cache_batched, period, normal_data_override=lattice.normal_data)
            condition_max = max(condition_max, fc["condition_max"])
            residual_max = max(residual_max, fc["residual_max"])

        oracle = owner in oracle_owners
        for pidx, pname in enumerate(core.PAIR_NAMES):
            if pname in core.CONSTANT_PAIRS:
                constant_action_max = max(constant_action_max, abs(N_b[pidx]), abs(D_b[pidx]))
            if fc is not None:
                for row in range(len(face_tuples)):
                    kind = core._face_kind(t.n, face_tuples[row][0])
                    if kind is not None:
                        zero_jump_max[kind] = max(zero_jump_max[kind], abs(fc["N"][row, pidx]), abs(fc["D"][row, pidx]))
            if not oracle:
                continue
            # Gate (i): the per-owner oracle, on the bounded oracle subset only.
            gen_role, tra_role = core.PAIRINGS[pname]
            gen, tra = roles[gen_role], roles[tra_role]
            info = p05n_operator.owner_centered(t, S, ref, owner, gen, tra, live_values, normal_coefficients,
                                                 patch_cache_oracle, ctx)
            dgen_role, dtra_role = core.DIRICHLET_COUNTERPART[gen_role], core.DIRICHLET_COUNTERPART[tra_role]
            dgen, dtra = roles[dgen_role], roles[dtra_role]
            dinfo = p05n_operator.owner_centered(t, S, ref, owner, dgen, dtra, live_values, normal_coefficients,
                                                  patch_cache_oracle, ctx)
            equivalence_max = max(equivalence_max,
                                   _headroom(N_b[pidx], info["centered"], TOL["equivalence_atol"], TOL["equivalence_rtol"]),
                                   _headroom(D_b[pidx], dinfo["centered"], TOL["equivalence_atol"], TOL["equivalence_rtol"]),
                                   _headroom(R_b[pidx], info["R"], TOL["equivalence_atol"], TOL["equivalence_rtol"]))
            if fc is not None:
                jump_oracle, zero_here = p05n_operator.owner_jump(t, S, ref, owner, gen, tra, live_values,
                                                                   normal_coefficients, patch_cache_oracle, ctx)
                djump_oracle, _ = p05n_operator.owner_jump(t, S, ref, owner, dgen, dtra, live_values,
                                                            normal_coefficients, patch_cache_oracle, ctx)
                for kind, value in zero_here.items():
                    zero_jump_max[kind] = max(zero_jump_max[kind], value)
                total_N = 0.0; total_D = 0.0
                for ridx, (key, lo, hi) in enumerate(face_tuples):
                    sign = 1.0 if lo == owner else -1.0
                    total_N += sign * fc["N"][ridx, pidx]
                    total_D += sign * fc["D"][ridx, pidx]
                total_N /= vol; total_D /= vol
                equivalence_max = max(equivalence_max,
                                       _headroom(total_N, jump_oracle, TOL["equivalence_atol"], TOL["equivalence_rtol"]),
                                       _headroom(total_D, djump_oracle, TOL["equivalence_atol"], TOL["equivalence_rtol"]))
        per_owner.append(dict(owner=owner, role=next((r["role"] for r in selection["all_owners"] if r["owner"] == owner), None)))

    equivalence_pass = equivalence_max <= 1.0

    # ---- Gate (ii): all-Dirichlet replay of the accepted P05 8-field catalogue,
    # against a vendored fixture (preflight_fixtures/extract.py) holding only
    # the 54 selected owners' expected centered/jump rows and the accepted
    # P05 observations restricted to the donor-owner closure those rows touch.
    with np.load(FIXTURES / f"N{n}.accepted_p05_replay.npz", allow_pickle=False) as data:
        fixture = {name: data[name] for name in data.files}
    if int(fixture["owner_count"]) != len(t.vol):
        raise ValueError(f"N{n} owner count changed against the vendored replay fixture")
    if not np.array_equal(fixture["selected_owners"], np.asarray(all_owners, dtype=np.int64)):
        raise ValueError(f"N{n} selection changed against the vendored replay fixture")
    n_old_fields = fixture["donor_values"].shape[1]
    # NaN-filled outside the precomputed donor closure: any donor id the
    # replay touches that extract.py did not anticipate propagates NaN into
    # the action, and the explicit finite check below fails loudly rather
    # than silently comparing garbage against the expected rows.
    old_owner_values = np.full((len(t.vol), n_old_fields), np.nan)
    old_owner_values[fixture["donor_owner_ids"]] = fixture["donor_values"]

    def old_dirichlet_trace_fn(q):
        return oldnum.boundary_trace(ref, q)

    def old_normal_data_fn(q):
        return np.zeros((len(q), old_owner_values.shape[1]))

    replay_headroom_max = 0.0
    for row_idx, owner in enumerate(all_owners):
        raw_ids, points, keys = p05n_operator.owner_raw(t, owner)
        cellvals = core.batched_cell_values(t, S, old_owner_values, keys, points, normal_coefficients=normal_coefficients,
                                             ctx=ctx, patch_cache=patch_cache_batched,
                                             dirichlet_trace_fn=old_dirichlet_trace_fn, normal_data_fn=old_normal_data_fn)
        grad = cellvals["dirichlet"][1]
        metric = ref._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jac = np.abs(metric["J"])
        actions, _ = direct_pair_actions(h, jac, grad, oldnum.PAIRS)
        vol_pts = t.rv[raw_ids]
        centered = (vol_pts[:, None] * actions).sum(axis=0) / t.vol[owner]

        faces = p05n_operator.owner_boundary_faces(t, owner)
        jump = np.zeros(len(oldnum.PAIRS))
        for key, lo, hi in faces:
            axis = key[0]
            fpoints, weights = core.pk.num.quadrature(t.faces, np.array([key]), 3, face=True)
            fpoints = fpoints[0]; weights = weights[0]
            common = core.batched_face_common_gradient(t, S, old_owner_values, key, fpoints,
                                                         normal_coefficients=normal_coefficients, ctx=ctx,
                                                         patch_cache=patch_cache_batched,
                                                         dirichlet_trace_fn=old_dirichlet_trace_fn,
                                                         normal_data_fn=old_normal_data_fn)
            side = core.batched_side_values(t, S, old_owner_values, key, fpoints,
                                             normal_coefficients=normal_coefficients, ctx=ctx,
                                             patch_cache=patch_cache_batched,
                                             dirichlet_trace_fn=old_dirichlet_trace_fn,
                                             normal_data_fn=old_normal_data_fn)
            fmetric = ref._metric(fpoints)
            fh = fmetric["bcov"] / fmetric["B"][:, None]
            face_jump = np.asarray(p05_scalar_face_jump(common["dirichlet"][None], side["lower"]["dirichlet"][None],
                                                         side["upper"]["dirichlet"][None], fh[None], weights[None],
                                                         np.array([axis]), np.array(oldnum.PAIRS)))[0]
            sign = 1.0 if lo == owner else -1.0
            jump += sign * face_jump
        jump /= t.vol[owner]
        if not np.isfinite(centered).all() or not np.isfinite(jump).all():
            raise ValueError(f"N{n} owner {owner}: replay touched a donor owner outside the vendored "
                              "fixture's precomputed closure (NaN propagated) -- fixture is stale, "
                              "re-run preflight_fixtures/extract.py")

        for pair_idx in range(len(oldnum.PAIRS)):
            expected_centered = float(fixture["expected_centered"][row_idx, pair_idx])
            expected_jump = float(fixture["expected_jump"][row_idx, pair_idx])
            replay_headroom_max = max(replay_headroom_max,
                                       _headroom(centered[pair_idx], expected_centered, TOL["replay_atol"], TOL["replay_rtol"]),
                                       _headroom(jump[pair_idx], expected_jump, TOL["replay_atol"], TOL["replay_rtol"]))
    replay_pass = replay_headroom_max <= 1.0

    # ---- Gate (iii, linearity sample): batched cell-value linearity in (owner values, g_N). ----
    zero_col = np.zeros_like(owner_values)
    linearity_max = 0.0
    sample = all_owners[: min(5, len(all_owners))]

    def zero_normal_data(q):
        return np.zeros((len(q), owner_values.shape[1]))

    for owner in sample:
        raw_ids, points, keys = p05n_operator.owner_raw(t, owner)

        def trace_fn(q):
            return core.dirichlet_trace_all(ref, q, period)

        nd_fn = lattice.normal_data

        both = core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                         ctx=ctx, patch_cache=patch_cache_batched, dirichlet_trace_fn=trace_fn,
                                         normal_data_fn=nd_fn)
        data_only = core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                              ctx=ctx, patch_cache=patch_cache_batched, dirichlet_trace_fn=trace_fn,
                                              normal_data_fn=zero_normal_data)
        bc_only = core.batched_cell_values(t, S, zero_col, keys, points, normal_coefficients=normal_coefficients,
                                            ctx=ctx, patch_cache=patch_cache_batched, dirichlet_trace_fn=trace_fn,
                                            normal_data_fn=nd_fn)
        v_both, g_both = both["neumann"]
        v_data, g_data = data_only["neumann"]
        v_bc, g_bc = bc_only["neumann"]
        linearity_max = max(linearity_max, float(np.max(np.abs(v_both - (v_data + v_bc)))),
                             float(np.max(np.abs(g_both - (g_data + g_bc)))))

    result = dict(
        n=n, selection_owners=len(all_owners), oracle_owners=sorted(oracle_owners),
        check2_replay=dict(headroom_max=replay_headroom_max, pass_=bool(replay_pass),
                            tolerance=dict(atol=TOL["replay_atol"], rtol=TOL["replay_rtol"])),
        check3_linearity=dict(max_defect=linearity_max, pass_=bool(linearity_max <= TOL["linearity_atol"])),
        check4_constant=dict(action_max=constant_action_max,
                              pass_=bool(constant_action_max <= TOL["constant_atol"])),
        check5_antisymmetry=dict(max=antisymmetry_max, pass_=bool(antisymmetry_max <= TOL["antisymmetry_atol"])),
        check6_exact_input=dict(max_defect=exact_input_defect_max,
                                 pass_=bool(exact_input_defect_max <= TOL["exact_input_atol"])),
        check7_zero_jump=dict(max_by_kind=zero_jump_max,
                               pass_=bool(all(v <= TOL["zero_jump_atol"] for v in zero_jump_max.values()))),
        check8_condition=dict(condition_max=condition_max, residual_max=residual_max,
                               pass_=bool(condition_max <= TOL["condition_max"])),
    )
    if oracle:
        result["check1_equivalence"] = dict(headroom_max=equivalence_max, pass_=bool(equivalence_pass),
                                            tolerance=dict(atol=TOL["equivalence_atol"], rtol=TOL["equivalence_rtol"]))
    result["all_pass"] = bool(all(result[k]["pass_"] for k in result if k.startswith("check")))
    return result
