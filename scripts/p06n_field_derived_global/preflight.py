"""P06N preflight gates: small-scale (a few tens of owners per grid), one
process per resolution (the campaign runs the resolutions concurrently).

Gate (a), replay: the accepted P06 curvature operator (all-Dirichlet, the
four accepted ``p06_structured_global.numerics.FIELD_NAMES`` states), run
through the campaign's own batched kernels (``core.raw_chunk``/``face_chunk``
with accepted-state ``FieldTables``) in accepted-census mode (the periodic
seam duplicate included, as the accepted campaign counted it). It is checked
against the vendored accepted artifact (``preflight_fixtures/N{n}.replay.npz``,
built by ``preflight_fixtures/extract.py``), which also checks that the run
environment reproduces the accepted numbers. Tolerance atol=1e-9, rtol=1e-12:
it absorbs the measured ~1.2-2.7e-10 absolute cross-platform gap in the
accepted campaign's finite-difference curvature (see README.md).

The one-off ``verify-equivalence`` command (``run(..., oracle=True)``) instead
compares the batched kernels with the per-owner oracle
(``operator.owner_q1``/``owner_q3``) for every catalogue variant on a covering
owner subset, atol=rtol=1e-12. It runs once per code change, not per campaign.

Gate (b), structural: batched-path (``core.raw_chunk``/``core.face_chunk``,
the campaign's own deduplicated census) row linearity/zero-data, constant
controls, the physical-wall characteristic correction (zero by construction:
the recovered-trace wall contract), a zero jump on the wall-adjacent families
(``radial_n_minus_1``/``transverse_last_two_layers``, where both sides share
the identical row) checked against a *nonconstant* case (a zero-jump check
against only the constant control would trivially pass and catch nothing),
omega-independence (the frozen oracle's ``check_omega_independence``, a
property of the shared algebra, not of which code path evaluates it), the
exact-input (O==R) defect, and the Neumann row condition number.

Gate (c), seam: at one theta-seam and one eta-seam owner (from the vendored
P06N bounded selection), the accepted-census (``dedupe=False``) face
correction minus the deduplicated-census (``dedupe=True``) face correction
equals exactly the seam face's own (duplicate) contribution.

Reads only package-vendored files and ``input_root`` geometry/reference
inputs -- never any local scratch workspace path; see
``preflight_fixtures/fixtures_manifest.json`` for exactly what was vendored
and from which sources (``preflight_fixtures/extract.py`` is the one
sanctioned exception, run manually).

Import as ``from p06n_field_derived_global import preflight`` (DRBX/scripts on
sys.path, never this package's own directory first -- see operator.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from p06n_field_derived_global import core
from p06n_field_derived_global import operator as p06n_operator
from p06n_field_derived_global.operator import Case, Role, make_role, owner_q1, owner_q3, check_omega_independence
import p06_structured_global.numerics as p06numerics

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "preflight_fixtures"

TOL = dict(replay_atol=1e-9, replay_rtol=1e-12,
           linearity_atol=1e-9, constant_atol=1e-8,
           zero_wall_atol=1e-12, zero_jump_atol=1e-12,
           omega_independence_atol=1e-10, exact_input_atol=1e-12,
           condition_max=1e8, seam_atol=1e-9)


def _headroom(actual, expected, atol, rtol):
    return abs(actual - expected) / (atol + rtol * abs(expected))


# ---------------------------------------------------------------------------
# Gate (a): all-Dirichlet replay of the accepted P06 campaign.
# ---------------------------------------------------------------------------
def _accepted_role(name, idx, ref, time_value):
    def evaluate(q, _n=name, _i=idx):
        v, g = p06numerics._evaluate_fields(_n, ref, q, time_value)
        return v[_i], g[_i]
    return Role(name=f"accepted:{name}:{idx}", bc="dirichlet", evaluate=evaluate)


def _accepted_case(name, ref, time_value):
    roles = [_accepted_role(name, i, ref, time_value) for i in range(5)]
    return Case(n=roles[0], Te=roles[1], Ti=roles[2], omega=roles[3], phi=roles[4])


def _replay_gate(t, S, ref, ctx, normal_coefficients, patch_cache, n):
    with np.load(FIXTURES / f"N{n}.replay.npz", allow_pickle=False) as data:
        fixture = {k: data[k] for k in data.files}
    if int(fixture["owner_count"]) != len(t.vol):
        raise ValueError(f"N{n} owner count changed against the vendored replay fixture")
    time_value = float(fixture["time_value"])
    owner_ids = [int(o) for o in fixture["owner_ids"]]
    donor_owner_ids = fixture["donor_owner_ids"]
    headroom_max = 0.0
    n_records = 0
    for state_idx, name in enumerate(p06numerics.FIELD_NAMES):
        case = _accepted_case(name, ref, time_value)
        owner_values = np.full((len(t.vol), 5), np.nan)
        owner_values[donor_owner_ids] = fixture["donor_values"][state_idx].T
        for row, owner in enumerate(owner_ids):
            q1 = owner_q1(t, S, ref, owner, case, owner_values, normal_coefficients=normal_coefficients,
                          context=ctx, patch_cache=patch_cache)
            q3 = owner_q3(t, S, ref, owner, case, owner_values, normal_coefficients=normal_coefficients,
                          context=ctx, patch_cache=patch_cache, dedupe=False)
            evolution_volume = float(q1["evolution_volume"])
            u_material = q1["N"]["material"] + q3["material_correction"] / evolution_volume
            u_total = u_material + q1["N"]["remainder"]
            centered = np.stack([q1["N"]["material"], q1["N"]["remainder"], q1["N"]["total"]])
            upwind = np.stack([u_material, q1["N"]["remainder"], u_total])
            if not np.isfinite(centered).all() or not np.isfinite(upwind).all():
                raise ValueError(f"N{n} owner {owner} state {name}: replay touched a donor owner outside the "
                                  "vendored fixture's precomputed closure (NaN propagated) -- fixture is stale, "
                                  "re-run preflight_fixtures/extract.py")
            expected_centered = fixture["candidate_centered"][state_idx, :, row, :]
            expected_upwind = fixture["candidate_U"][state_idx, :, row, :]
            headroom_max = max(headroom_max,
                                float(np.max(np.abs(centered - expected_centered) /
                                             (TOL["replay_atol"] + TOL["replay_rtol"] * np.abs(expected_centered)))),
                                float(np.max(np.abs(upwind - expected_upwind) /
                                             (TOL["replay_atol"] + TOL["replay_rtol"] * np.abs(expected_upwind)))))
            headroom_max = max(headroom_max, _headroom(evolution_volume, float(fixture["evolution_volume"][row]),
                                                        TOL["replay_atol"], TOL["replay_rtol"]))
            n_records += 1
    return dict(headroom_max=headroom_max, pass_=bool(headroom_max <= 1.0), n_records=n_records,
                tolerance=dict(atol=TOL["replay_atol"], rtol=TOL["replay_rtol"]))


def _accepted_tables(time_value):
    """Batched-kernel field tables for the four accepted P06 states (all Dirichlet)."""
    spec = {f"accepted:{state}": tuple((f"acc:{state}:{i}", "dirichlet") for i in range(5))
            for state in p06numerics.FIELD_NAMES}

    def evaluate(ref, q, name, period):
        _, state, component = name.split(":")
        v, g = p06numerics._evaluate_fields(state, ref, np.asarray(q, dtype=np.float64), time_value)
        return v[int(component)], g[int(component)], None

    return core.FieldTables(spec, evaluate)


def _owner_reduce(rc, owners, term_keys=("material", "remainder", "total")):
    """Evolution-weighted owner averages of a raw_chunk result: {owner: (V, 3, 4)}, {owner: volume}."""
    out = {}; volume = {}
    for owner in owners:
        m = rc["owner_ids"] == owner
        w = rc["evolution_weight"][m]
        volume[owner] = float(np.sum(w))
        out[owner] = np.stack([np.sum(w[None, :, None] * rc[k][:, m], axis=1) / volume[owner] for k in term_keys], axis=1)
    return out, volume


def _owner_correction(fc, owner):
    """(V, 4) face correction to ``owner``: every face side whose owner matches, both sides independently."""
    return (fc["correction_lo"][:, fc["lo"] == owner].sum(axis=1) + fc["correction_hi"][:, fc["hi"] == owner].sum(axis=1))


def _batched_replay_gate(t, S, ref, ctx, n):
    """Gate (a): the accepted P06 states through the campaign's own batched kernels.

    Accepted-census mode (the periodic seam duplicate included, as the accepted
    run counted it), against the vendored accepted artifact. This also checks
    that the run environment reproduces the accepted numbers.
    """
    with np.load(FIXTURES / f"N{n}.replay.npz", allow_pickle=False) as data:
        fixture = {k: data[k] for k in data.files}
    if int(fixture["owner_count"]) != len(t.vol):
        raise ValueError(f"N{n} owner count changed against the vendored replay fixture")
    time_value = float(fixture["time_value"])
    owners = [int(o) for o in fixture["owner_ids"]]
    tables = _accepted_tables(time_value)
    owner_values = np.full((len(t.vol), len(tables.names)), np.nan)
    for column, name in enumerate(tables.names):
        _, state, component = name.split(":")
        owner_values[fixture["donor_owner_ids"], column] = fixture["donor_values"][p06numerics.FIELD_NAMES.index(state)][int(component)]
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period, names=core.NAMES[:1])  # normals only; every accepted state is Dirichlet
    patch_cache = {}
    raw_ids = np.flatnonzero(np.isin(t.ro, owners))
    rc = core.raw_chunk(t, S, ref, ctx, raw_ids, owner_values, lattice.normal, patch_cache, period, tables=tables)
    centered, volume = _owner_reduce(rc, owners)
    keys, _lo, _hi = core.owner_face_union(t, owners, dedupe=False)
    fc = core.face_chunk(t, S, ref, ctx, keys, owner_values, lattice.normal, patch_cache, period,
                         dedupe=False, tables=tables)
    headroom_max = 0.0; records = 0
    for row, owner in enumerate(owners):
        correction = _owner_correction(fc, owner)
        for vi, variant in enumerate(tables.variant_names):
            state_idx = p06numerics.FIELD_NAMES.index(variant.split(":", 1)[1])
            c = centered[owner][vi]
            u = c.copy(); u[0] = c[0] + correction[vi] / volume[owner]; u[2] = u[0] + c[1]
            if not (np.isfinite(c).all() and np.isfinite(u).all()):
                raise ValueError(f"N{n} owner {owner}: replay touched a donor outside the vendored closure")
            for actual, expected in ((c, fixture["candidate_centered"][state_idx, :, row, :]),
                                     (u, fixture["candidate_U"][state_idx, :, row, :])):
                headroom_max = max(headroom_max, float(np.max(np.abs(actual - expected) /
                                                              (TOL["replay_atol"] + TOL["replay_rtol"] * np.abs(expected)))))
            records += 1
        headroom_max = max(headroom_max, _headroom(volume[owner], float(fixture["evolution_volume"][row]),
                                                    TOL["replay_atol"], TOL["replay_rtol"]))
    return dict(headroom_max=headroom_max, pass_=bool(headroom_max <= 1.0), n_records=records,
                path="batched core.raw_chunk/face_chunk, accepted census",
                tolerance=dict(atol=TOL["replay_atol"], rtol=TOL["replay_rtol"]))


def _equivalence_subset(selection):
    """Covering owner subset for the one-off batched-vs-oracle check."""
    first, second = {}, {}
    for rec in selection["wall_adjacent"]:
        layer = int(rec["labels"][0].split(":")[1][1:])
        (first if layer not in first else second).setdefault(layer, int(rec["owner"]))
    last_two = sorted(first)[-2:]
    owners = list(first.values()) + [second[l] for l in last_two if l in second]
    owners += [int(r["owner"]) for r in selection.get("seam", [])] + [int(r["owner"]) for r in selection.get("controls", [])]
    owners += [int(r["owner"]) for r in selection.get("hotspots", [])]
    return list(dict.fromkeys(owners))


def _equivalence_gate(t, S, ref, ctx, n):
    """One-off: batched kernels vs the per-owner oracle, every catalogue variant, deduplicated census."""
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period)
    owner_values = core.observations_all(t, ref, period)
    patch_batched = {}; patch_oracle = {}
    selection = json.loads((FIXTURES / f"N{n}.selection.json").read_text())
    owners = _equivalence_subset(selection)
    headroom_max = 0.0; records = 0
    atol = rtol = 1e-12
    for owner in owners:
        raw_ids = np.flatnonzero(t.ro == owner)
        rc = core.raw_chunk(t, S, ref, ctx, raw_ids, owner_values, lattice.normal, patch_batched, period,
                            normal_data_override=lattice.normal_data)
        batched, volume = _owner_reduce(rc, [owner])
        exact, _ = _owner_reduce(rc, [owner], ("R_material", "R_remainder", "R_total"))
        keys, _lo, _hi = core.owner_face_union(t, [owner], dedupe=True)
        correction = (_owner_correction(core.face_chunk(t, S, ref, ctx, keys, owner_values, lattice.normal, patch_batched,
                                                        period, normal_data_override=lattice.normal_data, dedupe=True), owner)
                      if len(keys) else np.zeros((len(core.VARIANT_NAMES), 4)))
        for vi, variant in enumerate(core.VARIANT_NAMES):
            roles = [_role(field, bc, ref, period, lattice) for field, bc in core.VARIANT_SPEC[variant]]
            case = Case(*roles)
            q1 = owner_q1(t, S, ref, owner, case, owner_values_for_case(owner_values, variant),
                          normal_coefficients=lattice.normal, context=ctx, patch_cache=patch_oracle)
            q3 = owner_q3(t, S, ref, owner, case, owner_values_for_case(owner_values, variant),
                          normal_coefficients=lattice.normal, context=ctx, patch_cache=patch_oracle, dedupe=True)
            oracle_c = np.stack([q1["N"][k] for k in ("material", "remainder", "total")])
            oracle_r = np.stack([q1["R"][k] for k in ("material", "remainder", "total")])
            pairs = ((batched[owner][vi], oracle_c), (exact[owner][vi], oracle_r),
                     (correction[vi], q3["material_correction"]), (np.array([volume[owner]]), np.array([q1["evolution_volume"]])))
            for actual, expected in pairs:
                headroom_max = max(headroom_max, float(np.max(np.abs(actual - expected) / (atol + rtol * np.abs(expected)))))
            records += 1
    return dict(headroom_max=headroom_max, pass_=bool(headroom_max <= 1.0), n_records=records, owners=owners,
                tolerance=dict(atol=atol, rtol=rtol))


def owner_values_for_case(owner_values, variant):
    """The oracle reads one column per state slot, in state order."""
    return owner_values[:, core.VARIANT_FIELD_INDEX[variant]]


# ---------------------------------------------------------------------------
# Gate (b): structural checks on the batched path.
# ---------------------------------------------------------------------------
def _role(field, bc, ref, period, lattice):
    normal_data = (lambda q, f=field: lattice.normal_data(q)[:, core.NAME_INDEX[f]]) if bc == "neumann" else None
    from p06n_field_derived_global import fields as p06n_fields
    return make_role(p06n_fields, ref, period, field, bc, normal_data=normal_data)


def _structural_gate(t, S, ref, ctx, n):
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period)
    normal_coefficients = lattice.normal
    owner_values = core.observations_all(t, ref, period)
    patch_cache = {}

    selection = json.loads((FIXTURES / f"N{n}.selection.json").read_text())
    all_owners = [int(rec["owner"]) for rec in selection["all_owners"]]

    condition_max = 0.0; residual_max = 0.0; closure_max = 0.0
    exact_input_defect_max = 0.0
    constant_action_max = 0.0
    zero_face_max: dict[str, float] = {}
    wall_exterior_defect_max = 0.0
    wall_faces_seen = 0

    control_variants = [core.VARIANT_NAMES.index(c) for c in core.CONTROL_CASES] + \
                        [core.VARIANT_NAMES.index(f"{c}:D") for c in core.CONTROL_CASES]

    for owner in all_owners:
        raw_ids = np.flatnonzero(t.ro == owner)
        rc = core.raw_chunk(t, S, ref, ctx, raw_ids, owner_values, normal_coefficients, patch_cache, period,
                             normal_data_override=lattice.normal_data)
        condition_max = max(condition_max, rc["condition_max"])
        residual_max = max(residual_max, rc["residual_max"])
        closure_max = max(closure_max, rc["closure_max"])
        exact_input_defect_max = max(exact_input_defect_max, float(np.max(rc["exact_input_defect"])))
        for vi in control_variants:
            constant_action_max = max(constant_action_max, float(np.max(np.abs(rc["total"][vi]))))

        keys, lo, hi = core.owner_face_union(t, [owner], dedupe=True)
        if len(keys):
            fc = core.face_chunk(t, S, ref, ctx, keys, owner_values, normal_coefficients, patch_cache, period,
                                  normal_data_override=lattice.normal_data, dedupe=True)
            condition_max = max(condition_max, fc["condition_max"])
            residual_max = max(residual_max, fc["residual_max"])
            wall_exterior_defect_max = max(wall_exterior_defect_max, fc["wall_exterior_defect_max"])
            wall_faces_seen += fc["wall_faces_seen"]
            for kind, value in fc["zero_face_max"].items():
                zero_face_max[kind] = max(zero_face_max.get(kind, 0.0), value)
            for vi in control_variants:
                own_lo = fc["correction_lo"][vi][fc["lo"] == owner].sum(axis=0)
                own_hi = fc["correction_hi"][vi][fc["hi"] == owner].sum(axis=0)
                constant_action_max = max(constant_action_max, float(np.max(np.abs(own_lo + own_hi))))

    # ---- Row linearity, on a small near-wall/seam sample. ----
    sample = [r for r in all_owners if any(r == int(rec["owner"]) for rec in
              selection["wall_adjacent"][:3] + selection["seam"][:2])]
    if not sample:
        sample = all_owners[:5]
    linearity_max = 0.0
    zero_col = np.zeros_like(owner_values)

    def zero_normal_data(q):
        return np.zeros((len(q), owner_values.shape[1]))

    from p05n_field_derived_global import core as p05n_core

    def trace_fn(q):
        return core.dirichlet_trace_all(ref, q, period)

    for owner in sample:
        raw_ids = np.flatnonzero(t.ro == owner)
        keys = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
        points = t.pts[raw_ids]
        both = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                              ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                              normal_data_fn=lattice.normal_data, radial_degree=3)
        data_only = p05n_core.batched_cell_values(t, S, owner_values, keys, points, normal_coefficients=normal_coefficients,
                                                   ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                                   normal_data_fn=zero_normal_data, radial_degree=3)
        bc_only = p05n_core.batched_cell_values(t, S, zero_col, keys, points, normal_coefficients=normal_coefficients,
                                                 ctx=ctx, patch_cache=patch_cache, dirichlet_trace_fn=trace_fn,
                                                 normal_data_fn=lattice.normal_data, radial_degree=3)
        v_both, g_both = both["neumann"]; v_data, g_data = data_only["neumann"]; v_bc, g_bc = bc_only["neumann"]
        linearity_max = max(linearity_max, float(np.max(np.abs(v_both - (v_data + v_bc)))),
                             float(np.max(np.abs(g_both - (g_data + g_bc)))))

    # ---- Omega independence (algebraic property; oracle-evaluated). ----
    probe_owner = int((selection["wall_adjacent"] or selection["hotspots"])[0]["owner"])
    base_spec = core.CASES["main_phi_neumann"]
    base_slots = dict(zip(core.STATE_SLOTS, base_spec))
    base_case = Case(**{s: _role(f, bc, ref, period, lattice) for s, (f, bc) in base_slots.items()})
    oracle_owner_values = p06n_operator.owner_values_matrix(t, ref, period, base_case.roles())
    alternate_omega = _role("rich_a_minus1", "dirichlet", ref, period, lattice)
    omega_defect = check_omega_independence(t, S, ref, probe_owner, base_case, oracle_owner_values,
                                             normal_coefficients=normal_coefficients, context=ctx,
                                             patch_cache=patch_cache, alternate_omega=alternate_omega)

    return dict(
        check_linearity=dict(max_defect=linearity_max, pass_=bool(linearity_max <= TOL["linearity_atol"])),
        check_constant=dict(action_max=constant_action_max, pass_=bool(constant_action_max <= TOL["constant_atol"])),
        check_wall_exterior=dict(max_defect=wall_exterior_defect_max, wall_faces_seen=wall_faces_seen,
                                  pass_=bool(wall_exterior_defect_max <= TOL["zero_wall_atol"])),
        check_zero_jump=dict(max_by_kind=zero_face_max,
                              pass_=bool(all(v <= TOL["zero_jump_atol"] for v in zero_face_max.values()))),
        check_omega_independence=dict(max_defect=omega_defect,
                                       pass_=bool(omega_defect <= TOL["omega_independence_atol"])),
        check_exact_input=dict(max_defect=exact_input_defect_max,
                                pass_=bool(exact_input_defect_max <= TOL["exact_input_atol"])),
        check_condition=dict(condition_max=condition_max, residual_max=residual_max,
                              pass_=bool(condition_max <= TOL["condition_max"])),
        check_closure=dict(max_defect=closure_max),
        selection_owners=len(all_owners),
    )


# ---------------------------------------------------------------------------
# Gate (c): seam census check.
# ---------------------------------------------------------------------------
def _seam_gate(t, S, ref, ctx, n):
    period = t.g.eta_period
    lattice = core.WallLattice(t, ref, period)
    normal_coefficients = lattice.normal
    owner_values = core.observations_all(t, ref, period)
    patch_cache = {}
    selection = json.loads((FIXTURES / f"N{n}.selection.json").read_text())

    def pick(prefix):
        """A seam owner whose OWN raw-cell traversal actually visits the
        periodic-duplicate face slot n (i.e. one with a member at the
        n-1 angular index, not the 0 index the vendored selection happens
        to label -- see owner_all_faces's docstring: only a raw cell's own
        (j, j+1) upper face can equal slot n)."""
        for rec in selection["seam"]:
            if any(label.startswith(prefix) for label in rec["labels"]):
                owner = int(rec["owner"])
                raw_ids = np.flatnonzero(t.ro == owner)
                ijk = np.array(np.unravel_index(raw_ids, (t.n,) * 3)).T
                axis_col = 1 if prefix == "seam_theta" else 2
                r0 = int(ijk[0, 0])
                other_idx = int(ijk[0, 2 if axis_col == 1 else 1])
                target = t.ro.reshape((t.n,) * 3)[(r0, t.n - 1, other_idx) if axis_col == 1
                                                   else (r0, other_idx, t.n - 1)]
                return int(target)
        return None

    theta_owner = pick("seam_theta")
    eta_owner = pick("seam_eta")
    results = {}
    for label, owner in (("theta", theta_owner), ("eta", eta_owner)):
        if owner is None:
            results[label] = dict(found=False)
            continue
        faces_accepted = p06n_operator.owner_all_faces(t, owner, dedupe=False)
        faces_dedup = p06n_operator.owner_all_faces(t, owner, dedupe=True)
        dedup_keys = {key for key, _lo, _hi in faces_dedup}
        extra = [(key, lo, hi) for key, lo, hi in faces_accepted if key not in dedup_keys]
        if len(extra) != 1:
            raise ValueError(f"N{n} owner {owner} ({label}-seam): expected exactly one accepted-only "
                              f"(duplicate) face, found {len(extra)}")
        dup_key, dup_lo, dup_hi = extra[0]

        keys_accepted = np.array([k for k, _l, _h in faces_accepted], dtype=np.int64)
        keys_dedup = np.array([k for k, _l, _h in faces_dedup], dtype=np.int64)
        keep_a = ~((keys_accepted[:, 0] == 0) & (keys_accepted[:, 1] == 0))
        keep_d = ~((keys_dedup[:, 0] == 0) & (keys_dedup[:, 1] == 0))
        fc_a = core.face_chunk(t, S, ref, ctx, keys_accepted[keep_a], owner_values, normal_coefficients, patch_cache,
                                period, normal_data_override=lattice.normal_data, dedupe=False)
        fc_d = core.face_chunk(t, S, ref, ctx, keys_dedup[keep_d], owner_values, normal_coefficients, patch_cache,
                                period, normal_data_override=lattice.normal_data, dedupe=True)
        fc_dup = core.face_chunk(t, S, ref, ctx, np.array([dup_key]), owner_values, normal_coefficients, patch_cache,
                                  period, normal_data_override=lattice.normal_data, dedupe=False)

        max_defect = 0.0
        for vi in range(len(core.VARIANT_NAMES)):
            sum_a = fc_a["correction_lo"][vi][fc_a["lo"] == owner].sum(axis=0) + \
                    fc_a["correction_hi"][vi][fc_a["hi"] == owner].sum(axis=0)
            sum_d = fc_d["correction_lo"][vi][fc_d["lo"] == owner].sum(axis=0) + \
                    fc_d["correction_hi"][vi][fc_d["hi"] == owner].sum(axis=0)
            dup_contribution = np.zeros(4)
            if dup_lo == owner:
                dup_contribution += fc_dup["correction_lo"][vi, 0]
            if dup_hi == owner:
                dup_contribution += fc_dup["correction_hi"][vi, 0]
            max_defect = max(max_defect, float(np.max(np.abs((sum_a - sum_d) - dup_contribution))))
        results[label] = dict(found=True, owner=owner, duplicate_face_key=[int(v) for v in dup_key],
                               max_defect=max_defect, pass_=bool(max_defect <= TOL["seam_atol"]))
    all_pass = all(r.get("pass_", False) for r in results.values() if r.get("found"))
    return dict(theta=results.get("theta"), eta=results.get("eta"), pass_=bool(all_pass))


# ---------------------------------------------------------------------------
def run(input_root, n, output_dir, oracle=False):
    """Campaign preflight (``oracle=False``): gates (a) batched replay of accepted
    P06, (b) structural and (c) seam, all through the batched kernels.
    ``oracle=True`` is the one-off ``verify-equivalence`` check: batched
    kernels vs the per-owner oracle on a covering owner subset.
    """
    input_root = Path(input_root); output_dir = Path(output_dir)
    t, ref = core.load(input_root, output_dir / "reference_sidecar.json", n)
    S = core.StructuredReconstruction(t)
    ctx = core.context(t)

    result = {"n": n}
    if oracle:
        result["check_equivalence"] = _equivalence_gate(t, S, ref, ctx, n)
        result["all_pass"] = bool(result["check_equivalence"]["pass_"])
        return result

    result["check_replay"] = _batched_replay_gate(t, S, ref, ctx, n)
    result.update(_structural_gate(t, S, ref, ctx, n))
    result["check_seam"] = _seam_gate(t, S, ref, ctx, n)
    checks = [result[k] for k in result if k.startswith("check_") and isinstance(result[k], dict) and "pass_" in result[k]]
    result["all_pass"] = bool(all(c["pass_"] for c in checks))
    return result
