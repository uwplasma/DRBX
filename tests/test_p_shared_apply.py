"""Validation for ``scripts/p_shared/apply.py`` (P08 step-1 design task 5).

Covers ``work/p08_step1_consolidation_design_20260928/design.md`` section 6
task 5: "Host apply.py. Compare against the per-owner oracles (p05n
rows/operator, p06n owner_q1/q3, p07n face_chunk)." -- concretely, this
module reproduces the frozen *chunk* functions' outputs
(``p05n_field_derived_global.core.raw_chunk``/``face_chunk``,
``p06n_field_derived_global.core.raw_chunk``/``face_chunk``,
``p07n_field_derived_global.core.face_chunk``) by building the same rows
those chunk functions build internally -- but via the *package* builders
(``drbx.geometry.fci_perpendicular_reconstruction.StructuredReconstruction``,
``fci_perpendicular_neumann_trace.prepare_neumann_point_rows``,
``fci_perpendicular_integrated_rows.prepare_integrated_face_rows``) instead
of the frozen campaigns' own ``scripts/perpendicular_structured`` copies --
and applying them with ``scripts/p_shared/apply.py``'s generic row appliers
plus its operator-specific wrappers around the frozen arithmetic (see that
module's docstring for the exact call sites reused by import).

All real-N32 tests below skip cleanly (module-wide) if the local HSX
workspace inputs the accepted campaigns read from, or the saved
``N32.owner_values.npz`` campaign outputs the task spec names, are not
present -- the same convention ``tests/test_stencils_geometry_provider.py``
uses.

Non-bitwise note (see this file's assertions and the final task report):
none of these comparisons are exactly bitwise because the row geometry is
built by a *different* (though AST-near-identical, see design.md section 1's
"Primitives" row) ``StructuredReconstruction``/``prepare_*`` implementation
than the one the frozen oracle's own chunk function uses internally
(package ``drbx.geometry`` vs. the accepted campaigns' own
``scripts/perpendicular_structured`` copy) -- every comparison here is
within roughly 1e-18 to 1e-21 absolute, far inside the 1e-12 relative
tolerance the task sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]  # .../DRBX
WORKSPACE = REPO.parent  # .../HSX drbx
sys.path.insert(0, str(REPO / "scripts"))

RUNTIME_INPUTS = WORKSPACE / "work/p05_direct_midpoint_global_565e1d1a_HsoyFbJ3/runtime_inputs"
SIDECAR = WORKSPACE / "work/p07n_extraction_hotspot_audit_20260926/localized_sidecar.json"
OWNER_VALUES_ROOT = WORKSPACE / "work/p05n_p06n_43250ccf_20260928T053254Z_c415a4cd"
P05N_OWNER_VALUES = OWNER_VALUES_ROOT / "p05n_upwind/N32.owner_values.npz"
P06N_OWNER_VALUES = OWNER_VALUES_ROOT / "p06n/N32.owner_values.npz"
N = 32


def _workspace_inputs_available() -> bool:
    if not RUNTIME_INPUTS.is_dir() or not SIDECAR.is_file():
        return False
    geometry_root = RUNTIME_INPUTS / "geometry_artifacts/rlp_convergence_32_48_64_20260917" / f"{N}x{N}x{N}"
    if not (geometry_root / "base_geometry.npz").is_file():
        return False
    if not (P05N_OWNER_VALUES.is_file() and P06N_OWNER_VALUES.is_file()):
        return False
    try:
        import json

        side = json.loads(SIDECAR.read_text())
        for key in ("metric_cache", "makegrid"):
            if not Path(side[key]["path"]).is_file():
                return False
        if not Path(side["artifact"]["path"]).is_dir():
            return False
    except Exception:
        return False
    return True


WORKSPACE_AVAILABLE = _workspace_inputs_available()
needs_workspace = pytest.mark.skipif(
    not WORKSPACE_AVAILABLE,
    reason="HSX workspace geometry inputs / saved N32.owner_values.npz campaign outputs are unavailable",
)


def _assert_tiny(actual, expected, *, label: str, atol: float = 1e-9, rtol: float = 1e-12):
    """Bitwise if possible, else within the task's 1e-12-relative tolerance.

    On failure the message reports the exact max abs/rel difference (the
    task spec: "report every non-bitwise array with its max abs/rel
    difference and the reason").
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if np.array_equal(actual, expected):
        return
    abs_diff = np.max(np.abs(actual - expected))
    scale = np.maximum(np.abs(expected), 1e-300)
    rel_diff = np.max(np.abs(actual - expected) / scale)
    assert abs_diff <= atol or rel_diff <= rtol, (
        f"{label}: not bitwise (row geometry rebuilt via the package builders instead of the "
        f"frozen campaign's own scripts copy -- design.md section 1's 'Primitives' row); "
        f"max abs diff {abs_diff:.3e}, max rel diff {rel_diff:.3e}"
    )


# ---------------------------------------------------------------------------
# P05N: raw_chunk (N, D, R) and face_chunk (N, D).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p05n_env():
    from p05n_field_derived_global import core as p05n_core
    from p05n_field_derived_global import fields as p05n_fields
    from perpendicular_structured.reconstruction import StructuredReconstruction as ScriptsSR
    from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction as PackageSR

    t, ref = p05n_core.load(RUNTIME_INPUTS, SIDECAR, N)
    with np.load(P05N_OWNER_VALUES, allow_pickle=False) as z:
        owner_values = z["values"]
    ctx = p05n_core.context(t)
    return dict(core=p05n_core, fields=p05n_fields, t=t, ref=ref, owner_values=owner_values,
                ctx=ctx, S_pkg=PackageSR(ctx), S_scripts=ScriptsSR(t))


@needs_workspace
@pytest.mark.slow
def test_p05n_raw_chunk_matches_apply(p05n_env):
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows

    p05n_core = p05n_env["core"]; p05n_fields = p05n_env["fields"]
    t = p05n_env["t"]; ref = p05n_env["ref"]; owner_values = p05n_env["owner_values"]
    ctx = p05n_env["ctx"]; S_pkg = p05n_env["S_pkg"]; S_scripts = p05n_env["S_scripts"]
    n = t.n; period = t.g.eta_period

    rng = np.random.default_rng(20260928)
    axis0 = np.unravel_index(np.arange(n ** 3), (n, n, n))[0]
    interior_ids = rng.choice(np.flatnonzero(axis0 < n - 2), size=4, replace=False)
    boundary_ids = rng.choice(np.flatnonzero(axis0 >= n - 2), size=4, replace=False)
    raw_ids = np.concatenate([interior_ids, boundary_ids])
    keys = np.array(np.unravel_index(raw_ids, (n,) * 3)).T
    points = t.pts[raw_ids]

    def dirichlet_trace_fn(q):
        return p05n_core.dirichlet_trace_all(ref, q, period)

    def normal_data_fn(q):
        return p05n_core.normal_data_all(ref, q, period)

    def normal_coefficients(q):
        return p05n_fields.normal(ref, q)

    patch_cache: dict = {}
    grad_d_list, grad_n_list = [], []
    for key, point in zip(keys, points):
        row = S_pkg.rows(tuple(int(v) for v in key), point[None, :], location="cell")
        value_d, grad_d = pshared_apply.apply_point_row(row, owner_values, dirichlet_trace_fn)
        if row.boundary_conditioned:
            nrow = prepare_neumann_point_rows(ctx, point[None, :], normal_coefficients=normal_coefficients,
                                              radial_degree=3, patch_cache=patch_cache)[0]
            g_bc = normal_data_fn(nrow.boundary_points)
            _value_n, grad_n = pshared_apply.apply_neumann_row(nrow, owner_values, g_bc)
        else:
            grad_n = grad_d[0]
        grad_d_list.append(grad_d[0])
        grad_n_list.append(grad_n)
    grad_d_arr = np.stack(grad_d_list)
    grad_n_arr = np.stack(grad_n_list)

    grad_roles = p05n_core._select_role_matrix(dict(dirichlet=grad_d_arr, neumann=grad_n_arr),
                                                p05n_core.ROLE_NAMES, p05n_core.ROLE_BC,
                                                p05n_core.ROLE_PHYSICAL_INDEX, is_gradient=True)

    metric = ref._metric(points)
    h = metric["bcov"] / metric["B"][:, None]
    jac = np.abs(metric["J"])
    actions, _antisym = pshared_apply.p05_centered_action(h, jac, grad_roles, p05n_core.ACTION_PAIR_INDEX)
    P = len(p05n_core.PAIR_NAMES)
    N_mat = actions[:, :P]
    D_mat = actions[:, P:]

    exact_grad = p05n_core.exact_grad_all(ref, points, period)
    R_mat = np.empty((len(points), P))
    for col, (a, b) in enumerate(p05n_core.R_PAIR_INDEX):
        R_mat[:, col] = pshared_apply.p05_bracket(h, jac, exact_grad[:, :, a], exact_grad[:, :, b])

    oracle = p05n_core.raw_chunk(t, S_scripts, ref, ctx, raw_ids, owner_values, normal_coefficients, {}, period)

    _assert_tiny(N_mat, oracle["N"], label="p05n raw_chunk N")
    _assert_tiny(D_mat, oracle["D"], label="p05n raw_chunk D")
    _assert_tiny(R_mat, oracle["R"], label="p05n raw_chunk R")


@needs_workspace
@pytest.mark.slow
def test_p05n_face_chunk_matches_apply(p05n_env, tmp_path):
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
    from p07_combined_global import kernels as pk
    from p07_combined_global import topology as pt

    p05n_core = p05n_env["core"]; p05n_fields = p05n_env["fields"]
    t = p05n_env["t"]; ref = p05n_env["ref"]; owner_values = p05n_env["owner_values"]
    ctx = p05n_env["ctx"]; S_pkg = p05n_env["S_pkg"]; S_scripts = p05n_env["S_scripts"]
    n = t.n; period = t.g.eta_period

    def dirichlet_trace_fn(q):
        return p05n_core.dirichlet_trace_all(ref, q, period)

    def normal_data_fn(q):
        return p05n_core.normal_data_all(ref, q, period)

    def normal_coefficients(q):
        return p05n_fields.normal(ref, q)

    rng = np.random.default_rng(20260929)
    face_ids_all, endpoints_all = p05n_core.global_faces(n, RUNTIME_INPUTS, tmp_path)
    keys_all = pt.decode(n, face_ids_all)
    interior_mask = keys_all[:, 1] < n - 2
    wall_mask = (keys_all[:, 0] == 0) & (keys_all[:, 1] == n)
    trans_mask = (keys_all[:, 0] != 0) & (keys_all[:, 1] >= n - 2)

    def pick(mask, k):
        idx = np.flatnonzero(mask)
        return rng.choice(idx, size=min(k, len(idx)), replace=False)

    chosen = np.concatenate([pick(interior_mask, 3), pick(wall_mask, 2), pick(trans_mask, 2)])
    face_ids = face_ids_all[chosen]
    endpoints = endpoints_all[chosen]
    keys = pt.decode(n, face_ids)

    order = 3
    patch_cache: dict = {}
    all_points, all_weights = pk.num.quadrature(t.faces, keys, order, face=True)

    N_list, D_list = [], []
    for row_i in range(len(face_ids)):
        key = keys[row_i]
        axis = int(key[0])
        points = all_points[row_i]
        weights = all_weights[row_i]
        keytuple = tuple(int(v) for v in key)

        kind = pk.family(n, key)
        boundary = kind in ("quartic_wall", "boundary_transverse")
        row = S_pkg.rows(keytuple, points, location="face")
        if axis == 0 and int(key[1]) == 0:
            _, gcommon = pshared_apply.apply_point_row(row, owner_values, None)
            gcommon_n = gcommon.copy()
        elif not boundary:
            _, gcommon = pshared_apply.apply_point_row(row, owner_values, None)
            gcommon_n = gcommon.copy()
        else:
            _, gcommon = pshared_apply.apply_point_row(row, owner_values, dirichlet_trace_fn)
            degree = 4 if kind == "quartic_wall" else 3
            nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                               radial_degree=degree, patch_cache=patch_cache)
            gcommon_n = np.stack(
                [pshared_apply.apply_neumann_row(nrow, owner_values, normal_data_fn(nrow.boundary_points))[1]
                 for nrow in nrows], axis=0)

        def cell_value(cell_key, exists, points=points):
            if not exists:
                return None
            cellkey = (int(cell_key[0]), int(cell_key[1]) % n, int(cell_key[2]) % n)
            conditioned = cellkey[0] >= n - 2
            crow = S_pkg.rows(cellkey, points, location="cell", fixed_anchor=True)
            assert bool(crow.boundary_conditioned) == conditioned
            if not conditioned:
                v, _ = pshared_apply.apply_point_row(crow, owner_values, None)
                return dict(dirichlet=v, neumann=v.copy())
            vd, _ = pshared_apply.apply_point_row(crow, owner_values, dirichlet_trace_fn)
            nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                               radial_degree=3, patch_cache=patch_cache)
            vn = np.stack(
                [pshared_apply.apply_neumann_row_value(nrow, owner_values, normal_data_fn(nrow.boundary_points))
                 for nrow in nrows], axis=0)
            return dict(dirichlet=vd, neumann=vn)

        ijk = [int(v) for v in key[1:]]
        left_key = ijk.copy(); left_key[axis] -= 1
        right_key = ijk.copy()
        left_exists = not (axis == 0 and left_key[0] < 0)
        right_exists = not (axis == 0 and right_key[0] >= n)
        L = cell_value(left_key, left_exists)
        R_side = cell_value(right_key, right_exists)
        trace_v = None
        if L is None or R_side is None:
            trace_v, _ = dirichlet_trace_fn(points)
        if L is None:
            L = dict(dirichlet=trace_v, neumann=R_side["neumann"].copy())
        if R_side is None:
            R_side = dict(dirichlet=trace_v, neumann=L["neumann"].copy())

        grad_roles = p05n_core._select_role_matrix(dict(dirichlet=gcommon, neumann=gcommon_n),
                                                    p05n_core.ROLE_NAMES, p05n_core.ROLE_BC,
                                                    p05n_core.ROLE_PHYSICAL_INDEX, is_gradient=True)
        lower_roles = p05n_core._select_role_matrix(L, p05n_core.ROLE_NAMES, p05n_core.ROLE_BC,
                                                     p05n_core.ROLE_PHYSICAL_INDEX, is_gradient=False)
        upper_roles = p05n_core._select_role_matrix(R_side, p05n_core.ROLE_NAMES, p05n_core.ROLE_BC,
                                                     p05n_core.ROLE_PHYSICAL_INDEX, is_gradient=False)

        metric = ref._metric(points)
        h = metric["bcov"] / metric["B"][:, None]
        jump = np.asarray(pshared_apply.p05_face_jump(
            grad_roles[None], lower_roles[None], upper_roles[None], h[None], weights[None],
            np.array([axis]), np.array(p05n_core.ACTION_PAIR_INDEX)))[0]
        P = len(p05n_core.PAIR_NAMES)
        N_list.append(jump[:P])
        D_list.append(jump[P:])

    N_mat = np.stack(N_list)
    D_mat = np.stack(D_list)

    oracle = p05n_core.face_chunk(t, S_scripts, ref, ctx, face_ids, endpoints, owner_values, normal_coefficients,
                                  {}, period, order=order)
    _assert_tiny(N_mat, oracle["N"], label="p05n face_chunk N")
    _assert_tiny(D_mat, oracle["D"], label="p05n face_chunk D")


# ---------------------------------------------------------------------------
# P06N: raw_chunk (material/remainder/total, R_*) and face_chunk
# (correction_lo/hi), dedupe=True and dedupe=False.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def p06n_env():
    from p06n_field_derived_global import core as p06n_core
    from p06n_field_derived_global import fields as p06n_fields
    from perpendicular_structured.reconstruction import StructuredReconstruction as ScriptsSR
    from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction as PackageSR

    t, ref = p06n_core.load(RUNTIME_INPUTS, SIDECAR, N)
    with np.load(P06N_OWNER_VALUES, allow_pickle=False) as z:
        owner_values = z["values"]
    ctx = p06n_core.context(t)
    return dict(core=p06n_core, fields=p06n_fields, t=t, ref=ref, owner_values=owner_values,
                ctx=ctx, S_pkg=PackageSR(ctx), S_scripts=ScriptsSR(t), variant="main_phi_neumann")


@needs_workspace
@pytest.mark.slow
def test_p06n_raw_chunk_matches_apply(p06n_env):
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
    from perpendicular_structured.reference_geometry import curvature_geometry
    import p06_structured_global.numerics as p06numerics

    p06n_core = p06n_env["core"]; p06n_fields = p06n_env["fields"]
    t = p06n_env["t"]; ref = p06n_env["ref"]; owner_values = p06n_env["owner_values"]
    ctx = p06n_env["ctx"]; S_pkg = p06n_env["S_pkg"]; S_scripts = p06n_env["S_scripts"]
    variant = p06n_env["variant"]
    n = t.n; period = t.g.eta_period
    tables = p06n_core.CATALOGUE_TABLES

    def dirichlet_trace_fn(q):
        return p06n_core.dirichlet_trace_all(ref, q, period, tables)

    def normal_data_fn(q):
        return p06n_core.normal_data_all(ref, q, period, tables)

    def normal_coefficients(q):
        return p06n_fields.normal(ref, q)

    rng = np.random.default_rng(20260930)
    axis0 = np.unravel_index(np.arange(n ** 3), (n, n, n))[0]
    interior_ids = rng.choice(np.flatnonzero(axis0 < n - 2), size=4, replace=False)
    boundary_ids = rng.choice(np.flatnonzero(axis0 >= n - 2), size=4, replace=False)
    raw_ids = np.concatenate([interior_ids, boundary_ids])
    keys = np.array(np.unravel_index(raw_ids, (n,) * 3)).T
    points = t.pts[raw_ids]

    patch_cache: dict = {}
    value_d_list, grad_d_list, value_n_list, grad_n_list = [], [], [], []
    for key, point in zip(keys, points):
        row = S_pkg.rows(tuple(int(v) for v in key), point[None, :], location="cell")
        value_d, grad_d = pshared_apply.apply_point_row(row, owner_values, dirichlet_trace_fn)
        if row.boundary_conditioned:
            nrow = prepare_neumann_point_rows(ctx, point[None, :], normal_coefficients=normal_coefficients,
                                              radial_degree=3, patch_cache=patch_cache)[0]
            g_bc = normal_data_fn(nrow.boundary_points)
            value_n, grad_n = pshared_apply.apply_neumann_row(nrow, owner_values, g_bc)
        else:
            value_n, grad_n = value_d[0], grad_d[0]
        value_d_list.append(value_d[0]); grad_d_list.append(grad_d[0])
        value_n_list.append(value_n); grad_n_list.append(grad_n)

    value_d_arr = np.stack(value_d_list); grad_d_arr = np.stack(grad_d_list)
    value_n_arr = np.stack(value_n_list); grad_n_arr = np.stack(grad_n_list)

    field_index = tables.field_index[variant]
    is_neumann = tables.is_neumann[variant]
    value = p06n_core._select(value_d_arr, value_n_arr, field_index, is_neumann, axis=1)
    gradient = p06n_core._select(grad_d_arr, grad_n_arr, field_index, is_neumann, axis=2)

    prepared = curvature_geometry(ref, points)
    material, remainder, total, *_ = pshared_apply.p06_q1_terms(value, gradient, prepared)

    spec = tables.variant_spec[variant]
    exact_values = np.empty_like(value.T)
    exact_gradients = np.empty_like(gradient.transpose(2, 0, 1))
    for j, (field, _bc) in enumerate(spec):
        ev, eg, _ = tables.evaluate(ref, points, field, period)
        exact_values[j] = ev
        exact_gradients[j] = eg
    exact = p06numerics._continuum_terms(exact_values, exact_gradients, prepared)
    r_material, r_remainder, r_total = exact[0], exact[1], exact[2]

    oracle = p06n_core.raw_chunk(t, S_scripts, ref, ctx, raw_ids, owner_values, normal_coefficients,
                                 {}, period, variants=[variant], tables=tables)

    _assert_tiny(material, oracle["material"][0], label="p06n raw_chunk material")
    _assert_tiny(remainder, oracle["remainder"][0], label="p06n raw_chunk remainder")
    _assert_tiny(total, oracle["total"][0], label="p06n raw_chunk total")
    _assert_tiny(r_material, oracle["R_material"][0], label="p06n raw_chunk R_material")
    _assert_tiny(r_remainder, oracle["R_remainder"][0], label="p06n raw_chunk R_remainder")
    _assert_tiny(r_total, oracle["R_total"][0], label="p06n raw_chunk R_total")


@needs_workspace
@pytest.mark.slow
@pytest.mark.parametrize("dedupe", [True, False])
def test_p06n_face_chunk_matches_apply(p06n_env, dedupe):
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
    from p07_combined_global import kernels as pk
    from p07_combined_global.kernels import num as pknum
    import p06_structured_global.numerics as p06numerics

    p06n_core = p06n_env["core"]; p06n_fields = p06n_env["fields"]
    t = p06n_env["t"]; ref = p06n_env["ref"]; owner_values = p06n_env["owner_values"]
    ctx = p06n_env["ctx"]; S_pkg = p06n_env["S_pkg"]; S_scripts = p06n_env["S_scripts"]
    variant = p06n_env["variant"]
    n = t.n; period = t.g.eta_period
    tables = p06n_core.CATALOGUE_TABLES

    def dirichlet_trace_fn(q):
        return p06n_core.dirichlet_trace_all(ref, q, period, tables)

    def normal_data_fn(q):
        return p06n_core.normal_data_all(ref, q, period, tables)

    def normal_coefficients(q):
        return p06n_fields.normal(ref, q)

    # Interior, physical wall, two transverse_last_two_layers, and one
    # periodic-duplicate (theta slot n) face -- exercising both census choices.
    keys = np.array([
        (0, n // 2, 5, 7),
        (0, n, 3, 4),
        (1, 5, n - 1, 6),
        (2, 6, 8, n - 1),
        (1, 10, n, 3),
    ], dtype=np.int64)

    patch_cache: dict = {}
    correction_lo_list, correction_hi_list = [], []
    all_points, all_weights = pknum.quadrature(t.faces, keys, 3, face=True)
    for row_i in range(len(keys)):
        key = keys[row_i]
        axis = int(key[0])
        points, weights = all_points[row_i], all_weights[row_i]
        J, B, K = p06numerics._face_geometry(ref, points)

        keytuple = tuple(int(v) for v in key)
        fam_kind = pk.family(n, key)
        boundary = fam_kind in ("quartic_wall", "boundary_transverse")
        crow = S_pkg.rows(keytuple, points, location="face")
        if not boundary:
            cval, _ = pshared_apply.apply_point_row(crow, owner_values, None)
            cval_n = cval.copy()
        else:
            cval, _ = pshared_apply.apply_point_row(crow, owner_values, dirichlet_trace_fn)
            degree = 4 if fam_kind == "quartic_wall" else 3
            nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                               radial_degree=degree, patch_cache=patch_cache)
            cval_n = np.stack(
                [pshared_apply.apply_neumann_row_value(nrow, owner_values, normal_data_fn(nrow.boundary_points))
                 for nrow in nrows], axis=0)

        def cell_value(cell_key, exists, points=points):
            if not exists:
                return None
            cellkey = (int(cell_key[0]), int(cell_key[1]) % n, int(cell_key[2]) % n)
            conditioned = cellkey[0] >= n - 2
            row = S_pkg.rows(cellkey, points, location="cell", fixed_anchor=True)
            assert bool(row.boundary_conditioned) == conditioned
            if not conditioned:
                v, _ = pshared_apply.apply_point_row(row, owner_values, None)
                return dict(dirichlet=v, neumann=v.copy())
            vd, _ = pshared_apply.apply_point_row(row, owner_values, dirichlet_trace_fn)
            nrows = prepare_neumann_point_rows(ctx, points, normal_coefficients=normal_coefficients,
                                               radial_degree=3, patch_cache=patch_cache)
            vn = np.stack(
                [pshared_apply.apply_neumann_row_value(nrow, owner_values, normal_data_fn(nrow.boundary_points))
                 for nrow in nrows], axis=0)
            return dict(dirichlet=vd, neumann=vn)

        ijk = [int(v) for v in key[1:]]
        left_key = ijk.copy(); left_key[axis] -= 1
        right_key = ijk.copy()
        left_exists = not (axis == 0 and left_key[0] < 0)
        right_exists = not (axis == 0 and right_key[0] >= n)
        L = cell_value(left_key, left_exists)
        R_side = cell_value(right_key, right_exists)
        trace_v = None
        if L is None or R_side is None:
            trace_v, _ = dirichlet_trace_fn(points)
        if L is None:
            L = dict(dirichlet=trace_v, neumann=R_side["neumann"].copy())
        if R_side is None:
            R_side = dict(dirichlet=trace_v, neumann=L["neumann"].copy())

        field_index = tables.field_index[variant]
        is_neumann = tables.is_neumann[variant]
        central_v = p06n_core._select(cval, cval_n, field_index, is_neumann, axis=1)
        lower_v = p06n_core._select(L["dirichlet"], L["neumann"], field_index, is_neumann, axis=1)
        upper_v = p06n_core._select(R_side["dirichlet"], R_side["neumann"], field_index, is_neumann, axis=1)

        normal_vec = J * K[:, axis] / np.maximum(B * B, 1e-30)
        is_wall = axis == 0 and int(key[1]) == n
        correction_lo, correction_hi, _fallback = pshared_apply.p06_q3_correction(
            central_v[:, :4], lower_v[:, :4], upper_v[:, :4], B, normal_vec, weights, wall=is_wall)
        correction_lo_list.append(correction_lo)
        correction_hi_list.append(correction_hi)

    correction_lo_arr = np.stack(correction_lo_list)
    correction_hi_arr = np.stack(correction_hi_list)

    oracle = p06n_core.face_chunk(t, S_scripts, ref, ctx, keys, owner_values, normal_coefficients,
                                  {}, period, variants=[variant], dedupe=dedupe, tables=tables)
    _assert_tiny(correction_lo_arr, oracle["correction_lo"][0], label=f"p06n face_chunk correction_lo (dedupe={dedupe})")
    _assert_tiny(correction_hi_arr, oracle["correction_hi"][0], label=f"p06n face_chunk correction_hi (dedupe={dedupe})")

    # The periodic-duplicate face (row 4: theta slot n) is the census choice
    # this parametrization exercises: dedupe=True drops it from both owner
    # endpoints (legacy_alias_slots), dedupe=False keeps it, matching
    # drbx.stencils.census.FaceCensus.dedupe_mask's documented convention.
    if dedupe:
        assert oracle["lo"][4] == -1 and oracle["hi"][4] == -1
    else:
        assert oracle["lo"][4] != -1 and oracle["hi"][4] != -1


# ---------------------------------------------------------------------------
# P07N: face_chunk (N, D) -- regular families (0/3/5/6/7) via IntegratedFaceRow,
# conditioned families (1/2/4) via the R2-N Neumann gradient contracted with
# the P07 tensor integrand.
# ---------------------------------------------------------------------------
@needs_workspace
@pytest.mark.slow
def test_p07n_face_chunk_matches_apply(tmp_path):
    sys.path.insert(0, str(REPO / "scripts/p07_combined_global"))
    sys.path.insert(0, str(REPO / "scripts/p07n_field_derived_global"))
    from p07n_field_derived_global import core as p07n_core
    from p07n_field_derived_global import fields as p07n_fields
    from drbx.geometry.fci_perpendicular_reconstruction import StructuredReconstruction as PackageSR
    from drbx.geometry.fci_perpendicular_neumann_trace import prepare_neumann_point_rows
    from drbx.geometry.fci_perpendicular_integrated_rows import prepare_integrated_face_rows
    from p_shared import apply as pshared_apply
    import topology
    from p07_diffusion_global import numerics as refnum

    t, ref = p07n_core.load(RUNTIME_INPUTS, SIDECAR, N)
    n = t.n
    ctx = p07n_core.context(t)
    S_pkg = PackageSR(ctx)

    topology.census(n, str(RUNTIME_INPUTS), str(tmp_path))
    with np.load(str(tmp_path) + f"/N{n}.topology.npz") as z:
        all_face_ids = z["face_ids"].copy()
        all_endpoints = z["endpoints"].copy()
        all_family = z["family"].copy()

    rng = np.random.default_rng(20261001)
    chosen = []
    for fam in (0, 3, 5, 6, 7, 1, 2, 4):
        idx = np.flatnonzero(all_family == fam)
        if len(idx):
            chosen.append(int(rng.choice(idx, size=1)[0]))
    chosen = np.array(chosen)
    face_ids = all_face_ids[chosen]
    family = all_family[chosen]
    endpoints = all_endpoints[chosen]
    assert set(family.tolist()) >= {0, 1, 2, 3, 4, 5, 6, 7}

    keys = topology.decode(n, face_ids)
    p, w = refnum.quadrature(t.faces, keys, 3, face=True)
    integ = np.zeros((len(face_ids), 9, 3))
    active = np.flatnonzero(family != 0)
    if len(active):
        tensor = ref._perpendicular_flux_tensor(p[active].reshape(-1, 3)).reshape(len(active), 9, 3, 3)
        weighted = w[active]
        integ_active = pshared_apply.contract_face_tensor(weighted, tensor, keys[active, 0])
        integ[active] = integ_active

    ordinary_mask = ~np.isin(family, (1, 2, 4))
    ordinary_idx = np.flatnonzero(ordinary_mask)
    irregular_idx = np.flatnonzero(~ordinary_mask)
    regular_rows = prepare_integrated_face_rows(ctx, keys[ordinary_idx], family[ordinary_idx],
                                                p[ordinary_idx], integ[ordinary_idx])

    def normal_coefficients(q):
        return p07n_fields.normal(ref, q)

    def dirichlet_trace_fn(q):
        v = np.column_stack([p07n_fields.evaluate(ref, q, name, t.g.eta_period)[0] for name in p07n_fields.NAMES])
        g = np.stack([p07n_fields.evaluate(ref, q, name, t.g.eta_period)[1] for name in p07n_fields.NAMES], axis=-1)
        return v, g

    patch_cache: dict = {}
    donors = [row.donor_ids for row in regular_rows]
    neumann_rows_by_face, dirichlet_rows_by_face = {}, {}
    for j in irregular_idx:
        degree = 3 if family[j] == 4 else 4
        nrows = prepare_neumann_point_rows(ctx, p[j], normal_coefficients=normal_coefficients,
                                           radial_degree=degree, patch_cache=patch_cache)
        neumann_rows_by_face[j] = nrows
        drow = prepare_integrated_face_rows(ctx, keys[j:j + 1], family[j:j + 1], p[j:j + 1], integ[j:j + 1])[0]
        dirichlet_rows_by_face[j] = drow
        donors.append(drow.donor_ids)
        donors.append(np.concatenate([r.donor_ids for r in nrows]))

    owner_values = p07n_core.selected_observations(t, ref, np.concatenate(donors) if donors else [])

    N_mat = np.zeros((len(face_ids), len(p07n_fields.NAMES)))
    D_mat = np.zeros((len(face_ids), len(p07n_fields.NAMES)))
    for local, j in enumerate(ordinary_idx):
        v = pshared_apply.p07_face_flux(regular_rows[local], owner_values, dirichlet_trace_fn)
        N_mat[j] = v
        D_mat[j] = v
    for j in irregular_idx:
        boundary_data = [
            np.column_stack([p07n_fields.normal_data(ref, nrow.boundary_points, name, t.g.eta_period)
                             for name in p07n_fields.NAMES])
            for nrow in neumann_rows_by_face[j]
        ]
        N_mat[j] = pshared_apply.p07_neumann_face_flux(neumann_rows_by_face[j], owner_values, boundary_data, integ[j])
        D_mat[j] = pshared_apply.p07_face_flux(dirichlet_rows_by_face[j], owner_values, dirichlet_trace_fn)

    oracle = p07n_core.face_chunk(t, ref, face_ids, family, endpoints, owner_values=owner_values)
    _assert_tiny(N_mat, oracle["N"], label="p07n face_chunk N")
    _assert_tiny(D_mat, oracle["D"], label="p07n face_chunk D")


# ---------------------------------------------------------------------------
# Light, geometry-free sanity for the generic row appliers and the thin
# frozen-arithmetic re-exports (project_raw_to_owners, the P07 owner scatter).
# ---------------------------------------------------------------------------
def test_apply_point_row_interior_matches_manual_contraction():
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_reconstruction import PointRows

    rng = np.random.default_rng(1)
    donor_ids = np.array([2, 5, 7])
    owner_values = rng.normal(size=(10, 3))
    value = rng.normal(size=(2, 3))
    gradient = rng.normal(size=(2, 3, 3))
    row = PointRows(donor_ids, value, gradient, False, np.empty((0, 3)), rng.normal(size=(2, 3)), {})

    v, g = pshared_apply.apply_point_row(row, owner_values)
    np.testing.assert_array_equal(v, value @ owner_values[donor_ids])
    np.testing.assert_array_equal(g, np.einsum("qad,df->qaf", gradient, owner_values[donor_ids]))


def test_apply_point_row_conditioned_applies_the_boundary_lift():
    from p_shared import apply as pshared_apply
    from drbx.geometry.fci_perpendicular_reconstruction import PointRows

    rng = np.random.default_rng(2)
    donor_ids = np.array([1, 2])
    owner_values = rng.normal(size=(5, 2))
    value = rng.normal(size=(1, 2))
    gradient = rng.normal(size=(1, 3, 2))
    donor_points = rng.normal(size=(2, 3))
    target_points = rng.normal(size=(1, 3))
    row = PointRows(donor_ids, value, gradient, True, donor_points, target_points, {})

    def trace(points):
        v = np.sum(points, axis=1, keepdims=True) * np.ones((len(points), 2))
        g = np.zeros((len(points), 3, 2))
        g[:, 0] = 1.0
        return v, g

    v, g = pshared_apply.apply_point_row(row, owner_values, trace)
    gd, _ = trace(donor_points)
    gt, dgt = trace(target_points)
    expected_v = value @ (owner_values[donor_ids] - gd) + gt
    expected_g = np.einsum("qad,df->qaf", gradient, owner_values[donor_ids] - gd)
    expected_g[:, 1:] += dgt[:, 1:]
    np.testing.assert_array_equal(v, expected_v)
    np.testing.assert_array_equal(g, expected_g)


def test_p05_project_to_owners_matches_direct_operator_formula():
    from p_shared import apply as pshared_apply

    rng = np.random.default_rng(3)
    raw_action = rng.normal(size=(6, 2))
    raw_volume = rng.uniform(0.5, 1.5, size=6)
    raw_owner = np.array([0, 0, 1, 1, 2, 2])
    owner_volume = np.array([raw_volume[0] + raw_volume[1], raw_volume[2] + raw_volume[3],
                             raw_volume[4] + raw_volume[5]])
    owner_action = pshared_apply.p05_project_to_owners(raw_action, raw_volume, raw_owner, owner_volume)
    expected = np.zeros((3, 2))
    np.add.at(expected, raw_owner, raw_volume[:, None] * raw_action)
    expected /= owner_volume[:, None]
    np.testing.assert_array_equal(owner_action, expected)


def test_p07_scatter_flux_matches_lower_minus_upper_plus_over_volume():
    from p_shared import apply as pshared_apply

    face_flux = np.array([[1.0, 2.0], [3.0, -1.0]])
    lower_owner = np.array([0, 1])
    upper_owner = np.array([1, -1])
    owner_volume = np.array([2.0, 4.0])
    out = np.asarray(pshared_apply.p07_scatter_flux(face_flux, lower_owner, upper_owner, owner_volume))
    expected = np.zeros((2, 2))
    expected[0] -= face_flux[0]
    expected[1] += face_flux[0] - face_flux[1]
    expected /= owner_volume[:, None]
    np.testing.assert_allclose(out, expected)


def test_p06_scatter_correction_adds_both_sides_independently():
    from p_shared import apply as pshared_apply

    lower_numerator = np.array([[1.0], [2.0]])
    upper_numerator = np.array([[3.0], [4.0]])
    lower_owner = np.array([0, 1])
    upper_owner = np.array([1, -1])
    evolution_volume = np.array([2.0, 5.0])
    out = np.asarray(pshared_apply.p06_scatter_correction(
        lower_numerator, upper_numerator, lower_owner, upper_owner, evolution_volume))
    expected = np.zeros((2, 1))
    expected[0] += lower_numerator[0]
    expected[1] += lower_numerator[1] + upper_numerator[0]
    expected /= evolution_volume[:, None]
    np.testing.assert_allclose(out, expected)
