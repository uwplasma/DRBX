"""C0 (inner_support="profile7") vs one candidate inner support on identical owners: real P row builders and operators.

    python -m p08_inner_support_eval.run 32 --c1 last_aggregate --input-root ROOT --out DIR [--per-ring 12]
        # from DRBX/scripts; writes DIR/results_N32.json and DIR/receipt_N32.json

Normally launched by ``campaign.py`` (one subprocess per grid x candidate).  Single-threaded, about 5-6 GiB peak at N64.
Read-only for ``DRBX/src`` and every other scripts package; nothing large is written.  The mode keys of the output stay
``C0`` (profile7) and ``C1`` (the candidate, whatever its campaign name); ``--candidate`` / ``--c1`` are recorded.
Everything at the current defaults (autodiff curvature K, q2 face quadrature; ``configuration.json``), C0 and the
candidate differing only in ``inner_support``.  The owner closure is built with
``p_shared.owner_closure.build_owner_rows``; levels:

* (a) reconstruction: R1 cell value+gradient at raw midpoints and R2 face-node value+gradient against exact, split into
  value / u / theta / eta / transverse (= sqrt(u^2 + theta^2)) / combined;
* (b) P07 per owner: N (integrated R4 rows), O (exact gradient on the same q3 integrand), R (continuum at raw midpoints,
  Hessians from jax): N-O, O-R, N-R;
* (c) P05 centered bracket + live jump (R1/R2/R3) and P06 q1 (K.grad) at raw midpoints.

Complex fields enter as real and imaginary columns, scored as sqrt(re^2 + im^2).  The short waves (lambda 0.5, 0.25) are a
response report only (amplitude ratio, phase error, gradient reversal), never part of an order or pooled statistic.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[1]                # .../DRBX/scripts; the package is imported by its location only
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared.replay_support import build_environment                         # noqa: E402
from p_shared import owner_closure as oc                                      # noqa: E402
from p_shared import apply as pshared_apply                                   # noqa: E402
from perpendicular_structured.reference_geometry import curvature_geometry    # noqa: E402
from drbx.geometry.fci_perpendicular_integrated_rows import contract_face_tensor   # noqa: E402
import drbx.geometry._fci_perpendicular_point_primitives as prim              # noqa: E402

from p08_inner_support_eval import q_fields as q                              # noqa: E402
from p08_inner_support_eval import sample_build as sb                         # noqa: E402
from p08_inner_support_eval.dispatch import dispatch_check, row_family        # noqa: E402

SCHEMA = 'drbx.p08-inner-support-evaluation.v1'
CONFIG = json.loads((HERE / 'configuration.json').read_text())
CURVATURE, FACE_QUADRATURE = CONFIG['curvature'], CONFIG['face_quadrature']
MODES = {'C0': CONFIG['baseline_inner_support'], 'C1': 'last_aggregate'}     # 'C1' (the candidate) is set from --c1
COMPS = ('value', 'u', 'theta', 'eta', 'transverse', 'combined')
NF = len(q.FIELDS)
C = q.N_COLUMNS
CM = q.N_MAIN_COLUMNS
M_MAIN = q.MAIN_MATRIX[:CM]                          # (CM, NF): main columns -> field (complex magnitude)
P05_PAIRS = (('fA', 'fB'),
             ('common', 'x_lambda2_m1:re'),
             ('x_lambda2_m1:re', 'y_lambda2_m1:re'),
             ('x_lambda2_m1:im', 'y_lambda2_m1:im'),
             ('wave_a60_lambda2:re', 'fresh_a105_lambda4:im'))


def col(spec):
    name, _, part = spec.partition(':')
    return q.COLUMN_INDEX[(name, part or 're')]


PAIR_COLS = [(col(a), col(b)) for a, b in P05_PAIRS]


def rss_gib():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (2 ** 30 if sys.platform == 'darwin' else 2 ** 20)


def save(path, obj):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.bool_):
            return bool(x)
        raise TypeError(type(x))
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')             # atomic: a partial file is never seen
    tmp.write_text(json.dumps(obj, default=default, allow_nan=True))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fieldmag(err):
    """``(..., CM)`` column errors -> ``(..., NF)`` field errors, sqrt(re^2 + im^2) for complex fields."""
    return np.sqrt((err * err) @ M_MAIN)


def comp_errors(value_err, grad_err):
    """value ``(S, CM)`` and gradient ``(S, 3, CM)`` column errors -> ``(S, 6, NF)`` (COMPS order)."""
    ev = fieldmag(value_err)
    eu, et, ee = (fieldmag(grad_err[:, a]) for a in range(3))
    return np.stack([ev, eu, et, ee, np.sqrt(eu ** 2 + et ** 2), np.sqrt(eu ** 2 + et ** 2 + ee ** 2)], axis=1)


def rms_max(x):
    """RMS and max over the leading axis; ``None`` for an empty sample."""
    if len(x) == 0:
        return None
    return dict(rms=np.sqrt(np.mean(x * x, axis=0)), max=np.max(x, axis=0))


class PinvRecorder:
    """Records (columns, rank, condition number) of every ``primitives.pinv`` call while active (harness-only)."""

    def __enter__(self):
        self.records, self.orig = [], prim.pinv

        def wrapped(A, root):
            C_, rank, cond = self.orig(A, root)
            self.records.append((int(np.shape(A)[1]), int(rank), float(cond)))
            return C_, rank, cond
        prim.pinv = wrapped
        return self

    def __exit__(self, *exc):
        prim.pinv = self.orig

    def summary(self):
        out = {}
        for cols in sorted({r[0] for r in self.records}):
            rec = np.array([r[1:] for r in self.records if r[0] == cols])
            out[str(cols)] = dict(calls=len(rec), min_rank=int(rec[:, 0].min()), cond_max=float(rec[:, 1].max()),
                                  cond_median=float(np.median(rec[:, 1])), cond_p99=float(np.quantile(rec[:, 1], 0.99)))
        return out


def owner_values_dense(t, chunk=32768):
    """Volume-weighted mean of the exact columns over each owner's raw cells, dense over every owner (the closure
    convention: ``owner_values`` is indexed directly by ``row.donor_ids``)."""
    num = np.zeros((len(t.vol), C))
    for start in range(0, len(t.ro), chunk):
        sl = slice(start, start + chunk)
        v, _ = q.exact_columns(t.pts[sl])
        for c in range(C):
            num[:, c] += np.bincount(t.ro[sl], weights=t.rv[sl] * v[:, c], minlength=len(t.vol))
    return num / t.vol[:, None]


def build_rows_safe(envs, providers, owners):
    """Build C0 and C1 owner rows on the same owners.  A ``RuntimeError`` (unsupported stencil) in either mode is
    reported per owner and the failing owners are dropped from both modes."""
    failures = {}
    recorders, built = {}, {}

    def attempt(owner_list):
        for mode in MODES:
            with PinvRecorder() as rec:
                t0 = time.time()
                built[mode] = oc.build_owner_rows(envs[mode], owner_list, provider=providers[mode])
                built[mode]['build_seconds'] = time.time() - t0
            recorders[mode] = rec
    try:
        attempt(owners)
    except Exception as exc:                                        # pragma: no cover - failure path
        failures['_whole_build'] = repr(exc)
        bad = {}
        for owner in owners:
            for mode in MODES:
                try:
                    oc.build_owner_rows(envs[mode], [owner], provider=providers[mode])
                except Exception as e2:
                    bad[owner] = f'{mode}: {e2!r}'
        failures.update({str(k): v for k, v in bad.items()})
        owners = [o for o in owners if o not in bad]
        attempt(owners)
    return owners, built, recorders, failures


# ---------------------------------------------------------------------------
# per-mode operator evaluation
# ---------------------------------------------------------------------------
def evaluate_mode(mode, env, built, shared):
    """Everything that depends on the rows: per-sample error arrays and row diagnostics (see module docstring)."""
    t, census = env.t, env.census
    OV = shared['OV']
    row_index = built['row_index']
    raw_ids, face_rows = shared['raw_ids'], shared['face_rows']
    geometry = built['geometry']
    Qf = shared['Qf']
    t0 = time.time()

    # --- R1 cells ---------------------------------------------------------------------------------------------
    n_raw = len(raw_ids)
    rec_v = np.empty((n_raw, C)); rec_g = np.empty((n_raw, 3, C))
    diag_r1 = []
    for a, raw in enumerate(raw_ids):
        row = row_index[('R1', int(raw))]
        if row.boundary_conditioned:
            raise RuntimeError(f'unexpected boundary-conditioned R1 row at raw {raw}')
        v, g = pshared_apply.apply_point_row(row, OV, trace=None)
        rec_v[a], rec_g[a] = v[0], g[0]
        diag_r1.append(row)
    ex_v, ex_g = shared['cell_v'], shared['cell_g']
    e_cell = comp_errors(rec_v[:, :CM] - ex_v[:, :CM], rec_g[:, :, :CM] - ex_g[:, :, :CM])

    # --- P06 q1 and P05 centered at raw midpoints --------------------------------------------------------------
    K = shared['K']
    p06 = fieldmag(np.einsum('qa,qac->qc', K, rec_g[:, :, :CM] - ex_g[:, :, :CM]))
    h_raw, jac_raw = shared['h_raw'], shared['jac_raw']
    p05c = np.empty((n_raw, len(PAIR_COLS)))
    for p, (ca, cb) in enumerate(PAIR_COLS):
        N_ab = pshared_apply.p05_bracket(h_raw, jac_raw, rec_g[:, :, ca], rec_g[:, :, cb])
        p05c[:, p] = np.abs(N_ab - shared['p05_R'][:, p])

    # --- R2 faces (and R3 side values for the jump) ------------------------------------------------------------
    n_face = len(face_rows)
    f_v = np.empty((n_face, Qf, C)); f_g = np.empty((n_face, Qf, 3, C))
    jump = np.full((n_face, len(PAIR_COLS)), np.nan)
    diag_r2, diag_r3 = [], []
    for a, ridx in enumerate(face_rows):
        row = row_index[('R2', int(ridx))]
        if row.boundary_conditioned:
            raise RuntimeError(f'unexpected boundary-conditioned R2 row at census row {ridx}')
        v, g = pshared_apply.apply_point_row(row, OV, trace=None)
        f_v[a], f_g[a] = v, g
        diag_r2.append(row)
        lo_row, hi_row = row_index.get(('R3', int(ridx) * 2)), row_index.get(('R3', int(ridx) * 2 + 1))
        for side_row in (lo_row, hi_row):
            if side_row is not None:
                diag_r3.append((a, side_row))
        if lo_row is None or hi_row is None:
            continue
        lo_val, _ = pshared_apply.apply_point_row(lo_row, OV, trace=None)
        hi_val, _ = pshared_apply.apply_point_row(hi_row, OV, trace=None)
        wq = geometry.p06_face_weight[a][None, :]
        axis_f = np.array([int(shared['keys'][ridx, 0])])
        for p, (ca, cb) in enumerate(PAIR_COLS):
            cols = [ca, cb]
            j = pshared_apply.p05_face_jump(g[None][..., cols], lo_val[None][..., cols], hi_val[None][..., cols],
                                            shared['h_face'][a][None], wq, axis_f, [(0, 1)])
            jump[a, p] = abs(float(j[0, 0]))
    fx_v, fx_g = shared['face_v'], shared['face_g']
    e_face = comp_errors((f_v[..., :CM] - fx_v[..., :CM]).reshape(-1, CM),
                         (f_g[..., :CM] - fx_g[..., :CM]).reshape(-1, 3, CM)).reshape(n_face, Qf, len(COMPS), NF)

    # --- P07: N from the integrated rows, scattered to the sample owners ------------------------------------
    p07_rows = shared['p07_rows']
    flux_N = np.empty((len(p07_rows), CM))
    diag_r4 = []
    for a, pid in enumerate(shared['p07_ids']):
        row = row_index[int(pid)]
        if row.boundary_conditioned:
            raise RuntimeError('unexpected boundary-conditioned P07 row')
        flux_N[a] = pshared_apply.apply_integrated_row(row, OV[:, :CM], trace=None)
        diag_r4.append(row)
    dense_N = np.zeros((len(t.vol), CM))
    np.add.at(dense_N, shared['p07_lower'], -flux_N)
    np.add.at(dense_N, shared['p07_upper'], flux_N)
    N_owner = dense_N[shared['owners']] / t.vol[shared['owners']][:, None]
    O_owner, R_owner = shared['O_owner'], shared['R_owner']
    e_p07 = np.stack([fieldmag(N_owner - O_owner), fieldmag(O_owner - R_owner), fieldmag(N_owner - R_owner)], axis=1)

    # --- short-wave response (complex, from the re/im columns) -------------------------------------------------
    def complex_short(arr):                                        # (..., C) -> (..., 4)
        cols = [q.FIELD_COLUMNS[name] for name in q.SHORT_FIELDS]
        return np.stack([arr[..., c[0]] + 1j * arr[..., c[1]] for c in cols], axis=-1)

    resp = dict(cell_v=complex_short(rec_v), cell_g=complex_short(rec_g), face_v=complex_short(f_v),
                face_g=complex_short(f_g))
    return dict(e_cell=e_cell, p06=p06, p05c=p05c, e_face=e_face, jump=jump, e_p07=e_p07, resp=resp,
                rows=dict(r1=diag_r1, r2=diag_r2, r3=diag_r3, r4=diag_r4), seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def row_stats(rows, coupled_flag):
    """``rows``: point rows (``coupled_flag`` = family string test) or P07 rows (family code 7)."""
    if not rows:
        return None
    donors = np.array([len(r.donor_ids) for r in rows])
    if rows and hasattr(rows[0], 'diagnostics'):
        coupled = np.array([row_family(r) == 'coupled_quartic' for r in rows])
        ranks = [r.diagnostics.get('min_rank') for r in rows if r.diagnostics.get('min_rank') is not None]
        res = [r.diagnostics.get('max_residual') for r in rows if r.diagnostics.get('max_residual') is not None]
        families = defaultdict(int)
        for r in rows:
            families[row_family(r)] += 1
        extra = dict(min_rank=int(min(ranks)) if ranks else None, max_residual=float(max(res)) if res else None,
                     families=dict(families))
    else:
        coupled = np.array([r.family == 7 for r in rows])
        families = defaultdict(int)
        for r in rows:
            families[str(r.family)] += 1
        extra = dict(families=dict(families))
    return dict(n=len(rows), coupled_fraction=float(coupled.mean()), donors_mean=float(donors.mean()),
                donors_max=int(donors.max()), **extra)


def response_stats(rec_v, rec_g, ex_v, ex_g):
    """Short-wave response per field: amplitude ratios (value, transverse gradient), phase error of the value (rad),
    fraction of gradient reversal Re(g_rec . conj(g_ex)) < 0 (transverse = u, theta components)."""
    if len(rec_v) == 0:
        return None
    out = {}
    for f, name in enumerate(q.SHORT_FIELDS):
        v_r, v_e = rec_v[:, f], ex_v[:, f]
        g_r, g_e = rec_g[:, :2, f], ex_g[:, :2, f]
        amp_v = np.abs(v_r) / np.abs(v_e)
        amp_g = np.linalg.norm(g_r, axis=1) / np.linalg.norm(g_e, axis=1)
        phase = np.angle(v_r * np.conj(v_e))
        dot = np.real(np.sum(g_r * np.conj(g_e), axis=1))
        out[name] = dict(n=len(v_r), amp_value=dict(mean=amp_v.mean(), min=amp_v.min(), max=amp_v.max()),
                         amp_grad=dict(mean=amp_g.mean(), min=amp_g.min(), max=amp_g.max()),
                         phase_rms=float(np.sqrt(np.mean(phase ** 2))), phase_max=float(np.abs(phase).max()),
                         reversal_fraction=float(np.mean(dot < 0)))
    return out


def aggregate(res, shared, groups):
    """Per group metrics from one mode's per-sample arrays."""
    t = shared['t']
    raw_owner = t.ro[shared['raw_ids']]
    w_raw = t.rv[shared['raw_ids']] / t.vol[raw_owner]
    lo, hi = shared['face_lo'], shared['face_hi']
    p07_lo, p07_hi = shared['p07_lower'], shared['p07_upper']
    owners = shared['owners']
    out = {}
    for gid, gowners in groups.items():
        gset = np.array(sorted(gowners))
        cell = np.flatnonzero(np.isin(raw_owner, gset))
        face = np.flatnonzero(np.isin(lo, gset) | np.isin(hi, gset))
        p07f = np.flatnonzero(np.isin(p07_lo, gset) | np.isin(p07_hi, gset))
        own = np.flatnonzero(np.isin(owners, gset))

        def owner_proj(err):                                       # audit's owner-projected mean magnitude, then RMS
            if len(cell) == 0:
                return None
            ro = raw_owner[cell]
            uniq, inv = np.unique(ro, return_inverse=True)
            w = w_raw[cell]
            num = np.zeros((len(uniq),) + err.shape[1:])
            np.add.at(num, inv, w.reshape((-1,) + (1,) * (err.ndim - 1)) * err[cell])
            den = np.bincount(inv, weights=w)
            return np.sqrt(np.mean((num / den.reshape((-1,) + (1,) * (err.ndim - 1))) ** 2, axis=0))

        entry = dict(n_owner=len(own), n_raw=len(cell), n_face=len(face))
        cs = rms_max(res['e_cell'][cell])
        if cs:
            entry['a_cell'] = {c: dict(rms=cs['rms'][i], max=cs['max'][i]) for i, c in enumerate(COMPS)}
        fs = rms_max(res['e_face'][face].reshape(-1, len(COMPS), NF))
        if fs:
            entry['a_face'] = {c: dict(rms=fs['rms'][i], max=fs['max'][i]) for i, c in enumerate(COMPS)}
        ps = rms_max(res['e_p07'][own])
        if ps:
            entry['b_p07'] = {name: dict(rms=ps['rms'][i], max=ps['max'][i]) for i, name in enumerate(('N-O', 'O-R', 'N-R'))}
        p06s = rms_max(res['p06'][cell])
        if p06s:
            entry['c_p06'] = dict(raw_rms=p06s['rms'], raw_max=p06s['max'], owner_rms=owner_proj(res['p06']))
        p05s = rms_max(res['p05c'][cell])
        if p05s:
            opj = owner_proj(res['p05c'])
            entry['d_p05_centered'] = {f'{a}|{b}': dict(raw_rms=float(p05s['rms'][i]), raw_max=float(p05s['max'][i]),
                                                        owner_rms=float(opj[i]))
                                       for i, (a, b) in enumerate(P05_PAIRS)}
        jv = res['jump'][face]
        if len(jv):
            entry['d_p05_jump'] = {f'{a}|{b}': dict(rms=float(np.sqrt(np.nanmean(jv[:, i] ** 2))) if np.any(np.isfinite(jv[:, i])) else None,
                                                    max=float(np.nanmax(jv[:, i])) if np.any(np.isfinite(jv[:, i])) else None,
                                                    n=int(np.isfinite(jv[:, i]).sum()))
                                   for i, (a, b) in enumerate(P05_PAIRS)}
        rows = res['rows']
        r1 = [rows['r1'][i] for i in cell]
        r2 = [rows['r2'][i] for i in face]
        face_set = set(int(i) for i in face)
        r3 = [r for (a, r) in rows['r3'] if a in face_set]
        r4 = [rows['r4'][i] for i in p07f]
        entry['rows'] = dict(R1=row_stats(r1, None), R2=row_stats(r2, None), R3=row_stats(r3, None), R4=row_stats(r4, None))
        rs = res['resp']
        entry['short_waves'] = dict(
            cell=response_stats(rs['cell_v'][cell], rs['cell_g'][cell], shared['short_cell_v'][cell], shared['short_cell_g'][cell]),
            face=response_stats(rs['face_v'][face].reshape(-1, len(q.SHORT_FIELDS)),
                                rs['face_g'][face].reshape(-1, 3, len(q.SHORT_FIELDS)),
                                shared['short_face_v'][face].reshape(-1, len(q.SHORT_FIELDS)),
                                shared['short_face_g'][face].reshape(-1, 3, len(q.SHORT_FIELDS))))
        out[gid] = entry
    return out


# ---------------------------------------------------------------------------
def main(n, per_ring, out_dir, *, input_root, sidecar_path, candidate=None, identity=None):
    wall0 = time.time()
    input_root, sidecar_path = Path(input_root), Path(sidecar_path)
    receipt = dict(schema=SCHEMA, n=n, per_ring=per_ring, curvature=CURVATURE, face_quadrature=FACE_QUADRATURE,
                   modes=dict(MODES), candidate=candidate, inner_support=MODES['C1'], identity=identity,
                   input_root=str(input_root), sidecar=str(sidecar_path), timings={})
    receipt['port_check'] = q.verify_port()

    t0 = time.time()
    envs = {mode: build_environment(n=n, input_root=input_root, sidecar_path=sidecar_path, curvature=CURVATURE,
                                    face_quadrature=FACE_QUADRATURE, inner_support=support)
            for mode, support in MODES.items()}
    providers = {mode: oc.load_provider_for_env(sidecar_path, curvature=CURVATURE, face_quadrature=FACE_QUADRATURE)
                 for mode in MODES}
    receipt['timings']['environments'] = time.time() - t0
    t = envs['C0'].t
    last = envs['C0'].S.last
    assert envs['C1'].S.last == last and envs['C1'].t.n == t.n
    census = envs['C0'].census
    profile = [int(x) for x in envs['C0'].S.profile]

    entries = sb.build_sample(t, n, last, per_ring)
    owners = sb.owners_of(entries)
    t0 = time.time()
    owners, built, recorders, failures = build_rows_safe(envs, providers, owners)
    receipt['timings']['build_owner_rows'] = {m: built[m]['build_seconds'] for m in MODES}
    receipt['timings']['build_owner_rows_total'] = time.time() - t0
    receipt['failures'] = failures
    entries = [e for e in entries if e['owner'] in set(owners)]
    receipt['n_owners'] = len(owners)
    b0 = built['C0']
    for k in ('raw_ids', 'face_row_indices', 'p07_row_indices'):
        assert np.array_equal(b0[k], built['C1'][k])
    for name in ('face_points', 'p06_face_weight', 'p07_face_tensor', 'p07_face_points', 'p07_face_weight'):
        assert np.array_equal(getattr(b0['geometry'], name), getattr(built['C1']['geometry'], name)), name
    geometry = b0['geometry']
    raw_ids, face_rows, p07_rows = b0['raw_ids'], b0['face_row_indices'], b0['p07_row_indices']
    receipt['closure'] = dict(n_raw=len(raw_ids), n_face_rows=len(face_rows), n_p07_rows=len(p07_rows),
                              face_nodes=int(geometry.face_points.shape[1]))
    for mode in MODES:
        for raw in raw_ids:
            if built[mode]['row_index'][('R1', int(raw))].boundary_conditioned:
                raise RuntimeError(f'unexpected boundary-conditioned R1 row at raw {raw}')

    # ---- shared (mode independent) quantities ---------------------------------------------------------------
    t0 = time.time()
    env = envs['C0']
    keys = census.keys()
    shared = dict(t=t, OV=owner_values_dense(t), raw_ids=raw_ids, face_rows=face_rows, p07_rows=p07_rows,
                  Qf=int(geometry.face_points.shape[1]), keys=keys, owners=np.array(owners, dtype=np.int64))
    receipt['timings']['owner_values'] = time.time() - t0

    pts_raw = t.pts[raw_ids]
    shared['cell_v'], shared['cell_g'] = q.exact_columns(pts_raw)
    shared['short_cell_v'] = np.stack([shared['cell_v'][:, c[0]] + 1j * shared['cell_v'][:, c[1]]
                                       for c in (q.FIELD_COLUMNS[s] for s in q.SHORT_FIELDS)], axis=-1)
    shared['short_cell_g'] = np.stack([shared['cell_g'][:, :, c[0]] + 1j * shared['cell_g'][:, :, c[1]]
                                       for c in (q.FIELD_COLUMNS[s] for s in q.SHORT_FIELDS)], axis=-1)
    Qf = shared['Qf']
    node_points = np.stack([np.asarray(built['C0']['row_index'][('R2', int(r))].trace_target_points) for r in face_rows]) \
        if len(face_rows) else np.zeros((0, Qf, 3))
    for mode in ('C1',):
        for r, pts in zip(face_rows, node_points):
            assert np.array_equal(np.asarray(built[mode]['row_index'][('R2', int(r))].trace_target_points), pts)
    flat_nodes = node_points.reshape(-1, 3)
    fv, fg = q.exact_columns(flat_nodes)
    shared['face_v'], shared['face_g'] = fv.reshape(len(face_rows), Qf, C), fg.reshape(len(face_rows), Qf, 3, C)
    short_cols = [q.FIELD_COLUMNS[s] for s in q.SHORT_FIELDS]
    shared['short_face_v'] = np.stack([shared['face_v'][..., c[0]] + 1j * shared['face_v'][..., c[1]] for c in short_cols], axis=-1)
    shared['short_face_g'] = np.stack([shared['face_g'][..., c[0]] + 1j * shared['face_g'][..., c[1]] for c in short_cols], axis=-1)
    shared['face_lo'], shared['face_hi'] = census.owner_lo[face_rows], census.owner_hi[face_rows]

    # K, h, jac at raw midpoints; h at face nodes
    shared['K'] = np.asarray(curvature_geometry(env.ref, pts_raw).K)
    metric_raw = env.ref._metric(pts_raw)
    shared['h_raw'] = metric_raw['bcov'] / metric_raw['B'][:, None]
    shared['jac_raw'] = np.abs(metric_raw['J'])
    eg = shared['cell_g']
    shared['p05_R'] = np.stack([pshared_apply.p05_bracket(shared['h_raw'], shared['jac_raw'], eg[:, :, ca], eg[:, :, cb])
                                for ca, cb in PAIR_COLS], axis=1)
    metric_f = env.ref._metric(flat_nodes)
    shared['h_face'] = (metric_f['bcov'] / metric_f['B'][:, None]).reshape(len(face_rows), Qf, 3)

    # P07: O_q3 and R, dense over the sample owners
    face_pos = np.searchsorted(face_rows, p07_rows)
    keys_p07 = keys[p07_rows]
    shared['p07_ids'] = census.p07_id[p07_rows]
    shared['p07_lower'], shared['p07_upper'] = census.owner_lo[p07_rows], census.owner_hi[p07_rows]
    integrand = contract_face_tensor(geometry.p07_weight[face_pos], geometry.p07_face_tensor[face_pos], keys_p07[:, 0])
    _v, g7 = q.exact_columns(geometry.p07_points[face_pos].reshape(-1, 3), q.FIELDS)
    flux_O = np.einsum('fqa,fqac->fc', integrand, g7.reshape(len(p07_rows), 9, 3, CM))
    dense_O = np.zeros((len(t.vol), CM))
    np.add.at(dense_O, shared['p07_lower'], -flux_O)
    np.add.at(dense_O, shared['p07_upper'], flux_O)
    shared['O_owner'] = dense_O[shared['owners']] / t.vol[shared['owners']][:, None]
    J = metric_raw['J']
    Tt, div = env.ref._perpendicular_geometry(pts_raw)
    _v, G, H = q.jax_vgh(pts_raw, q.FIELDS)
    pointwise = -(np.einsum('qj,qjc->qc', div, G) + np.einsum('qij,qijc->qc', Tt, H)) / J[:, None]
    num = np.zeros((len(t.vol), CM)); den = np.zeros(len(t.vol))
    np.add.at(num, t.ro[raw_ids], t.rv[raw_ids][:, None] * pointwise)
    np.add.at(den, t.ro[raw_ids], t.rv[raw_ids])
    shared['R_owner'] = num[shared['owners']] / den[shared['owners']][:, None]
    receipt['timings']['shared_exact'] = time.time() - t0

    # ---- groups ----------------------------------------------------------------------------------------------
    groups = defaultdict(set)
    for e in entries:
        for gid in sb.groups_of(e, last):
            groups[gid].add(e['owner'])

    # ---- modes -----------------------------------------------------------------------------------------------
    results = dict(schema=SCHEMA, n=n, candidate=candidate, inner_support=MODES['C1'], identity=identity,
                   per_ring=per_ring, last=last, profile=profile, fields=list(q.FIELDS), comps=list(COMPS),
                   short_fields=list(q.SHORT_FIELDS), p05_pairs=[list(p) for p in P05_PAIRS],
                   group_kinds={gid: gid.split(':')[0] for gid in groups},
                   groups={gid: sorted(int(o) for o in s) for gid, s in groups.items()},
                   modes={})
    for mode in MODES:
        t0 = time.time()
        res = evaluate_mode(mode, envs[mode], built[mode], shared)
        agg = aggregate(res, shared, groups)
        allrows = res['rows']
        results['modes'][mode] = dict(
            inner_support=MODES[mode], groups=agg,
            rows_all=dict(R1=row_stats(allrows['r1'], None), R2=row_stats(allrows['r2'], None),
                          R3=row_stats([r for _a, r in allrows['r3']], None), R4=row_stats(allrows['r4'], None)),
            pinv=recorders[mode].summary(), evaluate_seconds=res['seconds'], build_seconds=built[mode]['build_seconds'])
        receipt['timings'][f'evaluate_{mode}'] = time.time() - t0
    results['dispatch_check'] = dispatch_check(built['C0']['row_index'], built['C1']['row_index'],
                                               candidate_support=MODES['C1'])
    receipt['dispatch_violations'] = results['dispatch_check']['n_violations']
    results['failures'] = failures

    receipt['wall_seconds'] = time.time() - wall0
    receipt['peak_rss_gib'] = rss_gib()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save(Path(out_dir) / f'results_N{n}.json', results)
    save(Path(out_dir) / f'receipt_N{n}.json', receipt)            # last: a receipt implies complete results
    print(f"N{n}: {receipt['n_owners']} owners, closure {receipt['closure']}, dispatch violations "
          f"{results['dispatch_check']['n_violations']}, failures {len(failures)}, wall {receipt['wall_seconds']:.0f} s, "
          f"peak RSS {receipt['peak_rss_gib']:.2f} GiB")
    return results, receipt


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('n', type=int)
    ap.add_argument('--per-ring', type=int, default=CONFIG['per_ring'])
    ap.add_argument('--out', required=True, help='output folder of this (candidate, grid): results_N{n}.json, receipt_N{n}.json')
    ap.add_argument('--c1', default='last_aggregate', help="inner_support of the candidate mode 'C1' (e.g. fixed_radius)")
    ap.add_argument('--candidate', default=None, help='campaign name of the candidate (recorded only)')
    ap.add_argument('--input-root', required=True, help='immutable HSX input root (holds input_manifest.json inputs)')
    ap.add_argument('--sidecar', default=None,
                    help='localized continuum sidecar; default: localize from --input-root into --out')
    ap.add_argument('--identity', default=None, help='campaign identity to record (set by campaign.py)')
    return ap.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    MODES['C1'] = args.c1
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sidecar = args.sidecar
    if sidecar is None:
        from p08_step1_global.campaign import localize_sidecar
        sidecar = localize_sidecar(Path(args.input_root), Path(args.out))
    main(args.n, args.per_ring, args.out, input_root=args.input_root, sidecar_path=sidecar,
         candidate=args.candidate, identity=args.identity)
