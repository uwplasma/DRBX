"""Campaign-only, case-specific exact BC evaluation in bounded wall batches.

The original catalogue remains the authority. No reconstruction, tracing or
boundary law is changed. Dense nonwall padding is constructed only on restore.
"""
from types import SimpleNamespace
import numpy as np


def fields(points, case, common):
    """Literal catalogue formula for one state, retaining operation order."""
    if not isinstance(case, (int, np.integer)) or not 0 <= case < common.NF:
        raise ValueError('invalid Q08 catalogue case')
    p = np.asarray(points)
    u, th, e = np.moveaxis(p, -1, 0)
    x, y = u*np.cos(th), u*np.sin(th)
    dx = np.stack((np.cos(th), -y, np.zeros_like(u)), axis=-1)
    dy = np.stack((np.sin(th), x, np.zeros_like(u)), axis=-1)
    d = common.DESIGNS[case]
    vals, grads = [], []
    for j in range(5):
        if d['name'] == 'constant':
            f, g = np.zeros_like(u), np.zeros_like(p)
        elif d['name'] == 'smooth':
            f, g = common.primitive_modes(p, 'sx' if j % 2 == 0 else 'sy')
        else:
            a = np.deg2rad(d['angle'] + 13*j)
            k, m = 2*np.pi/d['wavelength'], 1+j % 2
            ph = k*(np.cos(a)*x + np.sin(a)*y) + m*e + d['phase'] + .3*j
            dp = k*(np.cos(a)*dx + np.sin(a)*dy)
            dp[..., 2] += m
            f, g = np.cos(ph), -np.sin(ph)[..., None]*dp
        vals.append(common.BASE[j] + common.AMP[j]*f)
        grads.append(common.AMP[j]*g)
    return np.stack(vals, axis=-1), np.stack(grads, axis=-2)


def phi_fields(points, case, common):
    a, da = common.primitive_modes(points, 'sx')
    b, db = common.primitive_modes(points, 'sy')
    v, g = .07*a + .04*b, .07*da + .04*db
    return (v*0, g*0) if case == 0 else (v, g)


def compact_boundaries(bank, case, common, *, batch_rows=128):
    """Return cache groups (shape, raw axis, exact template, wall values).

    Temporaries scale with wall count plus a bounded query batch, rather than
    all raw cells times all 22 fields. The affine omega padding is deliberate.
    """
    from drbx.native.q_parallel import QBoundaryData
    if not isinstance(case, (int, np.integer)) or not 0 <= case < common.NF:
        raise ValueError('invalid Q08 catalogue case')
    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError('positive boundary batch size required')
    nr, nw = len(bank.raw), len(bank.wall_index)
    shapes = [(nw, 35), (nw, 3), (nw, 3, 2), (nw, 35)]
    inner = [np.zeros((5, *s)) for s in shapes]
    outer = [np.zeros_like(a) for a in inner]
    phi = [np.zeros(s) for s in shapes]
    for start in range(0, nw, batch_rows):
        sl = slice(start, start + batch_rows)
        points = bank.query_table[bank.wall_node_query[sl]]
        v, g = fields(points, case, common)
        normal = np.einsum('wqa,wqfa->fwq', bank.boundary_wall_normal[sl], g)
        for target in (inner, outer):
            target[0][:, sl] = v.transpose(2, 0, 1)
            target[3][:, sl] = normal
        for target, slots in ((outer, common.SPAN_SLOTS[0]), (inner, common.SPAN_SLOTS[1])):
            query = bank.query_table[bank.wall_slot_query[sl][:, list(slots)]]
            v, g = fields(query, case, common)
            target[1][:, sl] = v.transpose(2, 0, 1)
            target[2][:, sl] = g[..., 1:].transpose(2, 0, 1, 3)
        v, g = phi_fields(points, case, common)
        phi[0][sl] = v
        phi[3][sl] = np.einsum('wqa,wqa->wq', bank.boundary_wall_normal[sl], g)
        query = bank.query_table[bank.wall_slot_query[sl][:, list(common.SPAN_SLOTS[1])]]
        v, g = phi_fields(query, case, common)
        phi[1][sl], phi[2][sl] = v, g[..., 1:]
    groups = []
    for gi, target in enumerate((inner, outer, phi)):
        arrays = []
        for i, a in enumerate(target):
            axis = 1 if gi < 2 else 0
            if gi < 2:
                extra = np.einsum('f...,f->...', a, common.WC)
                if i in (0, 1):
                    extra += common.WOFF
                a = np.concatenate((a, extra[None]), axis=0)
            rows = np.moveaxis(a, axis, 0).copy()
            template = np.zeros(rows.shape[1:], dtype=rows.dtype)
            if gi < 2 and i in (0, 1):
                template[5] = common.WOFF
            shape = list(a.shape)
            shape[axis] = nr
            arrays.append((tuple(shape), axis, template, rows))
        groups.append((QBoundaryData, arrays))
    return groups


def catalogue_replay(bank, common):
    """Independent old-producer gate on distributed actual wall queries.

    Eight wall rows, every state, all values/gradients/normals and nonwall
    padding are checked bitwise. Full-grid action gates remain unchanged.
    """
    from boundary_cache import restore
    nw = len(bank.wall_index)
    chosen = np.unique(np.linspace(0, nw-1, min(8, nw), dtype=int)) if nw else np.array([], dtype=int)
    sample = SimpleNamespace(raw=np.arange(len(chosen)+1), wall_index=np.arange(len(chosen)),
        query_table=bank.query_table, wall_node_query=bank.wall_node_query[chosen],
        wall_slot_query=bank.wall_slot_query[chosen], boundary_wall_normal=bank.boundary_wall_normal[chosen])
    for case in range(common.NF):
        actual = restore(sample.wall_index, compact_boundaries(sample, case, common))
        expected = common.boundaries(sample, case)
        for gi, (a, b) in enumerate(zip(actual, expected)):
            for ai, (x, y) in enumerate(zip(a, b)):
                if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                    raise ValueError(f'optimized boundary bitwise replay case={case} group={gi} array={ai}')
    return dict(passed=True, cases=common.NF, wall_positions=chosen.tolist(),
                all_arrays_bitwise=True, nonwall_padding_checked=True)
