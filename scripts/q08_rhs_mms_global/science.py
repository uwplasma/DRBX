"""Static six-field MMS oracle and continuum target; no row construction.

O uses analytic values/gradients on the frozen traced stencil. R uses analytic
field derivatives and an independent fourth-order coordinate flux derivative
of the same C3 magnetic evaluator. The finite difference is a reference device,
not an FCI leg or a face quadrature rule. Projection is performed last.
"""
import numpy as np
from scripts.q08_extraction_global import common as c

FIELDS = ('n', 'Te', 'Ti', 'Vi', 'Ve', 'omega')
TERMS = tuple(f'{a}_{f}' for a in ('centered', 'correction', 'diffusion', 'combined')
              for f in FIELDS) + ('current', 'omega_advection', 'omega_current',
                                  'phi_force', 'electron_material',
                                  'electron_ti_compensation', 'electron_generalized_force')
SPANS = (1/16, 1/32)
STEPS = (1e-4, 5e-5, 2.5e-5)
REGIONS = ('global', 'core', 'first_ring', 'inner', 'last_two_aggregate',
           'transition', 'bulk', 'wall', 'outermost_wall')
METRICS = ('N-O', 'O-R', 'N-R')


def project(bank, value):
    """(..., raw, components) -> (..., complete owners, components)."""
    return np.sum(np.take(value, bank.owner_raw, axis=-2)*bank.owner_weight[..., None], axis=-2)


def pack(centered, correction, diffusion, current, omega_adv, omega_cur,
         phi_force, electron_material, ti_compensation, generalized_force):
    return np.concatenate((centered, correction, diffusion,
        centered+correction+diffusion, np.stack((current, omega_adv, omega_cur,
        phi_force, electron_material, ti_compensation, generalized_force), axis=-1)), axis=-1)


def numerical(bank, result):
    if not np.asarray(result.inputs_valid).all() or not np.asarray(result.eigensystem_admissible).all():
        raise ValueError('invalid numerical thermodynamic state or characteristic split')
    cp = result.raw_current
    names = ('divergence_physical', 'vorticity_current', 'electron_phi',
             'electron_ti_compensation', 'electron_generalized_force')
    raw = np.stack([np.asarray(getattr(cp, k)) for k in names] +
                   [np.asarray(result.raw_electron_material)], axis=-1)
    p = project(bank, raw)
    nc, nu, nd, total = map(np.asarray, result[:4])
    out = pack(nc, nu, nd, p[..., 0], nc[..., 5]-p[..., 1], p[..., 1],
               p[..., 2], p[..., 5], p[..., 3], p[..., 4])
    np.testing.assert_allclose(out[..., 18:24], total, rtol=1e-14, atol=1e-12)
    if not np.isfinite(out).all():
        raise ValueError('nonfinite numerical action')
    return out


def oracle(bank, geometry):
    """Exact-point discrete O, with the independent CPU eig characteristic path."""
    import jax
    from drbx.native.q_parallel_material import material_from_slots
    from drbx.native.q_parallel_vorticity import vorticity_from_slots
    points = bank.diagnostics['slot_points']
    values, gradient = c.six_fields(points)
    v = values.transpose(2, 0, 1, 3)
    phi = c.phi_fields(points[:, [1, 3, 2]])[0].transpose(2, 0, 1)
    # All reference chunks share one CPU compilation, including short tails.
    nr = len(bank.raw)
    if not 0 < nr <= 128:
        raise ValueError('reference chunk exceeds frozen raw capacity')
    ix = np.r_[np.arange(nr), np.full(128-nr, nr-1)]
    L, beta, delta = (np.asarray(geometry[k]) for k in ('magnetic_L', 'b_eta', 'eta_step'))
    step = np.broadcast_to(delta, (nr,))
    mat = _material(v[:, ix, :, :5], phi[:, ix], L[ix], beta[ix], step[ix])
    if not all(np.asarray(a).all() for a in mat[5:]):
        raise ValueError('invalid exact-state characteristic oracle')
    mc, mu = np.asarray(mat.centered)[:, :nr], np.asarray(mat.correction)[:, :nr]
    ov = vorticity_from_slots(v[..., 5], v[:, :, 2, 3], beta, delta)
    dj = np.sum(L[None]*v[:, :, [1, 3, 2], 0]*(v[:, :, [1, 3, 2], 3]-v[:, :, [1, 3, 2], 4]), axis=-1)
    current = geometry['bmag'][None]**2/v[:, :, 2, 0]*dj
    gc = beta[None]/(2*delta)*(phi[:, :, 1]-phi[:, :, 0])
    gi = beta[None]/(2*delta)*(v[:, :, 3, 2]-v[:, :, 1, 2])
    centered = np.concatenate((mc, (np.asarray(ov.centered)+current)[..., None]), axis=-1)
    correction = np.concatenate((mu, np.asarray(ov.correction)[..., None]), axis=-1)
    outputs = []
    for ai, slots in enumerate(c.SPAN_SLOTS):
        g = gradient[:, list(slots)]
        gf = np.einsum('rsa,rscfa->crfs', bank.magnetic_b[:, list(slots)], g)
        diffusion = c.COEFF*np.einsum('rs,crfs->crf', bank.magnetic_L[ai], gf)
        outputs.append(project(bank, pack(centered, correction, diffusion, dj,
            np.asarray(ov.centered), current, c.MU*gc,
            np.asarray(mat.material)[:, :nr, 4], c.MU*c.TAU*gi, c.MU*(gc+c.TAU*gi))))
    return np.stack(outputs)


def _material(*args):
    import jax
    from drbx.native.q_parallel_material import material_from_slots
    if not hasattr(_material, 'compiled'):
        _material.compiled = jax.jit(lambda *a: material_from_slots(*a, tau=c.TAU, mu=c.MU,
                                                                  characteristic_method='eig'), backend='cpu')
    return _material.compiled(*args)


def continuum(bank, geometry, geom):
    """Independent center target; evaluate geometry only once per FD sample."""
    p = bank.diagnostics['slot_points'][:, 2]
    J0, b0, B0 = geom(p)
    value, gradient = c.six_fields(p)
    V = value.transpose(1, 0, 2)
    G = np.einsum('ra,rcfa->crf', b0, gradient)
    gp = np.einsum('ra,rca->cr', b0, c.phi_fields(p)[1])
    n, te, ti, vi, ve, om = np.moveaxis(V, -1, 0)
    gn, gt, gi, gvi, gve, go = np.moveaxis(G, -1, 0)
    results = []
    kappas = []
    for h in STEPS:
        pp = np.broadcast_to(p[:, None, None, :], (len(p), 3, 4, 3)).copy()
        for d in range(3):
            pp[:, d, :, d] += np.array([-2, -1, 1, 2])*h
        flat = pp.reshape(-1, 3)
        J, b, _ = geom(flat)
        _, g = c.six_fields(flat)
        gf = np.einsum('pa,pcfa->pcf', b, g)
        flux = (J[:, None, None, None]*b[:, :, None, None]*gf[:, None]).reshape(len(p), 3, 4, 3, c.NF, 6)
        w = np.array([1, -8, 8, -1])/(12*h)
        diffusion = sum(np.einsum('rqcf,q->rcf', flux[:, d, :, d], w) for d in range(3))/J0[:, None, None]
        jb = (J[:, None]*b).reshape(len(p), 3, 4, 3)
        kappa = sum(np.einsum('rq,q->r', jb[:, d, :, d], w) for d in range(3))/J0
        kappas.append(kappa)
        kap = kappa[None]
        dj = (vi-ve)*gn+n*(gvi-gve)+kap*n*(vi-ve)
        phi_force = c.MU*gp
        em = -ve*gve-c.MU*(te*gn+n*gt)/n-.71*c.MU*gt-c.MU*c.TAU*gi
        material = np.stack((-(ve*gn+n*gve+kap*n*ve),
            -ve*gt+2*te/(3*n)*(.71*dj-n*(gve+kap*ve)),
            -vi*gi+2*ti/(3*n)*(dj-n*(gvi+kap*vi)),
            -vi*gvi-((te+c.TAU*ti)*gn+n*(gt+c.TAU*gi))/n,
            -ve*gve-c.MU*(te*gn+n*gt)/n-.71*c.MU*gt+phi_force), axis=-1)
        adv = -vi*go
        cur = B0[None]**2/n*dj
        centered = np.concatenate((material, (adv+cur)[..., None]), axis=-1)
        results.append(project(bank, pack(centered, np.zeros_like(centered),
            c.COEFF*diffusion.transpose(1, 0, 2), dj, adv, cur,
            phi_force, em, c.MU*c.TAU*gi, c.MU*(gp+c.TAU*gi))))
    diagnostics = dict(center_b_replay_max=float(abs(b0[:, 2]-geometry['b_eta']).max()),
        center_B_replay_max=float(abs(B0-geometry['bmag']).max()),
        kappa_replay_max=float(abs(kappas[0]-geometry['magnetic_L'].sum(-1)).max()),
        reference_step_max_by_term=np.max(abs(np.stack(results[1:])-results[0]), axis=(0, 1, 2)).tolist(),
        reference_steps=list(STEPS))
    # Hardware replay guard, distinct from the reported FD sensitivity/science.
    if diagnostics['center_b_replay_max'] > 1e-10 or diagnostics['center_B_replay_max'] > 1e-10:
        raise ValueError('frozen C3 center geometry replay mismatch')
    if not np.isfinite(results).all():
        raise ValueError('nonfinite continuum target')
    return results[0], diagnostics


def masks(radial, n, last):
    r = np.asarray(radial)
    return (np.ones(len(r), bool), r == 0, r == 1, r <= last,
        (r >= last-1)&(r <= last), (r >= last-1)&(r <= last+2),
        (r > last)&(r < n-2), r >= n-2, r == n-1)


def reduce_case(N, O, R, owners, volume, radial, n, last):
    """One span/BC/state. Regions deliberately overlap, as in earlier Q gates."""
    N, O, R = map(np.asarray, (N, O, R))
    if N.shape != (len(owners), len(TERMS)) or O.shape != N.shape or R.shape != N.shape:
        raise ValueError('scientific action coverage/shape mismatch')
    if not all(np.isfinite(a).all() for a in (N, O, R, volume)) or np.any(volume <= 0):
        raise ValueError('nonfinite action or invalid physical volume')
    errors = np.stack((N-O, O-R, N-R), axis=1)
    out = dict(sum2=[], signed=[], maximum=[], max_owner=[], reference_sum2=[], volume=[], count=[])
    for mask in masks(radial, n, last):
        e, w = errors[mask], volume[mask]
        out['count'].append(int(mask.sum())); out['volume'].append(float(w.sum()))
        out['sum2'].append(np.einsum('o,omt->mt', w, e*e))
        out['signed'].append(np.einsum('o,omt->mt', w, e))
        out['reference_sum2'].append(np.einsum('o,ot->t', w, R[mask]**2))
        if mask.any():
            pos = abs(e).argmax(axis=0)
            out['maximum'].append(abs(e).max(axis=0)); out['max_owner'].append(owners[mask][pos])
        else:
            out['maximum'].append(np.zeros((3, len(TERMS))))
            out['max_owner'].append(np.full((3, len(TERMS)), -1, dtype=np.int64))
    return {k: np.asarray(v) for k, v in out.items()}
