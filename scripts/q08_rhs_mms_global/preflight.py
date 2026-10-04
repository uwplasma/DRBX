"""Replay scientific O/R and candidate N against persisted bounded HSX actions."""
from pathlib import Path
import time
import numpy as np
from scripts.q08_rhs_mms_global.campaign import write


def bounded(run, old, identity, canonical, *, gpu=False, test_cpu=False):
    import jax
    from scripts.q08_extraction_global import common as c
    from drbx.stencils.q_parallel import load_chunk
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import apply_q_plan
    from drbx.native.q_parallel import QBoundaryData
    from scripts.q08_rhs_mms_global.science import oracle, continuum, numerical, check_action_identities
    from gpu_stage import compiler_guard
    if gpu:
        from scripts.q08_extraction_global.gpu import device_inventory
        devices, inventory = device_inventory(test_cpu)
        device = devices[0]
    else:
        inventory = {}; device = jax.devices('cpu')[0]
    inputs = Path(old)/'inputs'
    model = c.load_geometry_model(inputs)
    maxima = dict(N=0., O=0., R=0.); fractions = dict(maxima); sensitivity = []
    tick = time.perf_counter()
    for n in (32, 48, 64):
        pair = tuple(load_chunk(inputs/f'bounded/N{n}_h{d}.npz') for d in (16, 32))
        bank = build_q_bank(*pair)
        with np.load(inputs/f'bounded/N{n}_inputs.npz') as z:
            data = {k: z[k].copy() for k in z.files}
        geo = {k: data[k] for k in ('magnetic_L', 'b_eta', 'eta_step', 'bmag')}
        with np.load(inputs/f'bounded/actions_N{n}.npz') as z:
            saved = {k: z[k].copy() for k in z.files}
        O = oracle(bank, geo)
        ctx, _ = model.context(n, canonical)
        R, diag = continuum(bank, geo, lambda p: model.geom(ctx, p))
        sensitivity.append(diag)
        state = np.broadcast_to(np.r_[c.BASE, .2][None, :, None],
                                (22, 6, bank.metadata['n_owner'])).copy()
        state[:, :, data['donor_ids']] = data['state']
        phi = np.zeros((22, bank.metadata['n_owner'])); phi[:, data['donor_ids']] = data['phi']
        boundaries = tuple(QBoundaryData(*(data[f'{name}_bc_{i}'] for i in range(4)))
                           for name in ('inner', 'outer', 'phi'))
        for ai, span in enumerate((1/16, 1/32)):
            plan = lower_q_plan(bank, diffusion_span=span, **geo)
            for ki, (kinds, pk) in enumerate(c.KINDS):
                found = dict(O=O[ai], R=R)
                if gpu:
                    call = jax.jit(lambda qp, x, bi, bo, p, pb, co: apply_q_plan(qp, x, bi, bo, p, pb, co,
                        kinds=kinds, phi_kind=pk, tau=c.TAU, mu=c.MU, characteristic_method='polynomial'))
                    args = jax.tree.map(lambda a: jax.device_put(a, device),
                        (plan, state, boundaries[0], boundaries[1], phi, boundaries[2], c.COEFF))
                    result = call(*args)
                    found['N'] = numerical(bank, result)
                    compiler_guard(call.lower(*args).compile().as_text(), test_cpu=test_cpu)
                for tag, actual in found.items():
                    # Saved baseline predates the last three diagnostics.
                    # Validate their signed force identities as well as the
                    # original 28-column persisted action replay.
                    check_action_identities(actual)
                    keys = ('centered', 'correction', 'diffusion', 'combined', 'current',
                            'omega_advection', 'omega_current', 'phi_force')
                    expected = np.concatenate([saved[f's{int(1/span)}_k{ki}_{key}_{tag}'] for key in keys], axis=-1)
                    err = abs(actual[..., :28]-expected)
                    frac = float(np.max(err/(1e-8+1e-11*abs(expected))))
                    if not np.isfinite(err).all() or frac > 1:
                        raise ValueError(f'bounded {tag} replay N{n}: {frac}')
                    maxima[tag] = max(maxima[tag], float(err.max())); fractions[tag] = max(fractions[tag], frac)
        del state, phi, ctx, plan
        jax.clear_caches()
        print(f'bounded scientific replay N{n} passed (GPU={gpu})', flush=True)
    result = dict(passed=True, identity=identity, test_only=test_cpu, gpu=gpu,
        device_inventory=inventory, seconds=time.perf_counter()-tick,
        max_abs=maxima, max_budget_fraction=fractions, reference_diagnostics=sensitivity,
        persisted_outputs=28, identity_checked_outputs=31,
        scope='21 complete HSX owners, all22 states, four BC patterns, both diffusion spans')
    write(Path(run)/('preflight_gpu.json' if gpu else 'preflight.json'), result)
    return result
