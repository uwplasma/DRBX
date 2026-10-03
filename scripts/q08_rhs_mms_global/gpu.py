"""One four-A100 action per case; no repeated performance/replay campaign."""
from pathlib import Path
import gc
import time
import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.q08_rhs_mms_global.campaign import BASE_ID, read, write, sha, require
from scripts.q08_extraction_global import gpu as oldgpu, common as c
from scripts.q08_extraction_global.common import atomic_npz
from scripts.q08_extraction_global.gpu import _place_plan, _sync
from boundary_cache import CachedAPI
from boundary_values import catalogue_replay
from gpu_stage import compiler_guard, memory_snapshot
from scripts.q08_rhs_mms_global.science import TERMS, SPANS, numerical, reduce_case


def load_references(run, old, n, identity):
    from scripts.q08_rhs_mms_global.references import validate
    receipt = require(run, f'references_N{n}', identity)
    # Validate shapes and coverage independently before creating merge scratch.
    checked = validate(run, old, identity, n)
    for k in ('files', 'owners', 'chunks', 'reference_step_max_by_term'):
        if checked[k] != receipt[k]:
            raise ValueError('reference completion differs from independent validation')
    root = run/'tmp'/f'N{n}'; root.mkdir(exist_ok=True)
    no = receipt['owners']; nt = len(TERMS)
    O = np.lib.format.open_memmap(root/'O.npy', mode='w+', dtype=np.float64, shape=(2, 22, no, nt))
    R = np.lib.format.open_memmap(root/'R.npy', mode='w+', dtype=np.float64, shape=(22, no, nt))
    volume = np.zeros(no); radial = np.zeros(no, dtype=np.int64)
    for i in range(receipt['chunks']):
        with np.load(run/'references'/f'N{n}'/f'chunk_{i:06d}.npz') as z:
            oo = z['owners']; O[:, :, oo, :] = z['O']; R[:, oo, :] = z['R']
            volume[oo] = z['volume']; radial[oo] = z['radial']
    O.flush(); R.flush()
    return O, R, volume, radial


def record_valid(path, identity, n, span, kind, case):
    if not path.exists():
        return False
    r = read(path)
    expected = dict(identity=identity, n=n, span=span, kind=kind, case=case, passed=True)
    if any(r.get(k) != v for k, v in expected.items()) or sha(path.with_suffix('.npz')) != r.get('sha256'):
        raise ValueError('scientific checkpoint identity/content mismatch')
    return r


def validate_grid(run, identity, n):
    root = run/'science'/f'N{n}'; files = {}; records = []; proofs = {}; coverage = None
    for span in SPANS:
        for ki in range(4):
            for ci in range(22):
                p = root/f's{int(1/span)}_k{ki}_c{ci:02d}.json'
                r = record_valid(p, identity, n, span, ki, ci)
                if not r:
                    raise ValueError('missing scientific action record')
                with np.load(p.with_suffix('.npz')) as z:
                    shapes = dict(sum2=(9, 3, len(TERMS)), signed=(9, 3, len(TERMS)),
                        maximum=(9, 3, len(TERMS)), max_owner=(9, 3, len(TERMS)),
                        reference_sum2=(9, len(TERMS)), volume=(9,), count=(9,))
                    if set(z.files) != set(shapes) or any(z[k].shape != s for k, s in shapes.items()):
                        raise ValueError('scientific reduction shape')
                    if any(not np.isfinite(z[k]).all() for k in z.files) or np.any(z['volume'] <= 0):
                        raise ValueError('scientific nonfinite/empty region')
                    if z['count'][0] != r['owners'] or z['count'][0] != {32:25376, 48:86016, 64:202304}[n]:
                        raise ValueError('global owner coverage mismatch')
                    if coverage is None:
                        coverage = (z['count'].copy(), z['volume'].copy())
                    elif not np.array_equal(coverage[0], z['count']) or not np.array_equal(coverage[1], z['volume']):
                        raise ValueError('case/BC/span regional coverage differs')
                    if np.any(z['max_owner'] < 0) or np.any(z['max_owner'] >= r['owners']):
                        raise ValueError('maximum owner identity outside grid')
                    if np.any(z['maximum'] < 0) or np.any(z['sum2'] < 0):
                        raise ValueError('negative norm statistic')
                if (not np.isfinite([r['constant_error'], r['source_pair_budget']]).all() or
                    not 0 <= r['constant_error'] <= 1e-7 or not 0 <= r['source_pair_budget'] <= 1):
                    raise ValueError('constant/source-pair gate')
                name = r['compiler']; proof = (run/name).resolve()
                if not proof.is_relative_to(run.resolve()):
                    raise ValueError('compiler proof path escapes output')
                if name not in proofs:
                    compiler_guard(proof.read_text()); proofs[name] = sha(proof)
                if proofs[name] != r['compiler_sha256'] or r['devices'] != 4 or r['method'] != 'polynomial':
                    raise ValueError('candidate compiler/backend receipt mismatch')
                for q in (p, p.with_suffix('.npz')):
                    files[str(q.relative_to(run))] = sha(q)
                records.append(r)
    files.update(proofs)
    return dict(passed=True, identity=identity, n=n, records=len(records), owners=records[0]['owners'],
        files=files, constant_error=max(r['constant_error'] for r in records),
        source_pair_budget=max(r['source_pair_budget'] for r in records),
        scientific_order_accepted=False, production_promoted=False)


def run_grid(run, old, identity, n, host_gib):
    run, old = Path(run), Path(old)
    if (run/f'gpu_N{n}.json').exists():
        checked = validate_grid(run, identity, n)
        if read(run/f'gpu_N{n}.json')['validation'] != checked:
            raise ValueError('saved grid summary differs from independent validation')
        return checked
    tick = time.perf_counter()
    devices, inventory = oldgpu.device_inventory(False)
    _, estimate = oldgpu.estimate_merge(old, n, BASE_ID, host_gib)
    # The previous replay measured up to 2.06x its declared host estimate.
    # Add explicit empirical margin and independent-reference merge allowance.
    required = 3*estimate['estimated_host_peak_upper_bytes']+8*2**30
    available = min(int(host_gib*2**30), estimate.get('observed_host_available_bytes', 2**63))
    if required > available:
        raise MemoryError(f'calibrated host guard needs {required/2**30:.2f} GiB; budget/available {available/2**30:.2f}')
    oldgpu._device_guard(estimate, inventory)
    root = run/'science'/f'N{n}'; root.mkdir(parents=True, exist_ok=True)
    mesh = Mesh(np.asarray(devices[:4], object), ('z',))
    bank, geometry, views, _ = oldgpu.merge_chunks(old, n, BASE_ID, host_gib)
    del views
    O, R, volume, radial = load_references(run, old, n, identity)
    plan_metadata = read(old/'inputs/plan.json')[str(n)]
    last = plan_metadata['last_aggregate']
    api = CachedAPI(c, optimize=True)
    api.cache.max_bytes = 2*estimate['one_case_boundary_bytes']
    boundary_audit = catalogue_replay(bank, c)
    state = np.load(old/'data'/f'N{n}'/'state.npy', mmap_mode='r')
    phi = np.load(old/'data'/f'N{n}'/'phi.npy', mmap_mode='r')
    topology = np.load(old/'data'/f'N{n}'/'raw_to_owner.npy', mmap_mode='r')
    if state.shape != (22, 6, len(bank.owners)) or phi.shape != (22, len(bank.owners)):
        raise ValueError('full owner input state coverage')
    proofs = {}; timings = dict(boundary=0., call=0., reduction=0., transfer=0.)
    for ai, span in enumerate(SPANS):
        plan = oldgpu.lower_q_plan(bank, diffusion_span=span, **geometry)
        stacked = oldgpu.shard_q_plan(plan, bank, topology, 4)
        placed = _sync(_place_plan(stacked, mesh)); del stacked
        for ki, (kinds, pk) in enumerate(c.KINDS):
            def stage(qp, x, bi, bo, p, pb, co, source):
                a = oldgpu.sharded_q_rhs(qp, x, bi, bo, p, pb, co, kinds=kinds,
                    phi_kind=pk, tau=c.TAU, mu=c.MU, mesh=mesh, characteristic_method='polynomial')
                return a, a.combined+source
            call = jax.jit(stage)
            proof = root/f'compiler_s{int(1/span)}_k{ki}.txt'
            proof_hash = None
            if proof.exists():
                compiler_guard(proof.read_text()); proof_hash = sha(proof)
                proofs[str(proof.relative_to(run))] = proof_hash
            for ci in range(22):
                path = root/f's{int(1/span)}_k{ki}_c{ci:02d}.json'
                if record_valid(path, identity, n, span, ki, ci):
                    continue
                start = time.perf_counter(); bi, bo, pb = api.boundaries(bank, ci)
                timings['boundary'] += time.perf_counter()-start
                # Steady MMS source S=-R uses the same assembled stage call.
                source = -np.asarray(R[ci, :, 18:24])
                start = time.perf_counter()
                args = jax.tree.map(lambda a: jax.device_put(a, NamedSharding(mesh, P())),
                    (np.asarray(state[ci]), bi, bo, np.asarray(phi[ci]), pb, c.COEFF, source))
                _sync(args); timings['transfer'] += time.perf_counter()-start
                start = time.perf_counter(); a, paired = _sync(call(placed, *args))
                timings['call'] += time.perf_counter()-start
                if proof_hash is None:
                    text = call.lower(placed, *args).compile().as_text()
                    compiler_guard(text); proof.write_text(text)
                    proof_hash = sha(proof)
                    proofs[str(proof.relative_to(run))] = proof_hash
                start = time.perf_counter(); N = numerical(bank, a)
                paired = np.asarray(paired)
                expected = N[:, 18:24]-R[ci, :, 18:24]
                pair_budget = float(np.max(abs(paired-expected)/(1e-8+1e-11*abs(expected))))
                if pair_budget > 1 or not np.isfinite(pair_budget):
                    raise ValueError('assembled source-pair identity gate')
                constant = float(abs(N-O[ai, ci]).max()) if ci == 0 else 0.
                if constant > 1e-7:
                    raise ValueError('constant N-O roundoff gate')
                stats = reduce_case(N, O[ai, ci], R[ci], bank.owners, volume, radial, n, last)
                atomic_npz(path.with_suffix('.npz'), **stats)
                timings['reduction'] += time.perf_counter()-start
                write(path, dict(passed=True, identity=identity, n=n, span=span, kind=ki, case=ci,
                    owners=len(bank.owners), raw=len(bank.raw), sha256=sha(path.with_suffix('.npz')),
                    constant_error=constant, source_pair_budget=pair_budget, compiler=str(proof.relative_to(run)),
                    compiler_sha256=proof_hash, method='polynomial', devices=4))
                print(f'GPU N{n}: h/{int(1/span)} BC{ki} case {ci+1}/22', flush=True)
                del a, args, paired, N, bi, bo, pb
            del call
        del plan, placed
        gc.collect()
    result = dict(passed=True, identity=identity, n=n, validation=validate_grid(run, identity, n),
        device_inventory=inventory, boundary_audit=boundary_audit,
        host_guard_bytes=required, host_budget_gib=host_gib, host_peak_rss_gib=oldgpu.peak_rss_gib(),
        device_memory=memory_snapshot(devices), timings=timings, wall_seconds=time.perf_counter()-tick,
        compiler_proofs=proofs, source='steady manufactured S=-R; same assembled GPU stage',
        scope='global static MMS, no time integration or physical sheath/SAT qualification')
    write(run/f'gpu_N{n}.json', result)
    del O, R
    for name in ('O.npy', 'R.npy'):
        (run/'tmp'/f'N{n}'/name).unlink()
    return result
