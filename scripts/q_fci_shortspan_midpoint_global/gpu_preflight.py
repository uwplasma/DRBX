"""Bounded CPU/GPU and RK4-step comparisons on the frozen preflight sample.

Run on CPUs after GPU preflight and its CPU reconstruction. This is a numerical
equivalence check, not a scaling study. All results are raw-member comparisons;
global norms still come from the complete-owner numerical campaign.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path

from . import numerical_runner as nr
from . import trace_store as store
import numpy as np


def scaled_defect(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise RuntimeError('comparison has invalid shape or nonfinite actions')
    return float(np.max(np.abs(a-b)/(1+np.abs(b)), initial=0))


def compare_trace(a, b, same_steps):
    for i in (1, 2, 3):
        if not np.array_equal(a[i], b[i]):
            raise RuntimeError('CPU/GPU or step-refinement trace classification differs')
    if same_steps and not np.array_equal(a[6], b[6]):
        raise RuntimeError('CPU/GPU first-crossing step differs')
    # Compare endpoints in regular x/y to respect the theta seam.
    def xyz(q):
        return np.column_stack((q[:, 0]*np.cos(q[:, 1]), q[:, 0]*np.sin(q[:, 1]), q[:, 2]))
    value = scaled_defect(xyz(a[0]), xyz(b[0]))
    if value > 1e-9:
        raise RuntimeError(f'trace endpoint scaled difference {value} exceeds 1e-9')
    return value


def actions(raw, seeds, ends):
    ijk, q, _ = nr.raw_geometry(raw)
    owner = int(nr.INDEX.labels[raw])
    kinds = ('D', 'N') if ijk[0] >= nr.N-7 else ('interior',)
    numerical, exact, selected = [], [], []
    for kind in kinds:
        nr.INDEX.begin_target()
        data, choice, _, _, _ = nr.fit(q, ijk, seeds, ends, owner, kind)
        comp = nr.compose(nr.CTX, seeds, q, data, kind, owner)
        selected.append(choice)
        normals = nr.normal_contravariant(nr.CTX, comp['wall_nodes']) if kind != 'interior' else None
        for name in nr.FIELDS:
            if kind != 'interior' and kind not in nr.PAIRS.get(name, nr.PAIRS['waves']):
                continue
            field = nr.callable_field(name)
            boundary = np.empty(0)
            if kind != 'interior':
                value, grad = field(comp['wall_nodes'])
                boundary = value if kind == 'D' else np.einsum('ni,ni->n', normals, grad)
            numerical.append(nr.apply(comp, nr.STATE[name][comp['donors']],
                                      nr.STATE[name][owner], boundary, kind))
            exact.append((comp['K'] @ field(ends.reshape(-1, 3))[0]).reshape(2, 3))
    return np.asarray(numerical), np.asarray(exact), selected


def init(n, input_root, output, screen):
    nr.init_worker(n, input_root, output, screen)
    nr.TRACER = nr.JaxHsxMagneticField.from_evaluators(nr.CTX['evaluator'], nr.CTX['bfield'])


def check_raw(raw):
    ijk, q, width = nr.raw_geometry(raw)
    seeds = nr.seeds_for(q, width, ijk[0] >= nr.N-7)
    gpu, _ = nr.TRACE_CACHE.get(raw, seeds)
    legs, deltas = store.pack_legs(seeds[None], nr.N)
    primary_steps = nr.DESIGN['method']['rk4_steps']
    primary = nr.trace_padded(nr.TRACER, legs, deltas, primary_steps)
    reference = nr.trace_padded(nr.TRACER, legs, deltas, 256) if primary_steps != 256 else primary
    for output in (primary, reference):
        store.validate_trace(output, nr.N)
    parity = compare_trace(gpu, primary, same_steps=True)
    sensitivity = compare_trace(primary, reference, same_steps=primary_steps == 256)
    ng, eg, sg = actions(raw, seeds, gpu[0].reshape(4, 12, 3))
    nc, ec, sc = actions(raw, seeds, primary[0].reshape(4, 12, 3))
    nh, eh, sh = actions(raw, seeds, reference[0].reshape(4, 12, 3)) if primary_steps != 256 else (nc, ec, sc)
    defects = dict(gpu_N=scaled_defect(ng, nc), gpu_E=scaled_defect(eg, ec),
                   step_N=scaled_defect(nc, nh), step_E=scaled_defect(ec, eh))
    passed = (max(defects['gpu_N'], defects['gpu_E']) <= 1e-8
              and max(defects['step_N'], defects['step_E']) <= 1e-7)
    return dict(raw=int(raw), owner=int(nr.INDEX.labels[raw]), passed=passed,
                endpoint_gpu_scaled=parity, endpoint_step_scaled=sensitivity,
                action_scaled_defects=defects, gpu_selected=sg, cpu_selected=sc,
                reference_selected=sh, primary_steps=primary_steps, reference_steps=256)


def run(campaign, n, input_root, screen, workers, host_gib, worker_gib):
    output = Path(campaign)/'preflight'
    design = nr.validate(input_root, output, screen)
    if design.get('trace_mode') != 'gpu_cache':
        raise RuntimeError('GPU preflight required')
    policy = nr.worker_policy(workers, host_gib, worker_gib)
    sample = design['rk4_512_raw_ids'][str(n)]
    if not sample:
        raise RuntimeError('missing frozen preflight comparison sample')
    if workers == 1:
        init(n, input_root, output, screen)
        rows = [check_raw(raw) for raw in sample]
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(workers, len(sample)),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=init, initargs=(n, input_root, output, screen)) as pool:
            rows = list(pool.map(check_raw, sample))
    result = dict(N=n, passed=all(r['passed'] for r in rows), rows=rows,
                  design_sha256=nr.sha(output/'design.json'),
                  trace_inventory_sha256=nr.sha(output/'traces'/f'complete_N{n}.json'),
                  worker_policy=policy, effective_workers=min(workers, len(sample)),
                  thresholds=dict(endpoint_scaled=1e-9, gpu_action_scaled=1e-8, step_action_scaled=1e-7),
                  interpretation='bounded raw-member numerical equivalence, not global convergence')
    nr.write(output/f'gpu_cpu_step_check_N{n}.json', result)
    if not result['passed']:
        raise RuntimeError('GPU/CPU or RK4-step action check failed; preserve receipt for local review')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--N', type=int, choices=(32, 48, 64), required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--exact-screen', required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--host-memory-gib', type=float, required=True)
    parser.add_argument('--worker-memory-gib', type=float, required=True)
    args = parser.parse_args()
    result = run(args.campaign, args.N, args.input_root, args.exact_screen,
                 args.workers, args.host_memory_gib, args.worker_memory_gib)
    print(json.dumps(dict(N=args.N, passed=result['passed'], samples=len(result['rows']))))


if __name__ == '__main__':
    main()
