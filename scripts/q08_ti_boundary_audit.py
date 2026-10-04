"""Read-only, CPU boundary-producer audit against a frozen Q08 Ti campaign.

No operator actions, GPU allocation, tracing, reference regeneration or changed
gates. A completed diagnostic never authorizes campaign continuation.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np

ARRAYS = ('wall_value', 'slot_value', 'slot_tangent', 'wall_normal')


def difference_summary(actual, expected, wall, multiplier):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.ndim not in (3, 4) or
            actual.shape[0] != 1 or actual.dtype != np.float64 or expected.dtype != np.float64):
        raise ValueError('boundary layout/float64 mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('nonfinite boundary array')
    wall = np.asarray(wall)
    if (wall.ndim != 1 or not np.issubdtype(wall.dtype, np.integer) or
            len(np.unique(wall)) != len(wall) or np.any((wall < 0) | (wall >= actual.shape[1]))):
        raise ValueError('invalid wall index')
    nonwall = np.ones(actual.shape[1], dtype=bool); nonwall[wall] = False
    error = abs(actual-expected)
    scale = np.maximum(1., abs(expected))
    budget = multiplier*np.finfo(np.float64).eps*scale
    fractions = error/budget
    ix = tuple(int(v) for v in np.unravel_index(np.argmax(fractions), fractions.shape))
    return dict(max_abs=float(error.max()), max_budget_fraction=float(fractions[ix]),
                differences=int(np.count_nonzero(error)), over_budget=int(np.count_nonzero(fractions > 1)),
                nonwall_exact=bool(np.array_equal(actual[:, nonwall], expected[:, nonwall])),
                worst=dict(index=list(ix), actual=float(actual[ix]), expected=float(expected[ix]),
                           actual_hex=float(actual[ix]).hex(), expected_hex=float(expected[ix]).hex(),
                           abs_error=float(error[ix]), budget=float(budget[ix])))


def query_context(bank, case, array_index, index, common, optimized_fields):
    """Map the worst entry to its actual saved query and normal-contraction terms."""
    raw_position, node = index[1:3]
    walls = np.flatnonzero(np.asarray(bank.wall_index) == raw_position)
    result = dict(raw_position=raw_position, raw_id=int(bank.raw[raw_position]),
                  active_wall=bool(len(walls)))
    if not len(walls):
        return result
    w = int(walls[0])
    if array_index in (0, 3):
        query = int(bank.wall_node_query[w, node])
    else:
        query = int(bank.wall_slot_query[w, common.SPAN_SLOTS[1][node]])
    point = np.asarray(bank.query_table[query])
    result.update(wall_position=w, query_id=query, point=point.tolist())
    if array_index == 2:
        result['coordinate_derivative'] = ('theta', 'eta')[index[3]]
    if array_index == 3:
        # Diagnostic pointwise probes deliberately expose another batch shape.
        # Save them without replacing either producer or the historical fixture.
        normal = np.asarray(bank.boundary_wall_normal[w, node])
        _, all_gradient = common.fields(point[None])
        _, one_gradient = optimized_fields(point[None], case, common)
        g_old = all_gradient[0, case, 2]; g_new = one_gradient[0, 2]
        result.update(normal=normal.tolist(),
            pointwise_original_gradient=g_old.tolist(), pointwise_optimized_gradient=g_new.tolist(),
            original_normal_products=(normal*g_old).tolist(),
            optimized_normal_products=(normal*g_new).tolist(),
            original_sum_abs_products=float(np.sum(abs(normal*g_old))),
            optimized_sum_abs_products=float(np.sum(abs(normal*g_new))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('campaign-source', 'baseline-source', 'inputs', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--expected-identity', required=True)
    a = p.parse_args()
    source, baseline, inputs, out = (v.resolve() for v in
                                    (a.campaign_source, a.baseline_source, a.inputs, a.output))
    for protected in (source, baseline, inputs):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            p.error('output must be disjoint from immutable source/input directories')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error('use a new empty diagnostic output directory')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.parents[1]))
    from scripts.q08_rhs_mms_global import campaign as control
    if Path(control.__file__).resolve().parent != source:
        raise ValueError('wrong frozen campaign import')
    identity, _ = control.check_source()
    if identity != a.expected_identity:
        raise ValueError('unexpected frozen campaign identity')
    control.load(baseline, out, gpu=False)
    input_manifest = control.read(baseline/'input_manifest.json')['files']
    from scripts.q08_rhs_mms_global import ti_replay as replay
    from scripts.q08_extraction_global import common as c
    from drbx.stencils.q_parallel import load_chunk
    from drbx.stencils.q_bank import build_q_bank
    from boundary_values import compact_boundaries, fields as optimized_fields
    from boundary_cache import restore
    import jax
    if jax.default_backend() != 'cpu':
        raise ValueError('boundary diagnostic must use CPU')
    tick = time.perf_counter(); records = []; provenance = {}
    for n in (32, 48, 64):
        files = [inputs/'bounded'/f'N{n}_h{d}.npz' for d in (16, 32)]
        fixture = inputs/'bounded'/f'N{n}_inputs.npz'
        for path in (*files, fixture):
            provenance[str(path)] = control.sha(path)
            if provenance[str(path)] != input_manifest[str(path.relative_to(inputs))]:
                raise ValueError('bounded input differs from frozen baseline: '+str(path))
        bank = build_q_bank(*(load_chunk(path) for path in files))
        with np.load(fixture, allow_pickle=False) as z:
            saved = [z[f'inner_bc_{i}'][:, 2:3] for i in range(4)]
        compact_all = [[] for _ in ARRAYS]; original_all = [[] for _ in ARRAYS]
        for case, design in enumerate(c.DESIGNS):
            compact = replay.scalar_boundary(bank, case)
            original = [v[2:3] for v in c.boundaries(bank, case)[0]]
            full_compact = restore(bank.wall_index, compact_boundaries(bank, case, c))[0]
            for ai, name in enumerate(ARRAYS):
                # Lossless slicing is tested separately from analytic re-evaluation.
                np.testing.assert_array_equal(compact[ai], full_compact[ai][2:3])
                compact_all[ai].append(compact[ai]); original_all[ai].append(original[ai])
                try:
                    replay.check_boundary_fixture(compact[ai], saved[ai][case], bank.wall_index)
                    gate = dict(passed=True)
                except (AssertionError, ValueError) as error:
                    gate = dict(passed=False, error=str(error))
                record = dict(n=n, case=case, case_name=design['name'], array=name, current_gate=gate)
                pairs = (('compact_saved', compact[ai], saved[ai][case]),
                         ('original_saved', original[ai], saved[ai][case]),
                         ('compact_original', compact[ai], original[ai]))
                for label, actual, expected in pairs:
                    stats = difference_summary(actual, expected, bank.wall_index, replay.BOUNDARY_EPS_MULTIPLIER)
                    stats['worst']['context'] = query_context(bank, case, ai, stats['worst']['index'], c, optimized_fields)
                    record[label] = stats
                records.append(record)
        np.savez_compressed(out/f'boundary_N{n}.npz',
            **{f'saved_{i}': v for i, v in enumerate(saved)},
            **{f'compact_{i}': np.stack(v) for i, v in enumerate(compact_all)},
            **{f'original_{i}': np.stack(v) for i, v in enumerate(original_all)},
            raw=bank.raw, wall_index=bank.wall_index, wall_node_query=bank.wall_node_query,
            wall_slot_query=bank.wall_slot_query, query_table=bank.query_table,
            boundary_wall_normal=bank.boundary_wall_normal)
        # Retain completed resolutions if later input/structure validation fails.
        control.write(out/f'records_N{n}.json', [r for r in records if r['n'] == n])
        print(f'Boundary-only N{n}: all 22 states and four arrays recorded', flush=True)
    worst = sorted(records, key=lambda r: r['compact_saved']['max_budget_fraction'], reverse=True)[:12]
    summary = dict(completed=True, allows_campaign_resume=False, identity=identity,
        script_sha256=control.sha(Path(__file__)), input_hashes=provenance,
        campaign_source=str(source), baseline_source=str(baseline), inputs=str(inputs),
        python=sys.version, numpy=np.__version__, backend=jax.default_backend(),
        eps_multiplier=replay.BOUNDARY_EPS_MULTIPLIER, seconds=time.perf_counter()-tick,
        arrays_checked=len(records), failed_arrays=sum(not r['current_gate']['passed'] for r in records),
        lossless_Ti_slice_exact=True, worst=worst,
        scope='Boundary producer/fixture diagnostic only; no N/O/R actions or changed gates')
    control.write(out/'diagnostic.json', summary)
    control.write(out/'records.json', records)
    files = [f for f in out.iterdir() if f.is_file()]
    control.write(out/'receipt.json', dict(completed=True, identity=identity,
        files={f.name:control.sha(f) for f in files}, allows_campaign_resume=False))
    print(json.dumps({k: summary[k] for k in ('completed','arrays_checked','failed_arrays','seconds')}))


if __name__ == '__main__':
    main()
