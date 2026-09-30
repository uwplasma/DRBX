"""Orders and candidate/C0 ratios over the grids present (``results_N{n}.json`` written by ``run.py``).

    python -m p08_inner_support_eval.analyze --dir DIR [--summary PATH]   # writes summary.json and prints the pooled tables

(The candidate is mode key ``C1`` in every results file, whatever its campaign name.)

Nothing is interpreted beyond flags: rebound (error larger on the finer grid), order < 2, dispatch violations, failures and
residual/rank numbers.  Orders are ``ln(e_coarse / e_fine) / ln(n_fine / n_coarse)`` between consecutive grids present;
they are computed for roles, tracks and the pooled groups (ring groups only carry values: a ring index is not the same
place on different grids).  The short waves are a response report only and appear in no order.
"""
import argparse
import json
import math
from pathlib import Path

ORDER_GROUP_KINDS = ('role', 'track', 'pool')
AUDIT_ROLES = ('coupled_control', 'switch', 'switch_plus1', 'switch_plus2', 'last_aggregate', 'first_singleton')
POOLS = ('pool:inner', 'pool:extended')
# metric key -> accessor(mode_group entry) -> list (per field) or scalar; None when absent
METRICS = {
    'a_cell.value.rms': lambda g: g['a_cell']['value']['rms'],
    'a_cell.transverse.rms': lambda g: g['a_cell']['transverse']['rms'],
    'a_cell.combined.rms': lambda g: g['a_cell']['combined']['rms'],
    'a_cell.transverse.max': lambda g: g['a_cell']['transverse']['max'],
    'a_face.transverse.rms': lambda g: g['a_face']['transverse']['rms'],
    'a_face.value.rms': lambda g: g['a_face']['value']['rms'],
    'b_p07.N-O.rms': lambda g: g['b_p07']['N-O']['rms'],
    'b_p07.O-R.rms': lambda g: g['b_p07']['O-R']['rms'],
    'b_p07.N-R.rms': lambda g: g['b_p07']['N-R']['rms'],
    'b_p07.N-O.max': lambda g: g['b_p07']['N-O']['max'],
    'c_p06.owner_rms': lambda g: g['c_p06']['owner_rms'],
}


def safe(entry, fn):
    try:
        return fn(entry)
    except (KeyError, TypeError):
        return None


def listify(x):
    return None if x is None else (list(x) if isinstance(x, (list, tuple)) else [x])


def order(e_coarse, e_fine, n_coarse, n_fine):
    if e_coarse is None or e_fine is None or e_coarse <= 0 or e_fine <= 0:
        return None
    return math.log(e_coarse / e_fine) / math.log(n_fine / n_coarse)


def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def series(results, mode, gid, key):
    """{n: per-field list} of one metric over the grids present."""
    out = {}
    for n, r in results.items():
        g = r['modes'][mode]['groups'].get(gid)
        v = listify(safe(g, METRICS[key])) if g else None
        if v is not None:
            out[n] = v
    return out


def elementwise(fn, a, b):
    return [fn(x, y) for x, y in zip(a, b)]


def main(directory, summary_path=None):
    """``summary_path`` defaults to ``<directory>/summary.json``."""
    results = {}
    for path in sorted(Path(directory).glob('results_N*.json')):
        r = json.loads(path.read_text())
        results[r['n']] = r
    grids = sorted(results)
    if not grids:
        raise SystemExit(f'no results_N*.json in {directory}')
    fields = results[grids[0]]['fields']
    modes = list(results[grids[0]]['modes'])
    out = dict(grids=grids, fields=fields, modes=modes, last={n: results[n]['last'] for n in grids},
               intervals=[f'{a}-{b}' for a, b in zip(grids, grids[1:])], orders={}, ratios={}, values={}, flags={})

    group_ids = sorted({gid for r in results.values() for m in r['modes'].values() for gid in m['groups']})
    for mode in modes:
        for gid in group_ids:
            kind = gid.split(':')[0]
            for key in METRICS:
                ser = series(results, mode, gid, key)
                if not ser:
                    continue
                out['values'].setdefault(mode, {}).setdefault(gid, {})[key] = {str(n): v for n, v in ser.items()}
                if kind in ORDER_GROUP_KINDS and len(ser) > 1:
                    ns = sorted(ser)
                    ords = {f'{a}-{b}': elementwise(lambda x, y: order(x, y, a, b), ser[a], ser[b])
                            for a, b in zip(ns, ns[1:])}
                    out['orders'].setdefault(mode, {}).setdefault(gid, {})[key] = ords
    if len(modes) == 2:
        c0, c1 = modes
        for gid in group_ids:
            for key in METRICS:
                s0, s1 = series(results, c0, gid, key), series(results, c1, gid, key)
                if s0 and s1:
                    out['ratios'].setdefault(gid, {})[key] = {
                        str(n): elementwise(ratio, s1[n], s0[n]) for n in sorted(set(s0) & set(s1))}

    # ---- flags (no interpretation) -------------------------------------------------------------------------------
    flags = out['flags']
    rebound = []
    for mode in modes:
        for gid in group_ids:
            if not (gid.startswith('track:') or gid.split(':', 1)[-1] in AUDIT_ROLES):
                continue
            for key in ('a_cell.transverse.rms', 'b_p07.N-O.rms'):
                ser = series(results, mode, gid, key)
                ns = sorted(ser)
                for a, b in zip(ns, ns[1:]):
                    bad = [fields[i] for i, (x, y) in enumerate(zip(ser[a], ser[b])) if y > x]
                    if bad:
                        rebound.append(dict(mode=mode, group=gid, metric=key, interval=f'{a}-{b}', n_fields=len(bad),
                                            fields=bad))
    flags['rebound'] = rebound
    low = []
    for mode in modes:
        for pool in POOLS:
            for key in ('a_cell.transverse.rms', 'b_p07.N-O.rms'):
                ords = out['orders'].get(mode, {}).get(pool, {}).get(key, {})
                for interval, vals in ords.items():
                    bad = [dict(field=fields[i], order=v) for i, v in enumerate(vals) if v is None or v < 2.0]
                    if bad:
                        low.append(dict(mode=mode, group=pool, metric=key, interval=interval, n_fields=len(bad), fields=bad))
    flags['order_below_2'] = low
    flags['dispatch_violations'] = {n: results[n]['dispatch_check']['n_violations'] for n in grids}
    flags['dispatch_counts'] = {n: dict(identical=results[n]['dispatch_check']['identical'],
                                        changed=results[n]['dispatch_check']['changed']) for n in grids}
    flags['build_failures'] = {n: results[n]['failures'] for n in grids if results[n]['failures']}
    flags['rows'] = {m: {n: {k: (None if v is None else dict(coupled_fraction=v['coupled_fraction'],
                                                              donors_mean=v['donors_mean'], donors_max=v['donors_max'],
                                                              min_rank=v.get('min_rank'),
                                                              max_residual=v.get('max_residual')))
                             for k, v in results[n]['modes'][m]['rows_all'].items()} for n in grids} for m in modes}
    flags['pinv'] = {m: {n: results[n]['modes'][m]['pinv'] for n in grids} for m in modes}
    flags['residual_above_1e-9'] = [dict(n=n, mode=m, kind=k, max_residual=v['max_residual'])
                                    for n in grids for m in modes
                                    for k, v in results[n]['modes'][m]['rows_all'].items()
                                    if v and v.get('max_residual') is not None and v['max_residual'] > 1e-9]

    # short-wave response passthrough (roles, rings, pools), compact
    out['short_waves'] = {m: {str(n): {gid: results[n]['modes'][m]['groups'][gid]['short_waves']
                                       for gid in results[n]['modes'][m]['groups']}
                              for n in grids} for m in modes}
    summary_path = Path(summary_path) if summary_path is not None else Path(directory) / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out))
    print_report(out, results)
    return out


def fmt(x, digits=2):
    return '  n/a' if x is None else f'{x:.{digits}f}'


def print_report(out, results):
    fields, modes, grids = out['fields'], out['modes'], out['grids']
    print(f"grids {grids}, last ring {out['last']}, modes {modes}")
    print('dispatch violations', out['flags']['dispatch_violations'], '| build failures', out['flags']['build_failures'])
    print('residual > 1e-9:', out['flags']['residual_above_1e-9'])
    for n in grids:
        for m in modes:
            rows = out['flags']['rows'][m][n]
            print(f"  N{n} {m}: coupled fraction " + ', '.join(f"{k} {v['coupled_fraction']:.3f}" for k, v in rows.items() if v)
                  + ' | donors mean ' + ', '.join(f"{k} {v['donors_mean']:.0f}" for k, v in rows.items() if v))
    for pool in POOLS:
        for key in ('a_cell.transverse.rms', 'b_p07.N-O.rms'):
            print(f'\n{pool}  {key}: per-field order ' + ' / '.join(out['intervals']) + ' and C1/C0 ratio per grid')
            for i, name in enumerate(fields):
                cells = []
                for m in modes:
                    ords = out['orders'].get(m, {}).get(pool, {}).get(key, {})
                    cells.append(m + ' ' + ' '.join(fmt(ords[iv][i]) for iv in out['intervals'] if iv in ords))
                ratios = out['ratios'].get(pool, {}).get(key, {})
                cells.append('C1/C0 ' + ' '.join(fmt(ratios[str(n)][i], 3) for n in grids if str(n) in ratios))
                print(f'  {name:18s} ' + ' | '.join(cells))
    if out['flags']['rebound']:
        print(f"\nrebound flags: {len(out['flags']['rebound'])} (see summary.json flags.rebound)")
    else:
        print('\nrebound flags: none')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', required=True)
    ap.add_argument('--summary', default=None, help='default: <dir>/summary.json')
    args = ap.parse_args()
    main(args.dir, args.summary)
