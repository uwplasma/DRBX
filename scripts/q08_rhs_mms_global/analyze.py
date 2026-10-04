"""Deterministic N-O-R reduction; scientific interpretation remains local."""
import csv
import io
from pathlib import Path
import numpy as np
from scripts.q08_rhs_mms_global.campaign import NS, read, write, sha, require, MUTABLE_PROVENANCE
from scripts.q08_rhs_mms_global.science import SPANS, TERMS, REGIONS, METRICS
from scripts.q08_extraction_global import common as c


def collect(run, identity):
    from scripts.q08_rhs_mms_global.gpu import validate_grid
    grids = []
    all_stats = {}
    for n in NS:
        require(run, f'references_N{n}', identity)
        checked = validate_grid(run, identity, n)
        receipt = read(run/f'gpu_N{n}.json')
        if receipt['identity'] != identity or receipt['validation'] != checked:
            raise ValueError('grid completion changed')
        grids.append(checked)
        entries = []
        for span in SPANS:
            kinds = []
            for ki in range(4):
                cases = []
                for ci in range(22):
                    p = run/'science'/f'N{n}'/f's{int(1/span)}_k{ki}_c{ci:02d}.npz'
                    with np.load(p) as z:
                        cases.append({k: z[k].copy() for k in z.files})
                kinds.append({k: np.stack([a[k] for a in cases]) for k in cases[0]})
            entries.append({k: np.stack([a[k] for a in kinds]) for k in kinds[0]})
        all_stats[n] = {k: np.stack([a[k] for a in entries]) for k in entries[0]}
    arrays = {k: np.stack([all_stats[n][k] for n in NS]) for k in all_stats[32]}
    return grids, arrays


def products(run, identity):
    grids, s = collect(run, identity)
    # resolution, span, BC, state, region, metric, term
    volume = s['volume'][..., None, None]
    rms = np.sqrt(s['sum2']/volume)
    reference_rms = np.sqrt(s['reference_sum2']/s['volume'][..., None])
    relative = np.full_like(rms, np.nan)
    np.divide(rms, reference_rms[..., None, :], out=relative,
              where=reference_rms[..., None, :] > 1e-30)
    orders = np.full_like(rms[:2], np.nan)
    for i, ratio in enumerate((48/32, 64/48)):
        good = (rms[i] > 1e-30)&(rms[i+1] > 1e-30)
        np.log(np.divide(rms[i], rms[i+1], out=np.ones_like(rms[i]), where=good), out=orders[i], where=good)
        orders[i] /= np.log(ratio)
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(['span','BC','case','region','metric','term','rms32','rms48','rms64',
                'p32_48','p48_64','relative32','relative48','relative64',
                'max32','max48','max64','owner32','owner48','owner64','signed32','signed48','signed64'])
    for ai, span in enumerate(SPANS):
        for ki, (kinds, pk) in enumerate(c.KINDS):
            for ci, case in enumerate(c.DESIGNS):
                for ri, region in enumerate(REGIONS):
                    for mi, metric in enumerate(METRICS):
                        for ti, term in enumerate(TERMS):
                            ix = (slice(None), ai, ki, ci, ri, mi, ti)
                            numbers = [*rms[ix], *orders[ix], *relative[ix],
                                *s['maximum'][ix], *s['max_owner'][ix], *s['signed'][ix]]
                            w.writerow([span,''.join(kinds)+'/phi'+pk,case['name'],region,metric,term,
                                        *[v if np.isfinite(v) else '' for v in numbers]])
    lines = ['# Q08 six-field static MMS', '',
        f'Campaign identity: `{identity}`', '',
        'N is the unchanged extracted GPU six-field RHS. O uses exact values and',
        'gradients at its saved C3 RK4-64 stencil points. R is the analytic center',
        'continuum action with a checked coordinate flux derivative for diffusion.',
        'All are projected from raw centers to complete owners with the same weights.',
        'No integrated cell reference, time stepping or new tracing is used.', '',
        'All 22 states, four D/N/mixed patterns, both diffusion spans; material h/32',
        'and outer correction samples h/16. The 31 terms are scored individually.',
        'O and R are independent of the imposed D/N representation for the same field.',
        'The steady source S=-R is added through the same assembled GPU stage;',
        'its residual equals N-R within the unchanged implementation replay budget.', '',
        'Tables use physical-volume weighted complete-owner RMS, maxima with owner',
        'IDs, signed integrals and relative RMS. Regions overlap intentionally.',
        'Undefined zero-error orders and zero-reference relative errors are blank.',
        'No order failures are converted into execution failures or tuned away.', '',
        '| smooth global term (diffusion h/32) | BC | metric | RMS32 | RMS48 | RMS64 | p32–48 | p48–64 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for ti in range(18, 24):
        for ki in (0, 1):
            for mi, metric in enumerate(METRICS):
                ix = (slice(None), 1, ki, 1, 0, mi, ti)
                values = [*rms[ix], *orders[ix]]
                lines.append('| '+TERMS[ti]+' | '+('D' if ki == 0 else 'N')+' | '+metric+' | '+
                    ' | '.join(f'{v:.6g}' if np.isfinite(v) else '—' for v in values)+' |')
    lines += ['', 'Reference sensitivity (max absolute change over the two finer reference',
              'steps, across all owners and states) is retained per term in analysis.json.',
              'The default target is never silently replaced by the most favorable step.', '',
              'Execution success establishes coverage and replay, not scientific acceptance.',
              'Physical grazing-sheath/SAT, wall power, exterior crossings, evolved stability,',
              'span/support selection, perpendicular coupling and production promotion remain open.']
    summary = dict(passed=True, identity=identity, grids=grids, resolutions=list(NS),
        spans=list(SPANS), terms=list(TERMS), regions=list(REGIONS), metrics=list(METRICS),
        cases=c.DESIGNS, kinds=c.KINDS,
        axes='resolution (or interval), span, BC, case, region, metric, term',
        norm='physical-volume weighted complete-owner RMS',
        reference_sensitivity={str(n):read(run/f'references_N{n}.json')['reference_step_max_by_term'] for n in NS},
        scientific_order_accepted=False, production_promoted=False)
    return summary, {**s, 'rms':rms, 'orders':orders, 'relative_rms':relative,
                     'reference_rms':reference_rms}, out.getvalue(), '\n'.join(lines)+'\n'


def analyze(run, identity):
    from scripts.q08_extraction_global.common import atomic_npz
    summary, arrays, table, report = products(run, identity)
    atomic_npz(run/'totals.npz', **arrays)
    (run/'orders.csv').write_bytes(table.encode('utf-8'))
    (run/'report.md').write_bytes(report.encode('utf-8'))
    write(run/'analysis.json', summary)
    write(run/'analysis_receipt.json', dict(passed=True, identity=identity,
        files={n:sha(run/n) for n in ('totals.npz','orders.csv','report.md','analysis.json')}))
    return summary


def completion(run, identity):
    summary, arrays, table, report = products(run, identity)
    if read(run/'analysis.json') != read_json_value(summary):
        raise ValueError('analysis summary mismatch')
    with np.load(run/'totals.npz') as z:
        if set(z.files) != set(arrays) or any(not np.array_equal(z[k], v, equal_nan=True) for k, v in arrays.items()):
            raise ValueError('independent completion reduction mismatch')
    # Compare the actual UTF-8 serialization; read_text normalizes CRLF from
    # csv.writer to LF, making a correctly written table fail validation.
    if (run/'orders.csv').read_bytes() != table.encode('utf-8') or (run/'report.md').read_bytes() != report.encode('utf-8'):
        raise ValueError('tables/report independent reduction mismatch')
    require(run, 'analysis_receipt', identity)
    files = {str(p.relative_to(run)):sha(p) for folder in
        ('science','references','science_source','source_snapshot','provenance')
        for p in (run/folder).rglob('*') if p.is_file() and str(p.relative_to(run)) not in MUTABLE_PROVENANCE}
    names = ('verification','binding','preflight','preflight_gpu','pilot','analysis','analysis_receipt',
             *(f'references_N{n}' for n in NS), *(f'gpu_N{n}' for n in NS))
    files.update({name+'.json':sha(run/(name+'.json')) for name in names})
    files.update({name:sha(run/name) for name in ('totals.npz','orders.csv','report.md')})
    result = dict(passed=True, identity=identity, files=files, scientific_order_accepted=False,
                  mutable_operational_logs=list(MUTABLE_PROVENANCE),
                  production_promoted=False, scope='complete static global six-field MMS artifacts')
    write(run/'completion.json', result)
    return result


def read_json_value(value):
    import json
    return json.loads(json.dumps(value))
