"""Deterministic checkpoint reductions, without numerical evolution or judgments."""
import csv
import math
from pathlib import Path
import numpy as np
from scripts.q09_evolved_mms import campaign as pilot
from scripts.q09_evolved_mms.provider import load_checkpoint
from scripts.q09_evolved_mms.evolution import step_count
from .campaign import GRIDS, END, COARSE_STEPS, MODES, LEVELS

FIELDS = ('n', 'Te', 'Ti', 'Vi', 'Ve', 'omega')


def snapshot_records(grid, n):
    """Validate all saved times, then reduce actual arrays (not report scalars)."""
    grid = Path(grid)
    provider = pilot.read(grid/'observations.json')['provider']
    with np.load(grid/'observations.npz') as data:
        obs = {k: data[k].copy() for k in data.files}
    volume = obs['volume']; regions = {k[7:]: v for k, v in obs.items() if k.startswith('region:')}
    regions['global'] = np.ones(len(volume), bool)
    records = []
    for mode in MODES:
        for kinds, phi in pilot.KINDS:
            key = pilot.case_key(mode, kinds, phi); folder = grid/'cases'/key
            for level in range(LEVELS):
                dt = (END/COARSE_STEPS[n])/2**level
                signature = pilot.signature(provider, mode, kinds, phi, dt, end=END)
                count = step_count(0., END, dt)
                final = load_checkpoint(folder/f'level{level}.npz', signature)
                for part in range(1, 6):
                    cp = load_checkpoint(folder/f'snapshots/level{level}_part{part}.npz', signature)
                    steps = part*(count//5)
                    t = END if steps == count else steps*dt
                    if (cp['accepted_steps'] != steps or cp['time'] != t
                        or cp['state'].shape != obs['initial'].shape or np.any(cp['state'][:3] <= 0)
                        or not np.isfinite(cp['state']).all()):
                        raise ValueError('snapshot time/coverage/positivity mismatch')
                    # Histories and integrated RHS are bound by checkpoint checksums.
                    expected_times = [i*dt+s*((END if i+1 == count else (i+1)*dt)-i*dt)
                                      for i in range(steps) for s in (0., .5, .5, 1.)]
                    np.testing.assert_allclose(cp['stage_times'], expected_times, atol=1e-14, rtol=1e-14)
                    if part == 5:
                        np.testing.assert_array_equal(cp['state'], final['state'])
                        np.testing.assert_array_equal(cp['integral_rhs'], final['integral_rhs'])
                    exact = obs[f'target:part{part}']; err = cp['state']-exact
                    for region, mask in regions.items():
                        if not mask.any(): continue
                        ids = np.flatnonzero(mask); v = volume[mask]; vtotal = v.sum()
                        e = err[:, mask]; r = exact[:, mask]
                        rms = np.sqrt(np.sum(e**2*v, axis=1)/vtotal)
                        norm = np.sqrt(np.sum(r**2*v, axis=1)/vtotal)
                        evolved = np.sqrt(np.sum((r-obs['initial'][:, mask])**2*v, axis=1)/vtotal)
                        maxima = np.argmax(abs(e), axis=1)
                        for f, field in enumerate(FIELDS):
                            records.append(dict(n=n, case=key, mode=mode, kinds=kinds, phi_kind=phi,
                                level=level, part=part, time=t, dt=dt, field=field, region=region,
                                rms=float(rms[f]), relative=float(rms[f]/norm[f]) if norm[f] else None,
                                relative_to_exact_change=float(rms[f]/evolved[f]) if evolved[f] else None,
                                maximum=float(abs(e[f, maxima[f]])), maximum_owner=int(ids[maxima[f]]),
                                signed_error_integral=float(e[f]@v)))
    return records


def orders_from_records(records):
    # One prescribed dt per grid. Different grids share physical end time and snapshot fractions.
    groups = {}
    for r in records:
        if r['level'] != LEVELS-1: continue
        key = (r['case'], r['field'], r['region'], r['part'])
        group = groups.setdefault(key, {})
        if r['n'] in group: raise ValueError('duplicate grid error record')
        group[r['n']] = r
    orders = []
    for (case, field, region, part), rows in sorted(groups.items()):
        if set(rows) != set(GRIDS): raise ValueError('spatial reduction lacks grid coverage')
        for low, high in ((32,48),(48,64),(32,64)):
            a, b = rows[low]['rms'], rows[high]['rms']
            orders.append(dict(case=case, field=field, region=region, part=part,
                time=END*part/5, interval=f'{low}-{high}', coarse_rms=a, fine_rms=b,
                order=math.log(a/b)/math.log(high/low) if a > 0 and b > 0 else None))
    return orders


def reduce_all(run, identity):
    records = []
    for n in GRIDS:
        records.extend(pilot.read(run/f'N{n}/time_history.json')['records'])
    orders = orders_from_records(records)
    pilot.write(run/'analysis.json', dict(identity=identity, records=records, spatial_orders=orders,
        temporal=dict(measured=False, reason='one fixed timestep per resolution, as requested'), interpretation='machine reductions; scientific interpretation remains local',
        error_definition='evolved numerical solution minus exact manufactured solution; not static N-O-R'))
    with (run/'orders.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(orders[0])); writer.writeheader(); writer.writerows(orders)
    lines = ['# Q09 full-domain finite-duration evolved MMS', '',
        'N32/N48/N64, prescribed phi, T=0.0001, h/32, eight mode/BC cases/grid, one timestep per resolution.',
        'Complete means the frozen six-field parallel RHS; no perpendicular coupling or phi solve.',
        'All arrays/steps/stage times/receipt reductions validated. These are operational gates, not scientific qualification.',
        'Temporal error is not independently measured: this is a spatial refinement campaign. No thresholds or numerics are changed.', '',
        '## Final global errors and spatial orders (prescribed dt)', '',
        '| Case | Field | N32 RMS | N48 RMS | N64 RMS | p32–48 | p48–64 |',
        '|---|---|---:|---:|---:|---:|---:|']
    lookup = {(r['case'],r['field'],r['n']):r for r in records if r['level']==LEVELS-1 and r['part']==5 and r['region']=='global'}
    order_lookup = {(r['case'],r['field'],r['interval']):r['order'] for r in orders if r['part']==5 and r['region']=='global'}
    fmt = lambda v: 'undefined' if v is None else f'{v:.6g}'
    for mode in MODES:
        for kinds, phi in pilot.KINDS:
            case = pilot.case_key(mode,kinds,phi)
            for field in FIELDS:
                values = [lookup[case,field,n]['rms'] for n in GRIDS]
                values += [order_lookup[case,field,interval] for interval in ('32-48','48-64')]
                lines.append('| '+case+' | '+field+' | '+' | '.join(map(fmt,values))+' |')
    lines += ['', 'Regional errors, maxima/owners, signed integrals and five time snapshots are in analysis.json.',
        'Also compare error to the exact temporal change; small error relative to the total field is not sufficient.',
        'No automatic candidate selection, model changes, stability claim or production promotion.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
