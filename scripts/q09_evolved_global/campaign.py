"""Full-grid prescribed-phi evolved MMS, using the accepted Q09 runtime."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import json
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.q09_evolved_mms import campaign as pilot

GRIDS = (32, 48, 64)
END = 1e-4
LEVELS = 1
MODES = ('diffusion', 'complete')
COARSE_STEPS = {32: 100, 48: 225, 64: 400}
TESTS = (*pilot.TESTS, 'tests/test_q09_evolved_global.py')
PILOT_COMMIT = '50b3a6d80bc33b123cc9d0df6e62d7721025103f'
PILOT_IDENTITY = 'eeb1abe34be5ea9535ecac529b8ec66269673e631c2dc847213a61b3653f6d70'


def configuration():
    return dict(schema='q09-evolved-global-v1', grids=list(GRIDS), end_time=END,
        coarse_steps=COARSE_STEPS, levels=LEVELS, checkpoint_every=25, snapshot_parts=5,
        modes=list(MODES), boundaries=[list(p) for p in pilot.KINDS],
        backend='gpu', execution='one visible A100; sequential cases and grids; CPU input preparation',
        diffusion_span=1/32, material_inner_span=1/32, material_outer_span=1/16,
        pilot_commit=PILOT_COMMIT, pilot_identity=PILOT_IDENTITY,
        total_cases=24, total_level_runs=24, total_steps=8*sum(COARSE_STEPS.values()),
        scope='finite-duration six-field prescribed-phi parallel RHS; no reconstructed phi or production promotion')


def files():
    # Frozen native/stencil copies are the numerical authority. Do not bind
    # unused production sources, which may be under concurrent development.
    import ast
    roots = list(pilot.HERE.rglob('*.py')) + [pilot.HERE/'inputs.example.json', pilot.HERE/'README.md']
    roots += [REPO/p for p in pilot.TESTS] + [REPO/'src/drbx/dev_docs/q09_evolved_mms_contract.md']
    tree = ast.parse((pilot.HERE/'provider.py').read_text())
    assignment = next(x for x in tree.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SOURCE_DEPENDENCIES' for t in x.targets))
    roots += [REPO/p for p in ast.literal_eval(assignment.value) if not p.startswith('src/')]
    result = {str(p.relative_to(REPO)): pilot.sha(p) for p in sorted(set(roots))}
    for p in [*HERE.glob('*.py'), HERE/'README.md', REPO/TESTS[-1], pilot.HERE/'manifest.json']:
        result[str(p.relative_to(REPO))] = pilot.sha(p)
    return dict(sorted(result.items()))


def runtime_matches_pilot():
    if pilot.sha(pilot.HERE/'manifest.json') != PILOT_IDENTITY:
        raise ValueError('original pilot manifest changed')
    original = pilot.read(pilot.HERE/'manifest.json')['files']
    for path, digest in original.items():
        if path.startswith('scripts/q09_evolved_mms/runtime/') and pilot.sha(REPO/path) != digest:
            raise ValueError('accepted pilot numerical runtime changed: '+path)


def freeze():
    runtime_matches_pilot()  # Never copy current production sources.
    pilot.write(HERE/'manifest.json', dict(configuration=configuration(), files=files()))
    print(pilot.sha(HERE/'manifest.json'))


def check_source():
    runtime_matches_pilot()
    manifest = pilot.read(HERE/'manifest.json')
    # JSON normalizes integer mapping keys; compare in that representation.
    if manifest != json.loads(json.dumps(dict(configuration=configuration(), files=files()))):
        raise ValueError('full evolved campaign source/configuration changed')
    return pilot.sha(HERE/'manifest.json')


def verify(args, identity):
    root = args.run
    for n in GRIDS:
        grid_args = SimpleNamespace(**vars(args)); grid_args.run = root/f'N{n}'
        pilot.verify(grid_args, identity, n=n)
    pilot.write(root/'verification.json', dict(passed=True, identity=identity,
        files={f'N{n}/verification.json': pilot.sha(root/f'N{n}/verification.json') for n in GRIDS}))


def require_verification(run, identity):
    receipt = pilot.require(run, 'verification', identity)
    if set(receipt['files']) != {f'N{n}/verification.json' for n in GRIDS}:
        raise ValueError('grid verification coverage')
    for name, digest in receipt['files'].items():
        if pilot.sha(run/name) != digest: raise ValueError('verified grid receipt changed')


def tests(run, identity):
    pilot.fixtures()
    import pytest
    result = pytest.main(['-q', '--noconftest', *[str(REPO/p) for p in TESTS]])
    if result != 0: raise RuntimeError('portable tests failed')
    pilot.write(run/'tests.json', dict(passed=True, identity=identity))
    for n in GRIDS:
        pilot.write(run/f'N{n}/tests.json', dict(passed=True, identity=identity))


def validate_grid(run, identity, n):
    pilot.validate(run/f'N{n}', identity, n=n, dt0=END/COARSE_STEPS[n], end=END, snapshots=True, modes=MODES, levels=LEVELS)
    from scripts.q09_evolved_global.reduce import snapshot_records
    records = snapshot_records(run/f'N{n}', n)
    pilot.write(run/f'N{n}/time_history.json', dict(identity=identity, records=records))
    return records


def validate_complete(run, identity):
    require_verification(run, identity); pilot.require(run, 'tests', identity)
    from scripts.q09_evolved_global.reduce import reduce_all
    for n in GRIDS: validate_grid(run, identity, n)
    reduce_all(run, identity)
    names = ['verification.json', 'tests.json', 'analysis.json', 'orders.csv', 'report.md']
    for n in GRIDS:
        names.extend(f'N{n}/'+p for p in ('completion.json', 'summary.json', 'time_history.json'))
        names.extend(f'N{n}/'+p for p in pilot.read(run/f'N{n}/completion.json')['files'])
    pilot.write(run/'completion.json', dict(passed=True, identity=identity, cases=24, level_runs=24,
        accepted_steps=configuration()['total_steps'], snapshots=120,
        files={p: pilot.sha(run/p) for p in sorted(set(names))},
        scientific_qualification=False, long_time_stability_qualification=False, production_promoted=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'source', 'tests', 'verify', 'preflight', 'run', 'validate'))
    p.add_argument('--run', type=Path)
    p.add_argument('--n', type=int, choices=GRIDS)
    p.add_argument('--baseline-run', type=Path); p.add_argument('--science-run', type=Path)
    p.add_argument('--canonical-root', type=Path); p.add_argument('--host-gib', type=float)
    a = p.parse_args()
    if a.command == 'freeze': freeze(); return
    identity = check_source()
    if a.command == 'source': print(identity); return
    if a.run is None: p.error('--run required')
    if a.command in ('run', 'preflight') and a.n is None: p.error('--n required')
    if a.command == 'verify' and any(getattr(a, k) is None for k in ('baseline_run', 'science_run', 'canonical_root', 'host_gib')):
        p.error('verify needs --baseline-run --science-run --canonical-root --host-gib')
    a.run = a.run.resolve(); a.run.mkdir(parents=True, exist_ok=True)
    from scripts.q09_evolved_mms.bootstrap import configure
    configure(a.run, gpu=a.command in ('preflight', 'run'))
    with (a.run/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tick = time.perf_counter()
        try:
            if a.command == 'tests': tests(a.run, identity)
            elif a.command == 'verify': verify(a, identity)
            elif a.command in ('preflight', 'run'):
                require_verification(a.run, identity); pilot.require(a.run, 'tests', identity)
                grid = a.run/f'N{a.n}'
                if a.command == 'preflight': pilot.preflight(grid, identity)
                else:
                    pilot.run_pilot(grid, identity, dt0=END/COARSE_STEPS[a.n], end=END,
                                    checkpoint_every=25, snapshots=True, modes=MODES, levels=LEVELS)
                    validate_grid(a.run, identity, a.n)
            elif a.command == 'validate': validate_complete(a.run, identity)
        except BaseException as exc:
            pilot.write(a.run/'status.json', dict(identity=identity, stage=a.command, n=a.n,
                status='failed', error=repr(exc), seconds=time.perf_counter()-tick))
            raise
        pilot.write(a.run/'status.json', dict(identity=identity, stage=a.command, n=a.n,
            status='complete', seconds=time.perf_counter()-tick, resources=pilot.resource_measurement()))


if __name__ == '__main__': main()
