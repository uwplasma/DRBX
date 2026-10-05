"""Frozen N32 GPU RK4 timestep pilot; no scientific pass or production change."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
TESTS = ('tests/test_q09_evolved_mms.py', 'tests/test_q09_evolved_inputs.py',
         'tests/test_q09_refinement_pilot.py')
KINDS = (('DDDDDD', 'D'), ('NNNNNN', 'N'), ('DNDNDN', 'N'), ('NDNDND', 'D'))
MODES = ('diffusion', 'complete')
DT = 1e-6
END = 1e-5


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); os.replace(temp, path)


def files():
    roots = list(HERE.rglob('*.py')) + [HERE/'inputs.example.json', HERE/'README.md']
    roots += [REPO/p for p in TESTS]
    roots += [REPO/'src/drbx/dev_docs/q09_evolved_mms_contract.md']
    # Capture the numerical/source identity inputs used by PreparedProvider,
    # including the input authority and the baseline merge implementation.
    import ast
    tree = ast.parse((HERE/'provider.py').read_text())
    assignment = next(x for x in tree.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'SOURCE_DEPENDENCIES' for t in x.targets))
    roots += [REPO/p for p in ast.literal_eval(assignment.value)]
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(roots))}


def freeze():
    """Local packaging only: copy current Q and the committed geometry type dependency."""
    runtime = HERE/'runtime/drbx'
    paths = list((REPO/'src/drbx/native').glob('q*.py'))
    paths += list((REPO/'src/drbx/stencils').glob('q*.py'))
    paths += [REPO/'src/drbx'/p for p in ('_host_guards.py', 'stencils/tensor_rows.py',
        'native/owner_plane_layout.py', 'native/fci_parallel_production_flux.py',
        'native/characteristic_wall_residual.py', 'native/fci_model.py',
        'native/fci_time_integrator.py', 'geometry/_reconstruction_primitives.py')]
    for group in ('', 'native', 'stencils', 'geometry'):
        (runtime/group).mkdir(parents=True, exist_ok=True)
        (runtime/group/'__init__.py').write_text('')
    for p in paths:
        target = runtime/p.relative_to(REPO/'src/drbx'); shutil.copy2(p, target)
    # FciModelState imports HaloLayout3D for unrelated halo helpers. Its type
    # module is frozen from committed HEAD, excluding concurrent P edits.
    geometry = subprocess.check_output(['git', 'show', 'HEAD:src/drbx/geometry/fci_geometry.py'], cwd=REPO)
    # Normalize blank-line trailing whitespace only; preserve the original
    # committed input digest separately from the emitted file's manifest hash.
    clean_geometry = b''.join(b'\n' if line.endswith(b'\n') and not line.strip() else line
                              for line in geometry.splitlines(keepends=True))
    (runtime/'geometry/fci_geometry.py').write_bytes(clean_geometry)
    manifest = dict(schema='q09-n32-rk4-pilot-v1', files=files(), n=32, modes=list(MODES),
        boundaries=[list(k) for k in KINDS], timestep=DT, end_time=END, levels=3,
        expected_steps=[10, 20, 40], total_cases=8, total_level_runs=24,
        backend='gpu', device_execution='one actual A100; sequential cases; no CPU RHS fallback',
        preparation='CPU reference geometry, saved rows and R diffusion; no tracing',
        diffusion_span=1/32, material_inner_span=1/32, material_outer_span=1/16,
        frozen_geometry_type_sha256=hashlib.sha256(geometry).hexdigest(),
        scope='bounded-duration full-domain engineering/temporal pilot; not evolved spatial/stability qualification')
    write(HERE/'manifest.json', manifest)
    print(sha(HERE/'manifest.json'))


def check_source():
    manifest = read(HERE/'manifest.json')
    if manifest['files'] != files(): raise ValueError('frozen Q09 pilot source changed')
    if (manifest['n'], manifest['timestep'], manifest['end_time'], manifest['modes'], manifest['boundaries']) != (
        32, DT, END, list(MODES), [list(k) for k in KINDS]):
        raise ValueError('frozen pilot configuration mismatch')
    return sha(HERE/'manifest.json')


def fixtures():
    """Unpack only checked bounded test inputs, inside this source checkout."""
    base = HERE.parent/'q08_extraction_global'
    manifest = read(base/'input_manifest.json'); design = read(base/'design.json')
    if sha(base/'inputs.tar.gz') != design['input_archive_sha256']:
        raise ValueError('bounded input archive changed')
    with tarfile.open(base/'inputs.tar.gz', 'r:gz') as archive:
        for name, expected in manifest['files'].items():
            if not name.startswith('bounded/'): continue
            member = archive.getmember('inputs/'+name)
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('unsafe bounded input member')
            dest = base/member.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, dest.open('wb') as out: shutil.copyfileobj(src, out)
            if sha(dest) != expected: raise ValueError('bounded fixture content mismatch')


def require(run, name, identity):
    r = read(run/f'{name}.json')
    if r.get('passed') is not True or r.get('identity') != identity: raise ValueError('missing/stale '+name+' gate')
    return r


def verify(args, identity, *, n=32):
    from scripts.q09_evolved_mms.inputs import audit
    run = args.run.resolve()
    for p in (args.baseline_run, args.science_run):
        p = p.resolve()
        if run.is_relative_to(p) or p.is_relative_to(run): raise ValueError('new separate RUN required')
    cfg = dict(schema='q09-prepared-inputs-v1', n=n, baseline_run=str(args.baseline_run.resolve()),
        science_run=str(args.science_run.resolve()), canonical_root=str(args.canonical_root.resolve()),
        cache_directory=str(run/'prepared_reference'), host_memory_gib=args.host_gib)
    path = run/'config/inputs.json'
    if path.exists() and read(path) != cfg: raise ValueError('RUN input configuration changed')
    write(path, cfg)
    _, hashes, plan, estimate = audit(run/'config')
    fixtures()
    result = dict(passed=True, identity=identity, inputs=hashes, config_sha256=sha(path), n=n,
        owners=plan['n_owner'], chunks=len(plan['chunks']), estimate=estimate)
    write(run/'verification.json', result); return result


def tests(run, identity):
    fixtures()
    import pytest
    # Bootstrap establishes CPU/x64. These self-contained campaign tests do
    # not need the broad production conftest/runtime initializer.
    result = pytest.main(['-q', '--noconftest', *[str(REPO/p) for p in TESTS]])
    if result != 0: raise RuntimeError('portable tests failed')
    write(run/'tests.json', dict(passed=True, identity=identity))


def resource_measurement():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(peak_host_rss_bytes=int(rss*(1 if sys.platform == 'darwin' else 1024)))
    if 'jax' in sys.modules:
        import jax
        result['devices'] = [dict(device=str(d), platform=d.platform,
            memory_stats={k: int(v) for k, v in (d.memory_stats() or {}).items()
                          if isinstance(v, int)}) for d in jax.devices()]
    return result


def hardware():
    import jax
    import numpy as np
    if not jax.config.jax_enable_x64 or jax.default_backend() != 'gpu': raise ValueError('actual GPU x64 required')
    devices = jax.devices()
    if any(d.platform != 'gpu' or 'A100' not in d.device_kind.upper() for d in devices):
        raise ValueError('pilot requires the specified actual A100 allocation')
    if np.dtype(jax.numpy.asarray(1., dtype=jax.numpy.float64).dtype) != np.dtype('float64'):
        raise ValueError('float64 unavailable')
    return dict(backend=jax.default_backend(), jax=jax.__version__, numpy=np.__version__,
        devices=[str(d) for d in devices], device_kind=devices[0].device_kind,
        selected_device=str(devices[0]), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)


def device_memory_gate(estimate):
    """Use observed free device memory, without changing scheduler visibility."""
    import jax
    from scripts.q08_extraction_global.gpu import _device_guard
    device = jax.devices()[0]
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True)
    import csv
    rows = []
    for row in csv.reader(output.splitlines()):
        i, uuid, name, total, free = (v.strip() for v in row)
        rows.append(dict(index=int(i), uuid=uuid, name=name,
                         total_bytes=int(total)*(1 << 20), free_bytes=int(free)*(1 << 20)))
    inventory = dict(test_only_cpu_emulation=False, nvidia_smi=rows,
        devices=[dict(id=device.id, local_hardware_id=device.local_hardware_id,
                      memory_stats=device.memory_stats() or {})])
    # Retain Q08's conservative plan/workspace bound and reserve additional
    # live stage states and time-dependent BC/source intermediates for RK4.
    bound = dict(estimate)
    bound['estimated_gpu0_peak_upper_bytes'] += (8*estimate['one_case_state_bytes']
                                               + 6*estimate['one_case_boundary_bytes'])
    _device_guard(bound, inventory)
    return dict(estimated_peak_upper_bytes=bound['estimated_gpu0_peak_upper_bytes'],
                inventory=inventory, scope='conservative admission bound, not measured peak')


def preflight(run, identity):
    verification = require(run, 'verification', identity); require(run, 'tests', identity)
    inventory = hardware(); fixtures()
    memory = device_memory_gate(verification['estimate'])
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scripts.q08_extraction_global import common as c
    from drbx.stencils.q_parallel import load_chunk
    from drbx.stencils.q_bank import build_q_bank
    from drbx.stencils.q_plan import lower_q_plan
    from drbx.native.q_plan import apply_q_plan
    folder = HERE.parent/'q08_extraction_global/inputs/bounded'
    n = verification['n']
    bank = build_q_bank(*(load_chunk(folder/f'N{n}_h{d}.npz') for d in (16, 32)))
    with np.load(folder/f'N{n}_inputs.npz') as z: saved = {k: z[k].copy() for k in z.files}
    geometry = {k: saved[k] for k in ('magnetic_L', 'b_eta', 'eta_step', 'bmag')}
    plan = lower_q_plan(bank, diffusion_span=1/32, **geometry)
    state = np.broadcast_to(np.r_[c.BASE, .2][:, None], (6, bank.metadata['n_owner'])).copy()
    state[:, saved['donor_ids']] = saved['state'][1]
    phi = np.zeros(bank.metadata['n_owner']); phi[saved['donor_ids']] = saved['phi'][1]
    bc = c.boundaries(bank, 1); metrics = []
    for kinds, pk in KINDS:
        fn = jax.jit(lambda p, x, a, b, f, fb: apply_q_plan(p, x, a, b, f, fb, c.COEFF,
            kinds=tuple(kinds), phi_kind=pk, tau=c.TAU, mu=c.MU, characteristic_method='polynomial'))
        with jax.default_device(jax.devices('cpu')[0]):
            host_args = jax.tree.map(lambda a: jax.device_put(a, jax.devices('cpu')[0]), (plan, state, bc[0], bc[1], phi, bc[2]))
            expected = fn(*host_args)
        device_args = jax.tree.map(lambda a: jax.device_put(a, jax.devices()[0]), (plan, state, bc[0], bc[1], phi, bc[2]))
        actual = fn(*device_args)
        metrics.append(c.check_outputs(actual, expected))
    write(run/'preflight.json', dict(passed=True, identity=identity, hardware=inventory, memory=memory, metrics=metrics,
        scope='bounded actual HSX Q action CPU/GPU replay; complete-domain RK4 follows'))


def case_key(mode, kinds, phi): return f'{mode}_{kinds}_phi{phi}'


def signature(provider, mode, kinds, phi, dt, *, end=END):
    from scripts.q08_extraction_global.common import digest
    return digest(dict(provider=provider, mode=mode, kinds=kinds, phi_kind=phi, start=0., end=end, dt=dt))


def validate_case(folder, provider, observations, mode, kinds, phi, *, dt0=DT, end=END, levels=3):
    import numpy as np
    from scripts.q09_evolved_mms.provider import load_checkpoint
    from scripts.q09_evolved_mms.evolution import error_report, temporal_comparison, step_count
    report = read(folder/'report.json')
    if report['identity'] != provider or len(report['runs']) != levels: raise ValueError('case report identity/coverage')
    states = []; v = observations['volume']; target = observations['target']; initial = observations['initial']
    regions = {k[7:]: a for k, a in observations.items() if k.startswith('region:')}
    for level, r in enumerate(report['runs']):
        dt = dt0/2**level
        cp = load_checkpoint(folder/f'level{level}.npz', signature(provider, mode, kinds, phi, dt, end=end))
        x = cp['state']; steps = step_count(0., end, dt)
        if (not r['completed'] or r['dt'] != dt or r['start'] != 0 or r['requested_end'] != end
            or r['actual_end'] != end or cp['time'] != end or cp['accepted_steps'] != steps
            or r['accepted_steps'] != steps or x.shape != initial.shape or np.any(x[:3] <= 0)):
            raise ValueError('time/shape/positivity completion gate')
        expected_times = [i*dt+s*((end if i+1 == steps else (i+1)*dt)-i*dt)
                          for i in range(steps) for s in (0., .5, .5, 1.)]
        np.testing.assert_allclose(cp['stage_times'], expected_times, atol=1e-14, rtol=1e-14)
        if r['stage_times'] != cp['stage_times'] or r['error'] != error_report(x, target, v, regions):
            raise ValueError('solution/error/history reduction mismatch')
        change = (x-initial)@v
        for name, expected in (('volume_integral_change', change), ('integrated_rhs', cp['integral_rhs']),
            ('rk_balance_residual', change-cp['integral_rhs']), ('mms_volume_integral_drift', (x-target)@v)):
            np.testing.assert_allclose(r[name], expected, atol=1e-14, rtol=1e-13)
        states.append(x)
    if report['temporal_self_convergence'] != temporal_comparison(states, v):
        raise ValueError('temporal reduction mismatch')
    return report


def run_pilot(run, identity, *, dt0=DT, end=END, checkpoint_every=1, snapshots=False, modes=MODES, levels=3):
    verification = require(run, 'verification', identity)
    require(run, 'tests', identity); require(run, 'preflight', identity)
    if sha(run/'config/inputs.json') != verification['config_sha256']:
        raise ValueError('verified input configuration changed')
    inventory = hardware()
    memory = device_memory_gate(verification['estimate'])
    import jax
    import numpy as np
    from types import SimpleNamespace
    from scripts.q09_evolved_mms.inputs import load
    from scripts.q09_evolved_mms.evolution import q_payload, q_stepper
    from scripts.q09_evolved_mms.cli import _run
    from scripts.q08_extraction_global.common import atomic_npz
    tick = time.perf_counter(); provider = load(run/'config')
    for key, value in verification['inputs'].items():
        if provider.input_hashes.get(key) != value: raise ValueError('verified input content changed')
    preparation_seconds = time.perf_counter()-tick; pid = provider.identity
    tick = time.perf_counter(); payload = q_payload(provider)
    jax.block_until_ready(payload); staging_seconds = time.perf_counter()-tick
    observations = dict(initial=np.asarray(provider.manufactured.state(0.)),
        target=np.asarray(provider.manufactured.state(end)), volume=provider.volume,
        **{'region:'+k: v for k, v in provider.regions.items()})
    if snapshots:
        observations.update({f'target:part{j}': np.asarray(provider.manufactured.state(end*j/5))
                             for j in range(1, 6)})
    opath = run/'observations.npz'
    if opath.exists():
        prior = require(run, 'observations', identity)
        if prior['provider'] != pid or sha(opath) != prior['sha256']: raise ValueError('observation identity/content changed')
        with np.load(opath) as z:
            for k, v in observations.items(): np.testing.assert_array_equal(z[k], v)
    else:
        atomic_npz(opath, **observations)
        write(run/'observations.json', dict(passed=True, identity=identity, provider=pid, sha256=sha(opath)))
    write(run/'preparation.json', dict(passed=True, identity=identity, provider=pid,
        seconds=preparation_seconds, staging_seconds=staging_seconds, hardware=inventory, memory=memory,
        resources=resource_measurement()))
    for mode in modes:
        for kinds, phi in KINDS:
            key = case_key(mode, kinds, phi); folder = run/'cases'/key; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'receipt.json').exists():
                receipt = read(folder/'receipt.json')
                if receipt['identity'] != identity or receipt['provider'] != pid or receipt.get('passed') is not True:
                    raise ValueError('case checkpoint identity')
                for name, h in receipt['files'].items():
                    if sha(folder/name) != h: raise ValueError('case content changed')
                validate_case(folder, pid, observations, mode, kinds, phi, dt0=dt0, end=end, levels=levels)
                continue
            (folder/'report.json').unlink(missing_ok=True)
            args = SimpleNamespace(dt=dt0, end=end, mode=mode, kinds=kinds, phi_kind=phi, output=folder, resume=True, checkpoint_every=checkpoint_every, snapshots=snapshots, levels=levels)
            tick = time.perf_counter()
            _run(args, provider, payload, pid, q_stepper(mode, tuple(kinds), phi))
            validate_case(folder, pid, observations, mode, kinds, phi, dt0=dt0, end=end, levels=levels)
            names = [str(p.relative_to(folder)) for p in sorted((folder/'snapshots').glob('*.npz'))] if snapshots else []
            names += ['report.json', 'timings.json', 'timing_history.jsonl', *[f'level{i}.npz' for i in range(levels)]]
            write(folder/'receipt.json', dict(passed=True, identity=identity, provider=pid,
                mode=mode, kinds=kinds, phi_kind=phi, seconds=time.perf_counter()-tick,
                resources=resource_measurement(),
                files={name: sha(folder/name) for name in names}))
    validate(run, identity, n=verification['n'], dt0=dt0, end=end, snapshots=snapshots, modes=modes, levels=levels)


def validate(run, identity, *, n=32, dt0=DT, end=END, snapshots=False, modes=MODES, levels=3):
    import numpy as np
    verification = require(run, 'verification', identity)
    require(run, 'tests', identity); require(run, 'preflight', identity)
    if sha(run/'config/inputs.json') != verification['config_sha256']:
        raise ValueError('verified input configuration changed')
    obs = require(run, 'observations', identity); prep = require(run, 'preparation', identity)
    if prep['provider'] != obs['provider'] or sha(run/'observations.npz') != obs['sha256']:
        raise ValueError('observation/preparation receipt mismatch')
    with np.load(run/'observations.npz') as z: observations = {k: z[k].copy() for k in z.files}
    if (observations['initial'].shape != (6, verification['owners'])
        or observations['target'].shape != observations['initial'].shape
        or observations['volume'].shape != (verification['owners'],)
        or verification['n'] != n or verification['owners'] != {32:25376, 48:86016, 64:202304}[n]
        or not all(np.isfinite(a).all() for a in observations.values())
        or np.any(observations['volume'] <= 0)):
        raise ValueError('full observation coverage/finite/volume gate')
    records = {}; consumed = {name: sha(run/name) for name in (
        'verification.json', 'tests.json', 'preflight.json', 'preparation.json', 'observations.json', 'observations.npz', 'config/inputs.json')}
    for mode in modes:
        for kinds, phi in KINDS:
            key = case_key(mode, kinds, phi); folder = run/'cases'/key; receipt = read(folder/'receipt.json')
            if (receipt.get('passed') is not True or receipt['identity'] != identity or receipt['provider'] != obs['provider']
                or (receipt['mode'], receipt['kinds'], receipt['phi_kind']) != (mode, kinds, phi)):
                raise ValueError('case completion identity')
            expected = {'report.json', 'timings.json', 'timing_history.jsonl'} | {f'level{i}.npz' for i in range(levels)}
            if snapshots:
                expected |= {f'snapshots/level{level}_part{j}.npz' for level in range(levels) for j in range(1, 6)}
            if set(receipt['files']) != expected:
                raise ValueError('case completion coverage')
            for name, h in receipt['files'].items():
                if sha(folder/name) != h: raise ValueError('completed case content changed')
                consumed[str((folder/name).relative_to(run))] = h
            consumed[str((folder/'receipt.json').relative_to(run))] = sha(folder/'receipt.json')
            records[key] = validate_case(folder, obs['provider'], observations, mode, kinds, phi, dt0=dt0, end=end, levels=levels)
    write(run/'summary.json', dict(identity=identity, cases=records, interpretation='machine reductions only; scientific analysis pending'))
    consumed['summary.json'] = sha(run/'summary.json')
    write(run/'completion.json', dict(passed=True, identity=identity, cases=len(modes)*len(KINDS), level_runs=len(modes)*len(KINDS)*levels,
        files=consumed, scientific_qualification=False, stability_qualification=False, production_promoted=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'fixtures', 'verify', 'tests', 'preflight', 'run', 'validate'))
    p.add_argument('--run', type=Path)
    p.add_argument('--baseline-run', type=Path); p.add_argument('--science-run', type=Path)
    p.add_argument('--canonical-root', type=Path); p.add_argument('--host-gib', type=float)
    args = p.parse_args()
    if args.command == 'freeze': freeze(); return
    if args.command == 'fixtures': fixtures(); return
    if args.run is None: p.error('--run is required')
    if args.command == 'verify' and any(getattr(args, k) is None for k in ('baseline_run', 'science_run', 'canonical_root', 'host_gib')):
        p.error('verify requires --baseline-run --science-run --canonical-root --host-gib')
    identity = check_source(); run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    from scripts.q09_evolved_mms.bootstrap import configure
    configure(run, gpu=args.command in ('preflight', 'run'))
    with (run/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tick = time.perf_counter()
        try:
            if args.command == 'verify': verify(args, identity)
            elif args.command == 'tests': tests(run, identity)
            elif args.command == 'preflight': preflight(run, identity)
            elif args.command == 'run': run_pilot(run, identity)
            elif args.command == 'validate': validate(run, identity)
        except BaseException as exc:
            write(run/'status.json', dict(identity=identity, stage=args.command, status='failed', error=repr(exc), seconds=time.perf_counter()-tick))
            raise
        write(run/'status.json', dict(identity=identity, stage=args.command, status='complete', seconds=time.perf_counter()-tick,
                                     resources=resource_measurement()))


if __name__ == '__main__': main()
