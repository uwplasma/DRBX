"""Sequential timestep ladder consuming an explicit checked prepared provider."""
import argparse
import importlib
import fcntl
import json
import time
from pathlib import Path
import jax
from scripts.q08_extraction_global.common import atomic_json, digest
from .provider import PreparedProvider, save_checkpoint, load_checkpoint
from .evolution import advance, q_stepper, q_payload, temporal_comparison


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', default='scripts.q09_evolved_mms.inputs:load',
        help='module:factory; default: checked Q08 saved-bank/reference adapter')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('diffusion', 'complete'), required=True)
    p.add_argument('--dt', type=float, required=True)
    p.add_argument('--end', type=float, required=True)
    p.add_argument('--kinds', default='DDDDDD')
    p.add_argument('--phi-kind', choices=('D', 'N'), default='D')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if len(args.kinds) != 6 or any(k not in 'DN' for k in args.kinds):
        p.error('--kinds requires six D/N characters')
    if not jax.config.jax_enable_x64:
        p.error('Q09 requires JAX_ENABLE_X64=true')
    module, name = args.provider.split(':')
    provider = getattr(importlib.import_module(module), name)(args.inputs)
    if not isinstance(provider, PreparedProvider):
        raise TypeError('factory must return PreparedProvider')
    provider.validate(); data = q_payload(provider)
    identity = provider.identity
    stepper = q_stepper(args.mode, tuple(args.kinds), args.phi_kind)
    args.output.mkdir(parents=True, exist_ok=True)
    binding = digest(dict(provider=identity, mode=args.mode, kinds=args.kinds,
        phi_kind=args.phi_kind, dt=args.dt, end=args.end))
    with (args.output/'.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())['binding'] != binding:
            raise ValueError('output directory bound to different inputs/source/run; choose a new output')
        atomic_json(manifest, dict(binding=binding, provider=identity, status='running'))
        # An interrupted or failed resume cannot expose an old successful report.
        (args.output/'report.json').unlink(missing_ok=True)
        _run(args, provider, data, identity, stepper)
        atomic_json(manifest, dict(binding=binding, provider=identity, status='completed'))


def _run(args, provider, data, identity, stepper):
    states = []; reports = []; timings = []
    attempt = time.time_ns()
    def history(record):
        # Append-only operational timing survives a partially finished ladder.
        # Checkpoints, not this log, determine which steps are accepted on resume.
        with (args.output/'timing_history.jsonl').open('a') as log:
            log.write(json.dumps(dict(attempt=attempt, **record), allow_nan=False)+'\n')
    history(dict(event='begin', identity=identity))
    # One compiled executable per boundary/mode case, reused for dt refinements.
    compiled = jax.jit(lambda s, t, h, payload: stepper(s, time=t, timestep=h, carry=payload))
    for level in range(3):
        dt = args.dt/2**level
        signature = digest(dict(provider=identity, mode=args.mode, kinds=args.kinds,
            phi_kind=args.phi_kind, start=0., end=args.end, dt=dt))
        path = args.output/f'level{level}.npz'
        if path.exists() and not args.resume:
            raise ValueError('checkpoint exists: choose a fresh output or explicitly --resume')
        resume = load_checkpoint(path, signature) if path.exists() else None
        samples = []; tick = time.perf_counter()
        def timed_step(i, seconds):
            sample = dict(step=i, seconds=seconds); samples.append(sample)
            history(dict(event='step', level=level, **sample))
        state, report = advance(stepper, provider.manufactured.state(0.), start=0., end=args.end,
            dt=dt, volume=provider.volume, target=provider.manufactured.state, regions=provider.regions,
            carry=data, resume=resume, compiled_step=compiled,
            timing_callback=timed_step,
            checkpoint=lambda payload: save_checkpoint(path, signature, payload))
        states.append(state); reports.append(report)
        timings.append(dict(level=level, wall_seconds=time.perf_counter()-tick,
            synchronized_step_including_validation=samples, resumed_steps=0 if resume is None else resume['accepted_steps']))
        history(dict(event='level_complete', level=level, accepted_steps=report['accepted_steps'],
                     wall_seconds=timings[-1]['wall_seconds']))
        print(f'{args.mode} level={level} start=0 end={report["actual_end"]} accepted={report["accepted_steps"]}', flush=True)
    atomic_json(args.output/'report.json', dict(identity=identity, runs=reports,
        temporal_self_convergence=temporal_comparison(states, provider.volume),
        qualification='unperformed: spatial solution order, stability envelope, sharding, reconstructed phi'))
    atomic_json(args.output/'timings.json', dict(identity=identity, levels=timings,
        scope='first executed step includes compilation; step samples exclude checkpoint I/O; wall includes it'))


if __name__ == '__main__':
    main()
