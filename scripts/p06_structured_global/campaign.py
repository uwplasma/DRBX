#!/usr/bin/env python3
"""Repository entry point for the common-structured P06 curvature qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Set before importing NumPy/JAX, also inherited by all spawned workers.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import parallel_runner as runner
import observations
sys.path.insert(0,str(REPO/'scripts'))
from perpendicular_structured import optimization_resume as upgrade


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(args):
    inputs = json.loads((HERE / 'input_manifest.json').read_text())
    for record in inputs['files']:
        path = args.input_root / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'missing or wrong-size frozen input: {path}')
        if runner._sha256(path) != record['sha256']:
            raise ValueError(f'changed frozen input: {path}')
    tracked_sources = [
        HERE / name for name in ('campaign.py', 'parallel_runner.py', 'observations.py', 'numerics.py',
                                 'configuration.json', 'input_manifest.json')
    ]
    tracked_sources += [
        REPO / name for name in (
            'hsx_mms_continuum_reference.py',
            'scripts/p07_diffusion_global/numerics.py',
            'scripts/p07_combined_global/kernels.py',
            'src/drbx/geometry/fci_perpendicular_bracket.py',
            'src/drbx/geometry/fci_boundary_functional_reconstruction.py',
            'src/drbx/native/fci_curvature_production_flux.py',
            'src/drbx/native/fci_operators.py',
        )
    ]
    shared = REPO / 'scripts/perpendicular_structured'
    tracked_sources += sorted(shared.rglob('*.py'))
    if not shared.is_dir() or not any(shared.rglob('*.py')):
        raise ValueError('shared perpendicular_structured service is required')
    sources = {str(path.relative_to(REPO)): runner._sha256(path) for path in tracked_sources}
    configuration = json.loads((HERE / 'configuration.json').read_text())
    content = {'configuration':configuration, 'inputs':inputs, 'sources':sources}
    manifest = {'schema':'drbx.p06-structured-campaign-v1', 'content_identity':content,
                'sha256':canonical(content)}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'campaign_manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        previous=json.loads(destination.read_text())
        if args.command=='adopt-optimization':
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
            upgrade.adopt(args.output,'p06',previous,sources,commit)
        else:
            upgrade.check(args.output,'p06',previous,sources)
        return previous
    if not destination.exists() and any(args.output.glob('N*.npz')):
        raise ValueError('unidentified/historical outputs in new campaign directory')
    runner._atomic_json(destination, manifest)
    return manifest


def namespace(args, **kwargs):
    return argparse.Namespace(config=HERE/'configuration.json', deployment_root=REPO,
        input_root=args.input_root.resolve(), output_root=args.output.resolve(),
        input_manifest=HERE/'input_manifest.json', prepare=None, **kwargs)


def fresh_stage(args, stage, resolution=None):
    command = [
        sys.executable,
        str(HERE / "parallel_runner.py"),
        "frozen-stage",
        "--config", str(HERE / "configuration.json"),
        "--deployment-root", str(REPO),
        "--input-root", str(args.input_root.resolve()),
        "--output-root", str(args.output.resolve()),
        "--stage", stage,
    ]
    if resolution is not None:
        command.extend(("--resolution", str(int(resolution))))
    subprocess.run(command, check=True)


def preflight(args):
    for n in args.resolutions:
        observations.run(args, n)
        fresh_stage(args, 'prepare', n)
        fresh_stage(args, f'validate:prepare_N{n}', n)
        plan = args.output / f'N{n}.preflight.plan.json'
        runner.command_plan(namespace(args, resolution=n, coverage='preflight', plan=plan))
        runner.command_execute(namespace(args, plan=plan, workers=args.workers,
            max_tasks_per_worker=args.max_tasks_per_worker, fail_unit=None,
            memory_budget_gib=args.memory_budget_gib,
            worker_memory_gib=args.worker_memory_gib,
            memory_reserve_gib=args.memory_reserve_gib))
        runner.command_validate(namespace(args, plan=plan))
        fresh_stage(args, 'preflight', n)
        fresh_stage(args, 'validate:preflight', n)


def run(args):
    if args.workers < 1:
        raise ValueError('workers must be positive')
    for n in args.resolutions:
        path = args.output / f'N{n}.preflight.json'
        if not path.exists():
            raise ValueError(f'run fresh preflight first: {path}')
        fresh_stage(args, 'validate:preflight', n)
        plan = args.output / f'N{n}.plan.json'
        runner.command_plan(namespace(args, resolution=n, coverage='global', plan=plan))
        runner.command_execute(namespace(args, plan=plan, workers=args.workers,
            max_tasks_per_worker=args.max_tasks_per_worker, fail_unit=None,
            memory_budget_gib=args.memory_budget_gib,
            worker_memory_gib=args.worker_memory_gib,
            memory_reserve_gib=args.memory_reserve_gib))
        runner.command_validate(namespace(args, plan=plan))
        runner.command_assemble(namespace(args, plan=plan))
        fresh_stage(args, f'validate:case_N{n}', n)
    if all((args.output/f'N{n}.json').is_file() for n in (32,48,64)):
        fresh_stage(args, 'merge')
        fresh_stage(args, 'validate:merge')


def validate(args):
    for n in args.resolutions:
        runner.command_validate(namespace(args, plan=args.output/f'N{n}.plan.json'))
        fresh_stage(args, f'validate:case_N{n}', n)
    if set(args.resolutions) == {32,48,64}:
        fresh_stage(args, 'merge')
        fresh_stage(args, 'validate:merge')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify-inputs','adopt-optimization','preflight','run','validate'))
    parser.add_argument('--input-root', type=Path, default=REPO.parent)
    parser.add_argument('--output', type=Path, default=REPO/'work/p06_structured_global_v1')
    parser.add_argument('--resolutions', type=int, nargs='+', choices=(32,48,64), default=[32,48,64])
    parser.add_argument('--workers', type=runner._positive_int)
    parser.add_argument('--max-tasks-per-worker', type=runner._positive_int, default=16)
    parser.add_argument('--memory-budget-gib', type=runner._positive_float)
    parser.add_argument('--worker-memory-gib', type=runner._positive_float)
    parser.add_argument('--memory-reserve-gib', type=runner._nonnegative_float, default=1.0)
    args = parser.parse_args()
    if args.command in ('preflight', 'run') and (args.workers is None or args.workers < 1):
        parser.error(f'{args.command} requires --workers chosen for the active allocation')
    # One campaign owner, including preparation/merge, not just chunk execution.
    with runner._exclusive_output(args.output/'campaign_control'):
        identity = verify(args)
        print(json.dumps({'campaign':identity['sha256'],'command':args.command}),flush=True)
        if args.command == 'preflight': preflight(args)
        elif args.command == 'run': run(args)
        elif args.command == 'validate': validate(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
