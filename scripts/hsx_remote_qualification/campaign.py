#!/usr/bin/env python3
"""Repository entry point for the clean CPU global qualification campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(args):
    inputs = json.loads((HERE / 'input_manifest.json').read_text())
    sources = json.loads((HERE / 'source_manifest.json').read_text())
    for root, records in ((args.input_root, inputs['files']), (REPO, sources['files'])):
        for record in records:
            path = root / record['path']
            if not path.is_file() or path.stat().st_size != record['bytes']:
                raise ValueError(f'missing or wrong-size frozen input/source: {path}')
            if runner._sha256(path) != record['sha256']:
                raise ValueError(f'changed frozen input/source: {path}')
    configuration = json.loads((HERE / 'configuration.json').read_text())
    content = {'configuration':configuration, 'inputs':inputs, 'sources':sources}
    manifest = {'schema':'drbx.hsx-clean-campaign-v3', 'content_identity':content,
                'sha256':canonical(content)}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'campaign_manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError('campaign identity changed; use a new output directory, never mix chains')
    if not destination.exists() and any(args.output.glob('N*.npz')):
        raise ValueError('unidentified/historical outputs in new campaign directory')
    runner._atomic_json(destination, manifest)
    return manifest


def namespace(args, **kwargs):
    return argparse.Namespace(config=HERE/'configuration.json', deployment_root=REPO,
        input_root=args.input_root.resolve(), output_root=args.output.resolve(),
        input_manifest=HERE/'input_manifest.json', prepare=None, **kwargs)


def fresh_stage(args, stage, resolution=None):
    runner.command_frozen_stage(namespace(args, stage=stage, resolution=resolution))


def preflight(args):
    for n in args.resolutions:
        fresh_stage(args, 'prepare', n)
        fresh_stage(args, f'validate:prepare_N{n}', n)
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
            max_tasks_per_worker=args.max_tasks_per_worker, fail_unit=None))
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
    parser.add_argument('command', choices=('verify-inputs','preflight','run','validate'))
    parser.add_argument('--input-root', type=Path, default=REPO.parent)
    parser.add_argument('--output', type=Path, default=REPO/'work/hsx_clean_remote_v3')
    parser.add_argument('--resolutions', type=int, nargs='+', choices=(32,48,64), default=[32,48,64])
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--max-tasks-per-worker', type=int, default=64)
    args = parser.parse_args()
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
