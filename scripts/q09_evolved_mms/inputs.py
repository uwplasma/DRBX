"""Read-only adapter for checked Q08 full banks and independent references.

Numerical rows are only loaded/merged. Geometry is evaluated for raw continuum
material data, never retraced. Each new reference chunk is content-checkpointed.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import fcntl
import json
import sys
import jax
import numpy as np
from scripts.q08_extraction_global.common import (sha, digest, atomic_json,
    atomic_npz, load_geometry_model)
from scripts.q08_extraction_global.gpu import estimate_merge, merge_chunks
from scripts.q08_rhs_mms_global.science import masks, REGIONS
from .provider import PreparedProvider, FROZEN_IDENTITY, source_hash
from .mms import Manufactured, ContinuumReference

SCIENCE_IDENTITY = 'e8303a1a3b93580ebfabd2e0c68b50dc4e8f9f9d90057840e9f38fb7cf389652'
FIELDS = ('values', 'gradients', 'phi_gradient', 'diffusion', 'kappa', 'bmag',
          'owner_raw', 'owner_weight')
BASE_SOURCE = Path(__file__).resolve().parents[1]/'q08_extraction_global'


def read(path):
    return json.loads(Path(path).read_text())


def child(root, relative):
    root = Path(root).resolve(); name = Path(relative)
    if name.is_absolute() or '..' in name.parts or not (root/name).resolve().is_relative_to(root):
        raise ValueError('input path escapes root')
    return root/name


def config(directory):
    directory = Path(directory).resolve()
    if not (directory/'inputs.json').is_file():
        raise ValueError('Q09 requires a prepared-input configuration: '+str(directory/'inputs.json'))
    cfg = read(directory/'inputs.json')
    if set(cfg) != {'schema', 'n', 'baseline_run', 'science_run', 'canonical_root',
                    'cache_directory', 'host_memory_gib'} or cfg['schema'] != 'q09-prepared-inputs-v1':
        raise ValueError('Q09 input configuration schema mismatch')
    if type(cfg['n']) is not int or cfg['n'] not in (32, 48, 64):
        raise ValueError('Q09 requires a canonical 32/48/64 grid')
    if not np.isfinite(cfg['host_memory_gib']) or cfg['host_memory_gib'] <= 0:
        raise ValueError('positive host memory budget required')
    for name in ('baseline_run', 'science_run', 'canonical_root', 'cache_directory'):
        p = Path(cfg[name]); cfg[name] = (directory/p).resolve() if not p.is_absolute() else p.resolve()
    cache = cfg['cache_directory']
    for name in ('baseline_run', 'science_run'):
        root = cfg[name]
        if cache.is_relative_to(root) or root.is_relative_to(cache):
            raise ValueError('reference cache must be separate from immutable input runs')
    return cfg


def audit(directory):
    """Verify the consumed lineage/content before dense load or Python imports."""
    cfg = config(directory); n = cfg['n']; old = cfg['baseline_run']; science = cfg['science_run']
    design = read(BASE_SOURCE/'design.json')
    if digest(design) != FROZEN_IDENTITY or sha(BASE_SOURCE/'input_manifest.json') != design['input_manifest_sha256']:
        raise ValueError('accepted Q08 input authority changed')
    manifest = read(BASE_SOURCE/'input_manifest.json')
    verification = read(old/'verification.json')
    cpu = read(old/f'cpu_N{n}.json')
    for record in (verification, cpu):
        if record.get('passed') is not True or record.get('campaign_identity') != FROZEN_IDENTITY:
            raise ValueError('accepted Q08 verification/CPU gate missing')
    if verification.get('canonical') != manifest['canonical']:
        raise ValueError('canonical receipt identity mismatch')
    if digest(read(old/'provenance/design.json')) != FROZEN_IDENTITY:
        raise ValueError('baseline design identity mismatch')
    hashes = dict(q08_receipt_identity=FROZEN_IDENTITY,
        q08_verification=sha(old/'verification.json'), cpu_gate=sha(old/f'cpu_N{n}.json'),
        input_manifest=sha(BASE_SOURCE/'input_manifest.json'))
    # Verify all geometry-model code before executing it, plus the owner plan.
    for name, expected in manifest['files'].items():
        if name != 'plan.json' and not name.startswith('geometry_model/'):
            continue
        path = child(old/'inputs', name)
        if sha(path) != expected:
            raise ValueError('saved geometry model/plan changed: '+name)
        hashes['input:'+name] = expected
    model_cfg = read(old/'inputs/geometry_model/inputs.json')
    canonical = [model_cfg['metric_cache'], model_cfg['makegrid']]
    canonical += [f"{model_cfg['geometry']}/{n}x{n}x{n}/{name}" for name in ('base_geometry.npz', 'rlp_topology.npz')]
    for name in canonical:
        expected = manifest['canonical'][name]
        if sha(child(cfg['canonical_root'], name)) != expected:
            raise ValueError('canonical content mismatch: '+name)
        hashes['canonical:'+name] = expected
    plan = read(old/'inputs/plan.json')[str(n)]
    chunks, estimate = estimate_merge(old, n, FROZEN_IDENTITY, cfg['host_memory_gib'])
    if (cpu.get('n') != n or cpu.get('raw') != n**3 or cpu.get('owners') != plan['n_owner']
        or cpu.get('chunks') != len(plan['chunks']) or len(chunks) != len(plan['chunks'])
        or set(cpu['chunk_receipts']) != {str(i) for i in range(len(chunks))}):
        raise ValueError('CPU full-domain coverage receipt mismatch')
    sr = read(science/f'references_N{n}.json'); binding = read(science/'binding.json')
    if (binding.get('identity') != SCIENCE_IDENTITY or binding.get('baseline_identity') != FROZEN_IDENTITY
        or sr.get('passed') is not True or sr.get('identity') != SCIENCE_IDENTITY
        or sr.get('n') != n or sr.get('owners') != plan['n_owner'] or sr.get('chunks') != len(chunks)):
        raise ValueError('scientific reference lineage/coverage mismatch')
    expected_files = {f'references/N{n}/chunk_{i:06d}.{ext}' for i in range(len(chunks)) for ext in ('json', 'npz')}
    if set(sr['files']) != expected_files:
        raise ValueError('reference file coverage mismatch')
    for i, (path, record) in enumerate(chunks):
        receipt_hash = sha(path/'stats.json')
        if cpu['chunk_receipts'][str(i)] != receipt_hash or record['owners'] != plan['chunks'][i]:
            raise ValueError('CPU chunk receipt/owner mismatch')
        if set(record['files']) != {'bank.npz', 'geometry.npz'}:
            raise ValueError('CPU chunk file manifest mismatch')
        hashes[f'chunk:{i}'] = receipt_hash
        ref_path = science/f'references/N{n}/chunk_{i:06d}.npz'
        rr = read(ref_path.with_suffix('.json'))
        for p in (ref_path, ref_path.with_suffix('.json')):
            if sha(p) != sr['files'][str(p.relative_to(science))]:
                raise ValueError('reference receipt/content changed')
        if (rr.get('passed') is not True or rr.get('identity') != SCIENCE_IDENTITY
            or rr.get('input_receipt') != receipt_hash or rr.get('owners') != record['owners']
            or rr.get('index') != i or rr.get('n') != n or rr.get('sha256') != sr['files'][str(ref_path.relative_to(science))]):
            raise ValueError('reference-to-bank linkage mismatch')
    hashes['scientific_reference_gate'] = sha(science/f'references_N{n}.json')
    # A missing old science completion receipt is immaterial: only checked R
    # payloads are consumed, not its former report/qualification status.
    return cfg, hashes, plan, estimate


def check_topology(bank, topology):
    """Map original raw order into merged owner-major order explicitly."""
    raw = np.asarray(bank.raw)
    np.testing.assert_array_equal(bank.raw_to_owner, topology.ro[raw])
    np.testing.assert_allclose(bank.diagnostics['slot_points'][:, 2], topology.pts[raw], rtol=0, atol=2e-14)
    np.testing.assert_allclose(bank.raw_weight, topology.rv[raw]/topology.vol[topology.ro[raw]], rtol=2e-13, atol=0)
    return topology.rv[raw], topology.vol.copy()


def saved_reference(path, owners, volume, radial):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'O', 'R', 'owners', 'volume', 'radial'}:
            raise ValueError('saved reference array manifest mismatch')
        np.testing.assert_array_equal(z['owners'], owners)
        np.testing.assert_allclose(z['volume'], volume, rtol=2e-13, atol=0)
        np.testing.assert_array_equal(z['radial'], radial)
        action = z['R']
    if action.shape != (22, len(owners), 31) or not np.isfinite(action).all():
        raise ValueError('saved R shape/finite mismatch')
    return action


def cached_reference(path, identity, builder):
    receipt = path.with_suffix('.json')
    if receipt.exists():
        record = read(receipt)
        if record.get('passed') is not True or record.get('identity') != identity or sha(path) != record.get('sha256'):
            raise ValueError('Q09 reference cache corrupted/stale')
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(FIELDS) or not all(np.isfinite(z[k]).all() for k in z.files):
                raise ValueError('invalid reference cache arrays')
            ref = ContinuumReference(*(z[k].copy() for k in FIELDS), record['diagnostics'])
        return ref
    # A payload without receipt is an interrupted write, never a valid reuse.
    ref = builder()
    atomic_npz(path, **{k: np.asarray(getattr(ref, k)) for k in FIELDS})
    atomic_json(receipt, dict(passed=True, identity=identity, sha256=sha(path), diagnostics=ref.diagnostics))
    return ref


def load(directory):
    """CLI factory: inputs.json -> checked full-domain PreparedProvider."""
    if not jax.config.jax_enable_x64:
        raise ValueError('Q09 preparation requires JAX x64')
    cfg, hashes, plan, estimate = audit(directory)
    cache = cfg['cache_directory']; cache.mkdir(parents=True, exist_ok=True)
    n = cfg['n']; old = cfg['baseline_run']; science = cfg['science_run']
    source = source_hash(); binding = digest(dict(inputs=hashes, source=source, n=n))
    with (cache/'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dest = cache/binding; dest.mkdir(exist_ok=True)
        bank, geometry, views, _ = merge_chunks(old, n, FROZEN_IDENTITY, cfg['host_memory_gib'])
        # Model files were checked before import; never write bytecode into OLD.
        prior = sys.dont_write_bytecode; sys.dont_write_bytecode = True
        try:
            model = load_geometry_model(old/'inputs')
            with jax.default_device(jax.devices('cpu')[0]):
                ctx, topology = model.context(n, cfg['canonical_root'])
                raw_volume, volume = check_topology(bank, topology)
                radial = (topology.pts[topology.order[topology.starts[:-1]], 0]*n).astype(int)
                refs = []
                for i, view in enumerate(views):
                    p = science/f'references/N{n}/chunk_{i:06d}.npz'
                    rr = read(p.with_suffix('.json'))
                    action = saved_reference(p, view.bank.owners, volume[view.bank.owners], radial[view.bank.owners])
                    key = digest(dict(binding=binding, bank=view.bank.identity, reference=rr['sha256']))
                    ref = cached_reference(dest/f'chunk_{i:06d}.npz', key,
                        lambda: ContinuumReference.from_saved_continuum(view.bank, view.geometry,
                            lambda points: model.geom(ctx, points), action, rr['diagnostics']))
                    # Check saved-cache data order before final concatenation.
                    np.testing.assert_array_equal(ref.owner_raw, view.bank.owner_raw)
                    np.testing.assert_array_equal(ref.owner_weight, view.bank.owner_weight)
                    refs.append(ref)
                    if (i+1) % 25 == 0 or i+1 == len(views):
                        print(f'Q09 reference inputs N{n}: {i+1}/{len(views)}', flush=True)
                ref = ContinuumReference(*(np.concatenate([np.asarray(getattr(r, k)) for r in refs], axis=0)
                    for k in FIELDS[:6]), bank.owner_raw, bank.owner_weight,
                    {'chunks': [r.diagnostics for r in refs], 'saved_reference_identity': SCIENCE_IDENTITY})
                mms = Manufactured.from_members(bank, bank.diagnostics['slot_points'][:, 2],
                    bank.raw_to_owner, raw_volume, volume)
                mms = replace(mms, owner_initial=np.asarray(mms.owner_initial), phi_initial=np.asarray(mms.phi_initial),
                    boundary_initial=jax.tree.map(np.asarray, mms.boundary_initial),
                    boundary_constant=jax.tree.map(np.asarray, mms.boundary_constant))
                regions = dict(zip(REGIONS[1:], masks(radial, n, plan['last_aggregate'])[1:], strict=True))
                hashes['prepared_bank_identity'] = bank.identity
                provider = PreparedProvider(bank, geometry, mms, ref, volume, regions, hashes).validate()
        finally:
            sys.dont_write_bytecode = prior
        if source_hash() != source:
            raise ValueError('Q09 source changed during preparation; no prepared receipt emitted')
        atomic_json(dest/'prepared.json', dict(passed=True, n=n, provider_identity=provider.identity,
            owner_count=len(bank.owners), raw_count=len(bank.raw), merge_estimate=estimate,
            source_identity=source, independent_reference=True, new_tracing=False,
            scientific_evolution_performed=False))
        return provider


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'prepare'))
    p.add_argument('--inputs', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'audit':
        cfg, hashes, plan, estimate = audit(args.inputs)
        print(json.dumps(dict(n=cfg['n'], passed=True, inputs=hashes, estimate=estimate,
            chunks=len(plan['chunks']), evolution_started=False), indent=2))
    else:
        provider = load(args.inputs)
        print(json.dumps(dict(passed=True, provider_identity=provider.identity, evolution_started=False)))


if __name__ == '__main__':
    main()
