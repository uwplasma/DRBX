"""Local-only export of frozen direct-cubic inputs; no global reference run."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare import prototype
from .common import FIELDS, atomic_json, sha
from .numerics import POLICY, centered_owner_moments


def export(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists():
        raise RuntimeError('seed export requires a new directory')
    output.mkdir(parents=True)
    c = prototype(workspace)
    provenance = {}
    for N in (32, 48, 64):
        ctx = c.compact.load_context(N)
        xyz, labels, rv, volume, centers, planes, _, _, states = c.owner_data(ctx)
        d = c.data(N)
        old = json.loads((c.INPUT/f'N{N}/manifest.json').read_text())
        assert sha(ctx['manifest']) == old['source_inputs']['reference_identity']['reference']['geometry_manifest_sha256']
        np.testing.assert_array_equal(volume, d['volume'])
        keys = ['volume','target'] + ['face.'+k for k in ('center','scale','minus','plus','boundary','plane')]
        keys += sorted(k for k in d if k.startswith('region.'))
        for k in keys:
            assert sha(c.INPUT/f'N{N}/{k}.npy') == old['files'][k+'.npy']['sha256'], k
        dest = output/f'N{N}'; dest.mkdir()
        payload = {k: d[k] for k in keys}
        payload['target'] = d['target'][:, :20]
        payload.update({'owner.centers':centers, 'owner.planes':planes, 'owner.states':states,
            'owner.moments':centered_owner_moments(xyz, labels, rv, volume, centers, N),
            'complete_owners':np.arange(len(volume)), 'face.ids':np.arange(len(d['face.minus']))})
        for axis in 'xyz':
            payload['grid.'+axis] = np.asarray(getattr(ctx['artifact'].geometry.grid,axis).faces)
        face_path = workspace/f'work/parallel_q03_compact_corrected_20260920/N{N}/canonical_faces.npz'
        with np.load(face_path) as faces:
            for k in ('minus','plus','boundary'):
                np.testing.assert_array_equal(faces[k], d['face.'+k])
            for k in ('axis','storage'):
                payload['canonical.'+k] = faces[k]
        for k,a in payload.items():
            np.save(dest/(k+'.npy'),a,allow_pickle=False)
        sample = workspace/f'work/parallel_q03_direct_owner_flux_20260921/N{N}'
        with np.load(sample/'comparison.npz') as z, np.load(sample/'qualified_reference.npz') as ref:
            np.savez_compressed(dest/'preflight.npz', faces=z['faces'], owners=z['owners'],
                direct=z['direct'], flux=z['flux'], q9=ref['flux_q9'], q11=ref['flux_q11'])
        fields = {f:ctx['fields'][f]['manifest'] for f in FIELDS}
        meta = {'resolution':N, 'policy':POLICY, 'field_manifests':fields,
            'geometry_sha256':sha(ctx['manifest']), 'frozen_manifest_sha256':sha(c.INPUT/f'N{N}/manifest.json'),
            'canonical_faces_sha256':sha(face_path),
            'currents':fields[FIELDS[0]]['provenance']['makegrid_currents'],
            'prototype_sha256':sha(sample/'comparison.npz'),
            'qualified_reference_sha256':sha(sample/'qualified_reference.npz')}
        atomic_json(dest/'metadata.json',meta)
        provenance[str(N)] = meta
        print(json.dumps({'exported':N,'faces':len(d['face.minus'])}),flush=True)
    # These existing immutable files stay outside the transfer bundle.
    first = provenance['32']['field_manifests'][FIELDS[0]]['provenance']
    metric = workspace/'.hsx_metric_cache/hsx_metric_7f6f0883c40ac27342736ebc.npz'
    # The historical Q01 catalogue records the original cache. The audited
    # direct prototype now resolves that alias to the axis-corrected d58 cache.
    # Freeze the actual continuous evaluator, retaining the old catalogue as
    # source-state provenance, never as the new reference's identity.
    assert metric.resolve() == (workspace/'hsx_metric_d58d392545fd3917efeb83b6.npz').resolve()
    external = {'metric_cache':{'filename':'hsx_metric_d58d392545fd3917efeb83b6.npz', 'sha256':sha(metric)},
                'makegrid':{'filename':'mgrid_res2p5cm_180pln.nc','sha256':first['makegrid_sha256']}}
    for rec in external.values():
        assert sha(workspace/rec['filename']) == rec['sha256']
    atomic_json(output/'manifest.json', {'schema':'q03-direct-seeds-v1','policy':POLICY,'external':external,
        'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.workspace,a.output)
