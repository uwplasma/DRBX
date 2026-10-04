"""Input lineage/corruption tests, plus bounded real-bank reference wiring."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts.q09_evolved_mms import inputs as a
from scripts.q09_evolved_mms.mms import ContinuumReference
from scripts.q08_extraction_global.common import sha, digest, atomic_json


@pytest.fixture
def receipt_tree(tmp_path, monkeypatch):
    """Tiny receipt-only mock; never presented as a numerical HSX full bank."""
    source, old, sci, canonical, config = (tmp_path/x for x in ('source', 'old', 'science', 'canonical', 'config'))
    for p in (source, old, sci, canonical, config): p.mkdir()
    def put(p, data):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return sha(p)
    cfg = dict(geometry='geometry', metric_cache='metric.npz', makegrid='makegrid.nc')
    files = {}
    model = json.dumps(cfg).encode()
    files['geometry_model/inputs.json'] = put(old/'inputs/geometry_model/inputs.json', model)
    files['geometry_model/model.py'] = put(old/'inputs/geometry_model/model.py', b'# checked only; never imported\n')
    plan = {'32': dict(n_owner=2, n_raw=32**3, last_aggregate=10, chunks=[[0, 1]])}
    files['plan.json'] = put(old/'inputs/plan.json', json.dumps(plan).encode())
    names = ['metric.npz', 'makegrid.nc', 'geometry/32x32x32/base_geometry.npz', 'geometry/32x32x32/rlp_topology.npz']
    canon = {name: put(canonical/name, name.encode()) for name in names}
    atomic_json(source/'input_manifest.json', dict(files=files, canonical=canon))
    design = dict(input_manifest_sha256=sha(source/'input_manifest.json'))
    identity = digest(design)
    atomic_json(source/'design.json', design); atomic_json(old/'provenance/design.json', design)
    monkeypatch.setattr(a, 'BASE_SOURCE', source); monkeypatch.setattr(a, 'FROZEN_IDENTITY', identity)
    chunk = old/'cpu/N32/chunk_000000'; chunk.mkdir(parents=True)
    values = {name: put(chunk/name, name.encode()) for name in ('bank.npz', 'geometry.npz')}
    record = dict(passed=True, campaign_identity=identity, n=32, index=0, owners=[0, 1], files=values)
    atomic_json(chunk/'stats.json', record)
    atomic_json(old/'verification.json', dict(passed=True, campaign_identity=identity, canonical=canon))
    atomic_json(old/'cpu_N32.json', dict(passed=True, campaign_identity=identity, n=32,
        owners=2, raw=32**3, chunks=1, chunk_receipts={'0': sha(chunk/'stats.json')}))
    atomic_json(sci/'binding.json', dict(identity=a.SCIENCE_IDENTITY, baseline_identity=identity))
    reference = sci/'references/N32/chunk_000000.npz'; reference.parent.mkdir(parents=True)
    np.savez(reference, R=np.zeros((22, 2, 31)), O=np.zeros((2, 22, 2, 31)),
        owners=np.arange(2), volume=np.ones(2), radial=np.arange(2))
    atomic_json(reference.with_suffix('.json'), dict(passed=True, identity=a.SCIENCE_IDENTITY,
        input_receipt=sha(chunk/'stats.json'), owners=[0, 1], n=32, index=0,
        sha256=sha(reference)))
    atomic_json(sci/'references_N32.json', dict(passed=True, identity=a.SCIENCE_IDENTITY,
        n=32, owners=2, chunks=1,
        files={str(p.relative_to(sci)): sha(p) for p in (reference, reference.with_suffix('.json'))}))
    atomic_json(config/'inputs.json', dict(schema='q09-prepared-inputs-v1', n=32,
        baseline_run=str(old), science_run=str(sci), canonical_root=str(canonical),
        cache_directory=str(tmp_path/'cache'), host_memory_gib=20))
    def estimate(root, n, ident, budget):
        from scripts.q08_extraction_global.gpu import checked_chunks
        return checked_chunks(root, n, ident), {'mock_headers_only': True}
    monkeypatch.setattr(a, 'estimate_merge', estimate)
    return config, old, sci, canonical


def test_receipt_lineage_without_science_completion(receipt_tree):
    config, old, sci, canonical = receipt_tree
    cfg, hashes, plan, _ = a.audit(config)
    assert cfg['n'] == 32 and plan['n_owner'] == 2
    assert hashes['scientific_reference_gate'] == sha(sci/'references_N32.json')
    assert not (sci/'completion.json').exists()  # Neither fabricated nor required.


@pytest.mark.parametrize('target', ['canonical', 'model', 'bank', 'reference', 'cpu_gate'])
def test_content_mutations_rejected(receipt_tree, target):
    config, old, sci, canonical = receipt_tree
    if target == 'cpu_gate':
        p = old/'cpu_N32.json'; data = a.read(p); data['chunk_receipts']['0'] = '0'*64; atomic_json(p, data)
    else:
        p = dict(canonical=canonical/'makegrid.nc', model=old/'inputs/geometry_model/model.py',
            bank=old/'cpu/N32/chunk_000000/bank.npz', reference=sci/'references/N32/chunk_000000.npz')[target]
        b = bytearray(p.read_bytes()); b[0] ^= 1; p.write_bytes(b)
    with pytest.raises(ValueError): a.audit(config)


def test_reference_bank_link_not_just_payload_hash(receipt_tree):
    config, old, sci, _ = receipt_tree
    p = sci/'references/N32/chunk_000000.json'; record = a.read(p)
    record['input_receipt'] = '0'*64; atomic_json(p, record)
    manifest = a.read(sci/'references_N32.json'); manifest['files'][str(p.relative_to(sci))] = sha(p)
    atomic_json(sci/'references_N32.json', manifest)
    with pytest.raises(ValueError, match='reference-to-bank'): a.audit(config)


def test_input_cache_separation_and_path_escape(receipt_tree):
    directory, old, sci, _ = receipt_tree
    with pytest.raises(ValueError, match='escapes'): a.child(old, '../science/anything')
    cfg = a.read(directory/'inputs.json'); cfg['cache_directory'] = str(sci/'cache')
    atomic_json(directory/'inputs.json', cfg)
    with pytest.raises(ValueError, match='separate'): a.config(directory)


def test_raw_topology_permutation():
    # The adapter must index raw topology by bank.raw, not by merge position.
    raw = np.array([2, 0, 1]); ro = np.array([0, 1, 0]); rv = np.array([1., 2., 3.])
    pts = np.arange(9).reshape(3, 3)*.01; vol = np.array([4., 2.])
    topology = SimpleNamespace(ro=ro, rv=rv, vol=vol, pts=pts)
    bank = SimpleNamespace(raw=raw, raw_to_owner=ro[raw], raw_weight=rv[raw]/vol[ro[raw]],
        diagnostics={'slot_points': np.repeat(pts[raw, None], 5, axis=1)})
    weights, volume = a.check_topology(bank, topology)
    np.testing.assert_array_equal(weights, rv[raw]); np.testing.assert_array_equal(volume, vol)
    bank.raw_to_owner = ro
    with pytest.raises(AssertionError): a.check_topology(bank, topology)


def test_reference_cache_reuse_and_corruption(tmp_path):
    ref = ContinuumReference(np.ones((2, 6)), np.ones((2, 6)), np.ones(2), np.ones((2, 6)),
        np.ones(2), np.ones(2), np.arange(2)[:, None], np.ones((2, 1)), {'control': True})
    p = tmp_path/'chunk.npz'; calls = []
    def build(): calls.append(1); return ref
    a.cached_reference(p, 'fixed', build); again = a.cached_reference(p, 'fixed', build)
    assert calls == [1]; np.testing.assert_array_equal(again.values, ref.values)
    with pytest.raises(ValueError, match='stale'): a.cached_reference(p, 'changed', build)
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='corrupted'): a.cached_reference(p, 'fixed', build)


def test_prepared_payload_stages_all_runtime_arrays():
    import jax
    from scripts.q09_evolved_mms.mms import Manufactured
    from scripts.q09_evolved_mms.evolution import q_payload
    ref = ContinuumReference(np.ones((2, 6)), np.ones((2, 6)), np.ones(2), np.ones((2, 6)),
        np.ones(2), np.ones(2), np.arange(2)[:, None], np.ones((2, 1)), {})
    mms = Manufactured(np.ones((6, 2)), np.ones(2), (np.zeros(2),), (np.zeros(2),))
    provider = SimpleNamespace(reference=ref, manufactured=mms, volume=np.ones(2),
        plan={'algebra_only': np.ones(3)})
    payload = q_payload(provider)
    for leaf in jax.tree.leaves(payload):
        assert isinstance(leaf, jax.Array)
        assert leaf.devices() == {jax.devices()[0]}


def test_saved_continuum_replay_and_independence():
    from drbx.stencils.q_parallel import load_chunk
    from drbx.stencils.q_bank import build_q_bank
    from scripts.q08_rhs_mms_global.science import continuum
    root = Path(__file__).resolve().parents[1]/'scripts/q08_extraction_global/inputs/bounded'
    bank = build_q_bank(*(load_chunk(root/f'N32_h{d}.npz') for d in (16, 32)))
    # Algebra-only geometry control on actual bank topology. Real C3 is checked
    # separately with canonical inputs, not implied by this portable control.
    def geom(p): return np.ones(len(p)), np.broadcast_to([.1, .2, 1.], (len(p), 3)), np.ones(len(p))*2
    geometry = dict(b_eta=np.ones(len(bank.raw)), bmag=np.ones(len(bank.raw))*2,
        magnetic_L=np.zeros((len(bank.raw), 3)))
    action, diagnostics = continuum(bank, geometry, geom)
    ref = ContinuumReference.from_saved_continuum(bank, geometry, geom, action, diagnostics)
    rebuilt = ContinuumReference.prepare(bank, geometry, geom)
    for t in (0., .4, 1.):
        np.testing.assert_allclose(ref.rhs(t, 'complete'), rebuilt.rhs(t, 'complete'), atol=1e-11, rtol=1e-12)
    # Poisoning diffusion rows cannot affect the independent reference.
    bank.arrays['diffusion_D'][:] = np.nan
    repeat = ContinuumReference.from_saved_continuum(bank, geometry, geom, action, diagnostics)
    np.testing.assert_array_equal(repeat.diffusion, ref.diffusion)
    bad = action.copy(); bad[1, :, 12] += .001
    with pytest.raises(AssertionError):
        ContinuumReference.from_saved_continuum(bank, geometry, geom, bad, diagnostics)
