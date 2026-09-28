"""CPU correctness coverage for the GPU trace kernel, interchange and scheduler.

These tests make no GPU performance or CPU/GPU numerical equivalence claim.
"""
from dataclasses import dataclass
import threading
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.q_fci_shortspan_midpoint_global import gpu_trace as gpu
from scripts.q_fci_shortspan_midpoint_global import trace_store as store
from scripts.q_fci_shortspan_midpoint_global.span_contract import LEG_FACTORS, make_seeds
from scripts.q_fci_shortspan_midpoint_global.trace import trace_fixed_batch, trace_padded


class Metric(NamedTuple):
    def position_and_jacobian(self, q):
        return q, jnp.broadcast_to(jnp.eye(3), (len(q), 3, 3))


class Bfield(NamedTuple):
    R: object
    Z: object

    def evaluate_cartesian(self, q):
        return jnp.broadcast_to(jnp.array([.05, .02, 1.]), q.shape)


class Field(NamedTuple):
    metric: Metric
    bfield: Bfield


def test_fixed_batch_preserves_nine_outputs_and_padding():
    cpu = jax.local_devices(backend='cpu')[0]
    field = jax.device_put(Field(Metric(), Bfield(np.array([0., 20.]), np.array([-20., 20.]))), cpu)
    # Both signs, seam, near axis, wall crossing, plus an invalid zero-radius
    # control. Invalid rows must remain invalid; padding must not escape.
    seeds = np.array([[.4, .3, .2], [.6, 6.28, .3], [.001, .1, .2],
                      [.999, .1, .3], [.999, .1, .3], [0., .1, .2]])
    delta = np.array([.1, -.1, .001, .1, -.1, .1])
    expected = trace_padded(field, seeds, delta, steps=64)
    actual = trace_fixed_batch(field, seeds, delta, 64, 192, cpu)
    split = [trace_fixed_batch(field, seeds[i:i+2], delta[i:i+2], 64, 96, cpu)
             for i in range(0, len(seeds), 2)]
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=3e-13, atol=3e-13, equal_nan=True)
        np.testing.assert_allclose(np.concatenate([x[i] for x in split]), b,
                                   rtol=3e-13, atol=3e-13, equal_nan=True)
    assert actual[2][3] and not actual[2][4]
    assert not actual[1][-1]
    np.testing.assert_allclose(actual[0][:5, 0], seeds[:5, 0]+.05*delta[:5], atol=1e-12)
    with pytest.raises(ValueError):
        trace_fixed_batch(field, seeds, delta, 64, 2, cpu)


def grid(n=32):
    faces = [np.linspace(0, 1, n+1), np.linspace(0, 2*np.pi, n+1), np.linspace(0, 2*np.pi, n+1)]
    centers = [(a[1:]+a[:-1])/2 for a in faces]
    return centers, faces


def test_seed_and_leg_order_matches_inline_contract():
    n = 32
    centers, faces = grid(n)
    raw = [0, 111, 25*n*n, n**3-1]
    seeds = store.seed_batch(centers, faces, n, raw)
    legs, delta = store.pack_legs(seeds, n)
    for idx, cell in enumerate(raw):
        ijk = np.unravel_index(cell, (n, n, n))
        q = np.array([centers[a][ijk[a]] for a in range(3)])
        w = np.array([faces[a][ijk[a]+1]-faces[a][ijk[a]] for a in range(3)])
        expected, _ = make_seeds(q, w)
        if ijk[0] < n-7:
            expected = expected[np.r_[6:12, 0:6]]
        np.testing.assert_array_equal(seeds[idx], expected)
        np.testing.assert_array_equal(legs[48*idx:48*(idx+1)], np.tile(expected, (4, 1)))
        np.testing.assert_array_equal(delta[48*idx:48*(idx+1)], np.repeat(LEG_FACTORS*(2*np.pi/n), 12))


@dataclass(frozen=True)
class Device:
    id: int
    platform: str = 'gpu'
    device_kind: str = 'test device'


def fake_trace(field, seeds, deltas, steps, capacity, device):
    assert len(seeds) <= capacity
    end = seeds.copy()
    end[:, 2] += deltas
    count = len(end)
    return (end, np.ones(count, bool), np.zeros(count, bool), np.zeros(count, bool),
            seeds[:, 0].copy(), np.ones(count), np.full(count, -1),
            seeds.copy(), np.full((count, 3), np.nan))


def make_store(tmp_path, raw, checks=(), batch=2, steps=64):
    stage = tmp_path/'global'
    stage.mkdir()
    store.write(stage/'design.json', dict(trace_mode='gpu_cache', method=dict(rk4_steps=steps),
                                         rk4_512_raw_ids={'32': list(checks)}))
    root, groups = store.freeze_store(stage, 32, raw, batch)
    return stage, root, groups


def test_four_device_queue_resume_and_partial_chunks(tmp_path):
    # Noncontiguous IDs include a one-cell final block and cross owner ordering.
    raw = [0, 1, 2, 3, 4, 5, 6, 7, 11]
    stage, root, groups = make_store(tmp_path, raw, checks=(0, 6))
    centers, faces = grid()
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    seen = set()
    calls = []

    def trace(field, seeds, deltas, steps, capacity, device):
        with lock:
            first = device.id not in seen
            seen.add(device.id)
            calls.append((device.id, steps, len(seeds), capacity))
        if first:
            barrier.wait(timeout=10)
        return fake_trace(field, seeds, deltas, steps, capacity, device)

    devices = [(Device(i), None) for i in range(4)]
    result = gpu.trace_queue(root, 32, centers, faces, groups, devices, trace)
    assert seen == {0, 1, 2, 3} and result['computed'] == 5
    assert sum(x[1] == 512 for x in calls) == 2
    reader = store.TraceStore(stage, 32, max_chunks=1)
    reader.require_raw_ids(raw)
    with pytest.raises(RuntimeError, match='cover'):
        reader.require_raw_ids([10])
    for cell in reversed(raw):
        seeds = store.seed_batch(centers, faces, 32, [cell])[0]
        outputs, check = reader.get(cell, seeds)
        legs, delta = store.pack_legs(seeds[None], 32)
        np.testing.assert_array_equal(outputs[0][:, 2], legs[:, 2]+delta)
        assert (check is not None) == (cell in (0, 6))
        assert len(reader.cache) <= 1
    with pytest.raises(RuntimeError, match='seed ordering'):
        reader.get(0, store.seed_batch(centers, faces, 32, [1])[0])

    def forbidden(*args):
        raise AssertionError('resume retraced a completed chunk')

    rerun = gpu.trace_queue(root, 32, centers, faces, groups, devices, forbidden)
    assert rerun['computed'] == 0 and rerun['skipped'] == 5
    assert not list(root.rglob('*.lock'))
    # Changing capacity or step count must not relabel existing output.
    with pytest.raises(RuntimeError, match='design differs'):
        store.freeze_store(stage, 32, raw, 4)
    d = store.load(stage/'design.json')
    d['method']['rk4_steps'] = 256
    store.write(stage/'design.json', d)
    with pytest.raises(RuntimeError, match='identity differs'):
        store.TraceStore(stage, 32)


def test_trace_store_rejects_missing_corrupt_and_stale_chunks(tmp_path):
    stage, root, groups = make_store(tmp_path, [0, 1])
    centers, faces = grid()
    with pytest.raises(FileNotFoundError):
        store.TraceStore(stage, 32)
    path = store.chunk_path(root, 32, 0)
    with store.exclusive(path.with_suffix('.lock')):
        with pytest.raises(FileExistsError):
            gpu.trace_queue(root, 32, centers, faces, groups, [(Device(0), None)], fake_trace)
    gpu.trace_queue(root, 32, centers, faces, groups, [(Device(0), None)], fake_trace)
    receipt = path.with_suffix('.json')
    original = receipt.read_bytes()
    receipt.unlink()
    with pytest.raises(RuntimeError, match='interrupted'):
        store.completed(root, 32, 0, [0, 1])
    receipt.write_bytes(original)
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(RuntimeError, match='corrupt'):
        store.TraceStore(stage, 32).get(0, store.seed_batch(centers, faces, 32, [0])[0])


def test_failed_trace_does_not_publish_complete_stage(tmp_path):
    stage, root, groups = make_store(tmp_path, [0, 1])
    centers, faces = grid()

    def bad(*args):
        output = list(fake_trace(*args))
        output[3][0] = True
        return tuple(output)

    with pytest.raises(RuntimeError, match='reentry'):
        gpu.trace_queue(root, 32, centers, faces, groups, [(Device(0), None)], bad)
    assert not (root/'complete_N32.json').exists()
    assert not list(root.rglob('*.lock'))


def test_device_selection_requires_requested_gpus():
    devices = [Device(i) for i in range(4)]
    assert gpu.select_devices(devices, [0, 1, 2, 3]) == devices
    with pytest.raises(ValueError):
        gpu.select_devices(devices, [0, 0])
    with pytest.raises(RuntimeError, match='visible'):
        gpu.select_devices(devices[:1], [0, 1, 2, 3])
    with pytest.raises(RuntimeError, match='never falls back'):
        gpu.select_devices([Device(0, 'cpu')], [0])


def test_grid_plan_keeps_all_aggregate_members(tmp_path):
    n = 2
    directory = tmp_path/'geometry'/'2x2x2'
    directory.mkdir(parents=True)
    centers, faces = grid(n)
    arrays = {}
    for i, axis in enumerate('xyz'):
        arrays[f'grid.{axis}.centers'] = centers[i]
        arrays[f'grid.{axis}.faces'] = faces[i]
    np.savez(directory/'base_geometry.npz', **arrays)
    active = np.zeros(8, bool)
    active[[0, 3, 4]] = True
    np.savez(directory/'rlp_topology.npz', is_active_owner=active.reshape(2, 2, 2),
             aggregate_id=np.array([0, 0, 0, 3, 4, 4, 4, 4]).reshape(2, 2, 2))
    _, _, labels = gpu.load_grid(tmp_path, n, {'geometry': 'geometry'})
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(np.flatnonzero(np.isin(labels, [0, 2])), [0, 1, 2, 4, 5, 6, 7])


def test_trace_gates_reject_precision_loss_and_inconsistent_crossing():
    q = np.array([[.5, .2, .1]])
    output = list(fake_trace(None, q, np.array([.01]), 64, 64, Device(0)))
    output[0] = output[0].astype(np.float32)
    with pytest.raises(RuntimeError, match='float64'):
        store.validate_trace(tuple(output), 32)
    output[0] = output[0].astype(np.float64)
    output[6][0] = 7
    with pytest.raises(RuntimeError, match='crossing flags'):
        store.validate_trace(tuple(output), 32)


def test_gpu_entry_verifies_frozen_sources_and_inputs(monkeypatch, tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    source = repo/'kernel.py'
    source.write_text('original source')
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    data = inputs/'geometry.bin'
    data.write_bytes(b'geometry')
    campaign = tmp_path/'campaign'
    stage = campaign/'pilot'
    stage.mkdir(parents=True)
    design = dict(trace_mode='gpu_cache', method=dict(rk4_steps=64),
                  sources={'kernel.py': store.sha(source)},
                  input_manifest={'files': [dict(path='geometry.bin', bytes=data.stat().st_size,
                                                 sha256=store.sha(data))]})
    store.write(stage/'design.json', design)
    store.write(campaign/'dispatch.json', dict(source_design_sha256={'pilot': store.sha(stage/'design.json')},
                                             stages={'pilot': {'32': [[1, 7]]}}))
    monkeypatch.setattr(gpu, 'REPO', repo)
    assert gpu.verify_stage(campaign, 'pilot', 32, inputs)[2] == [1, 7]
    source.write_text('changed source')
    with pytest.raises(RuntimeError, match='source differs'):
        gpu.verify_stage(campaign, 'pilot', 32, inputs)
    source.write_text('original source')
    data.write_bytes(b'corrupt!')
    with pytest.raises(RuntimeError, match='input differs'):
        gpu.verify_stage(campaign, 'pilot', 32, inputs)


def test_preflight_comparison_respects_theta_seam_and_step_index():
    from scripts.q_fci_shortspan_midpoint_global.gpu_preflight import compare_trace
    q = np.array([[.5, .2, .1]])
    a = list(fake_trace(None, q, np.array([.01]), 64, 64, Device(0)))
    b = [x.copy() for x in a]
    b[0][:, 1] += 2*np.pi
    assert compare_trace(a, b, same_steps=True) < 1e-15
    a[6][0] = 3
    b[6][0] = 12
    assert compare_trace(a, b, same_steps=False) < 1e-15
    with pytest.raises(RuntimeError, match='crossing step'):
        compare_trace(a, b, same_steps=True)
    b[2][0] = True
    with pytest.raises(RuntimeError, match='classification'):
        compare_trace(a, b, same_steps=False)


def test_preflight_action_comparison_rejects_nonfinite_values():
    from scripts.q_fci_shortspan_midpoint_global.gpu_preflight import scaled_defect
    assert scaled_defect([1+1j], [1+1j]) == 0
    assert scaled_defect([1.], [0.]) == 1
    with pytest.raises(RuntimeError, match='nonfinite'):
        scaled_defect([np.nan], [0.])
