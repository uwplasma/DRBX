"""Pilot bookkeeping on a known ODE, not a scientific HSX evolution claim."""
from types import SimpleNamespace
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.q09_evolved_mms import campaign as c
from scripts.q09_evolved_mms.cli import _run
from scripts.q09_evolved_mms.evolution import advance, stage_stepper, step_count
from scripts.q09_evolved_mms.provider import save_checkpoint, load_checkpoint


@pytest.mark.parametrize('level,expected', [(0, 10), (1, 20), (2, 40)])
def test_pilot_step_grid_and_interrupted_replay(tmp_path, level, expected):
    dt = c.DT/2**level
    assert step_count(0., c.END, dt) == expected
    initial = np.ones((6, 2)); volume = np.array([.3, .7])
    stepper = stage_stepper(lambda x, t: (-x, True), lambda t: jnp.zeros_like(initial), volume)
    args = dict(start=0., end=c.END, dt=dt, volume=volume, target=lambda t: initial*np.exp(-t))
    compiled = jax.jit(lambda s, t, h, data: stepper(s, time=t, timestep=h, carry=data))
    path = tmp_path/'checkpoint.npz'
    advance(stepper, initial, **args, max_steps=3, compiled_step=compiled,
            checkpoint=lambda a: save_checkpoint(path, 'fixed', a))
    partial = load_checkpoint(path, 'fixed')
    resumed, report = advance(stepper, initial, **args, resume=partial, compiled_step=compiled)
    fresh, direct = advance(stepper, initial, **args, compiled_step=compiled)
    np.testing.assert_array_equal(resumed, fresh)
    assert report == direct
    assert report['accepted_steps'] == expected and report['actual_end'] == c.END
    assert len(report['stage_times']) == 4*expected
    assert report['stage_times'][-1] == c.END
    assert step_count(0., 1.03e-5, dt) == int(np.ceil(1.03e-5/dt))


def control_case(tmp_path):
    initial = np.ones((6, 2)); volume = np.array([.2, .8])
    target = lambda t: initial*np.exp(-t)
    provider = SimpleNamespace(manufactured=SimpleNamespace(state=target),
        volume=volume, regions={'left': np.array([True, False])})
    args = SimpleNamespace(output=tmp_path, dt=c.DT, end=c.END, mode='diffusion',
                           kinds='DDDDDD', phi_kind='D', resume=True)
    stepper = stage_stepper(lambda x, t: (-x, True), lambda t: jnp.zeros_like(initial), volume)
    _run(args, provider, None, 'control', stepper)
    observations = dict(initial=initial, target=target(c.END), volume=volume,
                        **{'region:left': provider.regions['left']})
    return observations, args, provider, stepper


def test_independent_case_reduction_and_resume(tmp_path):
    obs, args, provider, stepper = control_case(tmp_path)
    report = c.validate_case(tmp_path, 'control', obs, 'diffusion', 'DDDDDD', 'D')
    assert [r['accepted_steps'] for r in report['runs']] == [10, 20, 40]
    _run(args, provider, None, 'control', stepper)
    assert c.validate_case(tmp_path, 'control', obs, 'diffusion', 'DDDDDD', 'D') == report
    timings = c.read(tmp_path/'timings.json')
    assert [r['resumed_steps'] for r in timings['levels']] == [10, 20, 40]
    assert all(not r['synchronized_step_including_validation'] for r in timings['levels'])
    history = [json.loads(line) for line in (tmp_path/'timing_history.jsonl').read_text().splitlines()]
    assert sum(r['event'] == 'begin' for r in history) == 2
    assert sum(r['event'] == 'step' for r in history) == 70


@pytest.mark.parametrize('target', ['solution_error', 'integral', 'temporal', 'time'])
def test_validator_rejects_bad_reductions(tmp_path, target):
    obs, _, _, _ = control_case(tmp_path)
    report = c.read(tmp_path/'report.json')
    if target == 'solution_error': report['runs'][0]['error']['global']['rms'][0] += .01
    elif target == 'integral': report['runs'][0]['integrated_rhs'][0] += .01
    elif target == 'temporal': report['temporal_self_convergence']['self_difference_rms'][0][0] += .01
    else: report['runs'][0]['actual_end'] = c.END/2
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises((ValueError, AssertionError)):
        c.validate_case(tmp_path, 'control', obs, 'diffusion', 'DDDDDD', 'D')


def test_gpu_memory_admission_uses_selected_device_and_stops(monkeypatch):
    # Resource-gate control only; no emulation of numerical GPU qualification.
    device = SimpleNamespace(id=0, local_hardware_id=2,
        memory_stats=lambda: dict(bytes_in_use=1024, bytes_limit=80*(1 << 30)))
    monkeypatch.setattr(jax, 'devices', lambda: [device])
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *a, **k:
        '0, GPU-other, A100, 81920, 1\n2, GPU-selected, A100, 81920, 70000\n')
    estimate = dict(estimated_gpu0_peak_upper_bytes=1 << 30,
                    one_case_state_bytes=1024, one_case_boundary_bytes=1024)
    assert c.device_memory_gate(estimate)['estimated_peak_upper_bytes'] == (1 << 30)+14*1024
    estimate['estimated_gpu0_peak_upper_bytes'] = 100*(1 << 30)
    with pytest.raises(MemoryError): c.device_memory_gate(estimate)


def test_cpu_cannot_be_reported_as_gpu():
    if jax.default_backend() == 'cpu':
        with pytest.raises(ValueError, match='actual GPU'): c.hardware()
