"""Stage-aware shared RK4 adapter and finite-run research diagnostics."""
from dataclasses import dataclass
from typing import NamedTuple
import time
import jax
import jax.numpy as jnp
import numpy as np
from drbx.native.fci_model import FciModelState
from drbx.native.fci_time_integrator import Rk4Stepper
from drbx.native.q_plan import apply_q_plan, _diffusion
from .mms import Manufactured, ContinuumReference
from .mms import COEFFICIENTS


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SixState(FciModelState):
    n: object
    Te: object
    Ti: object
    Vi: object
    Ve: object
    omega: object

    @classmethod
    def from_array(cls, array):
        if jnp.shape(array)[0] != 6 or jnp.ndim(array) != 2:
            raise ValueError('expected (6,owners) state')
        return cls(*array)

    def array(self):
        return jnp.stack(self.field_values())


class StageDiagnostics(NamedTuple):
    time: object
    valid: object
    positive: object
    finite: object
    rhs_integral: object


def stage_stepper(numerical, source, volume=None):
    """Callbacks get current state/time; numerical returns (R, admissible)."""
    def rhs(state, t, carry):
        x = state.array()
        value, valid = numerical(x, t)
        forcing = source(t)
        total = value+forcing
        positive = jnp.all(x[:3] > 0)
        finite = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(total)) & jnp.all(jnp.isfinite(forcing))
        return SixState.from_array(total), carry, StageDiagnostics(t, jnp.all(valid), positive, finite, total@jnp.asarray(volume) if volume is not None else jnp.sum(total, axis=1))
    return Rk4Stepper(rhs)


class QPayload(NamedTuple):
    plan: object
    owner_initial: object
    phi_initial: object
    boundary_initial: object
    boundary_constant: object
    reference_arrays: object
    volume: object


def q_payload(provider):
    r = provider.reference; m = provider.manufactured
    payload = QPayload(provider.plan, m.owner_initial, m.phi_initial, m.boundary_initial, m.boundary_constant,
        tuple(getattr(r, k) for k in ('values', 'gradients', 'phi_gradient', 'diffusion',
            'kappa', 'bmag', 'owner_raw', 'owner_weight')), jnp.asarray(provider.volume))
    # Preparation may run on CPU; all live RHS leaves are explicitly staged
    # onto the selected application device. Never mix committed CPU/GPU inputs.
    return jax.tree.map(lambda a: jax.device_put(a, jax.devices()[0]), payload)


def q_stepper(mode, kinds, phi_kind):
    """Large prepared arrays are dynamic carry leaves, never XLA constants."""
    if mode not in ('diffusion', 'complete'):
        raise ValueError('mode must be diffusion or complete')
    if len(kinds) != 6 or any(k not in ('D', 'N') for k in kinds) or phi_kind not in ('D', 'N'):
        raise ValueError('six D/N field kinds and one phi D/N kind required')
    def rhs(state, t, data):
        mms = Manufactured(data.owner_initial, data.phi_initial, data.boundary_initial, data.boundary_constant)
        ref = ContinuumReference(*data.reference_arrays, {})
        x = state.array(); inner, outer, pb = mms.boundaries(t)
        if mode == 'diffusion':
            selected, coefficient_valid, finite_action = _diffusion(data.plan, x, inner, kinds,
                jnp.take(x, data.plan.donor, axis=-1), COEFFICIENTS)
            valid = coefficient_valid & finite_action
        else:
            result = apply_q_plan(data.plan, x, inner, outer, mms.phi(t), pb,
                COEFFICIENTS, kinds=kinds, phi_kind=phi_kind, tau=1., mu=1836.,
                characteristic_method='polynomial')
            selected = result.combined
            valid = jnp.all(result.inputs_valid & result.eigensystem_admissible)
        forcing = ref.source(mms, t, mode)
        total = selected.T+forcing
        finite = jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(total)) & jnp.all(jnp.isfinite(forcing))
        integral = total@data.volume
        finite = finite & jnp.all(jnp.isfinite(integral))
        diag = StageDiagnostics(t, jnp.all(valid), jnp.all(x[:3] > 0), finite, integral)
        return SixState.from_array(total), data, diag
    return Rk4Stepper(rhs)


def error_report(state, target, volume, regions):
    x, exact, vol = map(np.asarray, (state, target, volume))
    if x.shape != exact.shape or x.shape != (6, len(vol)):
        raise ValueError('solution error shape mismatch')
    result = {}
    for name, mask in {'global': np.ones(len(vol), bool), **regions}.items():
        mask = np.asarray(mask)
        if mask.dtype.kind != 'b' or mask.shape != vol.shape:
            raise ValueError('regional mask shape/type mismatch')
        if not mask.any():
            result[name] = {'count': 0}; continue
        e, v, r = x[:, mask]-exact[:, mask], vol[mask], exact[:, mask]
        rms = np.sqrt(np.sum(v*e*e, axis=1)/v.sum())
        norm = np.sqrt(np.sum(v*r*r, axis=1)/v.sum())
        result[name] = dict(count=int(mask.sum()), rms=rms.tolist(),
            relative=[float(a/b) if b > 0 else None for a, b in zip(rms, norm)],
            maximum=np.max(abs(e), axis=1).tolist())
    return result


def step_count(start, end, dt):
    """Avoid a spurious final step when an integral ratio rounds upward."""
    ratio = (end-start)/dt
    nearest = round(ratio)
    return max(1, nearest if abs(ratio-nearest) <= 8*np.finfo(float).eps*max(1., abs(ratio))
               else int(np.ceil(ratio)))


def advance(stepper, initial, *, start, end, dt, volume, target, regions=None,
            resume=None, checkpoint=None, max_steps=None, carry=None,
            compiled_step=None, timing_callback=None):
    """Host finite-run driver; no clipping and no stability/pass receipt.

    Check every stage and accepted endpoint before committing a checkpoint.
    A failed stage raises; no successful-run report is returned.
    """
    if not np.isfinite([start, end, dt]).all() or not 0 <= start < end <= 1 or dt <= 0:
        raise ValueError('invalid time interval/timestep')
    vol = np.asarray(volume)
    if vol.ndim != 1 or not np.isfinite(vol).all() or np.any(vol <= 0):
        raise ValueError('invalid physical owner volumes')
    initial = np.asarray(initial)
    if initial.shape != (6, len(vol)) or not np.isfinite(initial).all() or np.any(initial[:3] <= 0):
        raise ValueError('invalid initial state')
    count = step_count(start, end, dt)
    grid_time = lambda i: end if i == count else start+i*dt
    state = SixState.from_array(jnp.asarray(initial)); accepted = 0
    integral_rhs = np.zeros(6); times = []
    if resume is not None:
        accepted = resume['accepted_steps']
        state = SixState.from_array(jnp.asarray(resume['state']))
        integral_rhs = np.asarray(resume['integral_rhs']); times = list(resume['stage_times'])
        if not 0 <= accepted <= count or len(times) != 4*accepted:
            raise ValueError('invalid checkpoint step/stage count')
        expected = grid_time(accepted)
        if resume['time'] != expected:
            raise ValueError('checkpoint time/grid mismatch')
        if state.array().shape != initial.shape or not np.isfinite(state.array()).all() or np.any(np.asarray(state.array())[:3] <= 0):
            raise ValueError('invalid checkpoint state')
    call = compiled_step or jax.jit(lambda s, t, h, data: stepper(s, time=t, timestep=h, carry=data))
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 0):
        raise ValueError('max_steps requires a nonnegative integer')
    if resume is not None:
        expected_times = [grid_time(i)+s*(grid_time(i+1)-grid_time(i))
                          for i in range(accepted) for s in (0., .5, .5, 1.)]
        if not np.allclose(times, expected_times, atol=1e-14, rtol=1e-14) or integral_rhs.shape != (6,) or not np.isfinite(integral_rhs).all():
            raise ValueError('checkpoint stage history/integral mismatch')
    budget = count if max_steps is None else min(count, accepted+max_steps)
    for index in range(accepted, budget):
        t = grid_time(index); h = grid_time(index+1)-t
        tick = time.perf_counter()
        result = call(state, t, h, carry)
        for stage in result.stage_aux:
            if not all(bool(np.asarray(a)) for a in (stage.valid, stage.positive, stage.finite)):
                raise ValueError(f'invalid/admissible/positive RK stage at t={float(stage.time)}')
        x = np.asarray(result.state.array())
        if not np.isfinite(x).all() or np.any(x[:3] <= 0):
            raise ValueError(f'invalid accepted endpoint at t={t+h}')
        increments = np.stack([np.asarray(s.rhs_integral) for s in result.stage_aux])
        if timing_callback:
            timing_callback(index, time.perf_counter()-tick)
        integral_rhs += h/6*(increments[0]+2*increments[1]+2*increments[2]+increments[3])
        times.extend(float(s.time) for s in result.stage_aux)
        state = result.state; accepted = index+1
        if checkpoint:
            checkpoint(dict(state=x, accepted_steps=accepted, time=grid_time(accepted),
                integral_rhs=integral_rhs.copy(), stage_times=times.copy()))
    actual_end = grid_time(accepted)
    x = np.asarray(state.array()); change = (x-initial)@vol
    report = dict(scope='finite prescribed-phi research evolution; no stability qualification',
        completed=accepted == count, start=start, requested_end=end, actual_end=actual_end,
        dt=dt, accepted_steps=accepted, stage_times=times, finite=True, positive=True,
        volume_integral_change=change.tolist(), integrated_rhs=integral_rhs.tolist(),
        rk_balance_residual=(change-integral_rhs).tolist(),
        mms_volume_integral_drift=((x-np.asarray(target(actual_end)))@vol).tolist(),
        error=error_report(x, target(actual_end), vol, regions or {}))
    return x, report


def temporal_comparison(states, volume):
    """Self differences are separate from fixed-spatial-grid MMS error."""
    v = np.asarray(volume)
    d = [np.sqrt(np.sum(v*(np.asarray(a)-np.asarray(b))**2, axis=1)/v.sum())
         for a, b in zip(states[:-1], states[1:])]
    return dict(self_difference_rms=[a.tolist() for a in d],
        observed_orders=[float(np.log2(a/b)) if a > 0 and b > 0 else None for a, b in zip(*d)])
