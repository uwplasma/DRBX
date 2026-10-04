"""Portable algebra/stage and bounded-HSX replay; no global qualification."""
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scripts.q08_extraction_global import common as c
from scripts.q09_evolved_mms.mms import Manufactured, ContinuumReference, smooth_fields, amplitude, BASE
from scripts.q09_evolved_mms.evolution import (SixState, stage_stepper, advance, q_stepper, QPayload, temporal_comparison)
from scripts.q09_evolved_mms.provider import save_checkpoint, load_checkpoint, source_hash, PreparedProvider, FROZEN_IDENTITY, validate_frozen_policy
from drbx.stencils.q_parallel import load_chunk
from drbx.stencils.q_bank import build_q_bank
from drbx.stencils.q_plan import lower_q_plan
from drbx.native.q_plan import apply_q_plan
from drbx.native.q_parallel import QBoundaryData

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bounded():
    root = ROOT/'scripts/q08_extraction_global/inputs/bounded'
    bank = build_q_bank(*(load_chunk(root/f'N32_h{d}.npz') for d in (16, 32)))
    with np.load(root/'N32_inputs.npz') as z:
        data = {k: z[k].copy() for k in z.files}
    geo = {k: data[k] for k in ('magnetic_L', 'b_eta', 'eta_step', 'bmag')}
    return bank, data, geo


def analytic_fixture(no=2):
    x = np.broadcast_to(BASE[:, None], (6, no)).copy(); x[:, 0] += .01
    m = Manufactured(jnp.asarray(x), jnp.arange(no)*.01, (), ())
    r = ContinuumReference(jnp.asarray(x.T), jnp.ones((no, 6))*.02,
        jnp.ones(no)*.03, jnp.ones((no, 6))*.001, jnp.ones(no)*.04,
        jnp.ones(no)*1.2, jnp.arange(no)[:, None], jnp.ones((no, 1)), {})
    return m, r


def test_analytic_derivative_jit_jvp_and_catalogue():
    points = np.array([[.2, .3, .4], [.8, 1.2, 2.4]])
    v, g, p, pg = smooth_fields(points, 0.)
    vv, gg = c.six_fields(points); pp, ppg = c.phi_fields(points)
    np.testing.assert_allclose(v, vv[:, 1], atol=2e-15)
    np.testing.assert_allclose(g, gg[:, 1], atol=2e-15)
    np.testing.assert_allclose(p, pp[:, 1], atol=2e-15)
    np.testing.assert_allclose(pg, ppg[:, 1], atol=2e-15)
    m, r = analytic_fixture(); t = .37
    _, derivative = jax.jvp(jax.jit(m.state), (t,), (1.,))
    np.testing.assert_allclose(derivative, m.derivative(t), atol=2e-15)
    for mode in ('diffusion', 'complete'):
        source = jax.jit(lambda t: r.source(m, t, mode))(t)
        np.testing.assert_allclose(source+r.rhs(t, mode), m.derivative(t), atol=1e-13)
    assert not np.allclose(r.rhs(t, 'complete'), r.rhs(t, 'diffusion'))
    # Nonlinear continuum cannot be scaled as a whole from t0.
    assert not np.allclose(r.rhs(t, 'complete'), amplitude(t)*r.rhs(0., 'complete'))
    assert np.min(np.asarray(smooth_fields(points, 1.)[0])[:, :3]) > 0


def test_stage_times_evolving_state_and_known_ode_order():
    initial = np.ones((6, 2)); volume = np.array([.3, .7])
    stepper = stage_stepper(lambda x, t: (x, True), lambda t: jnp.zeros((6, 2)), volume)
    single = stepper(SixState.from_array(jnp.asarray(initial)), time=.2, timestep=.1, carry=None)
    np.testing.assert_allclose([d.time for d in single.stage_aux], [.2, .25, .25, .3])
    # k2 and k3 reflect evolving stage state rather than an analytic replacement.
    np.testing.assert_allclose([d.rhs_integral[0] for d in single.stage_aux], [1., 1.05, 1.0525, 1.10525])
    errors = []; states = []
    for dt in (.2, .1, .05):
        x, report = advance(stepper, initial, start=0., end=1., dt=dt, volume=volume,
            target=lambda t: np.ones((6, 2))*np.exp(t))
        states.append(x); errors.append(abs(x[0, 0]-np.e))
        assert report['accepted_steps'] == round(1/dt)
        assert report['completed'] and report['actual_end'] == 1.
        assert len(report['stage_times']) == 4*report['accepted_steps']
        np.testing.assert_allclose(report['rk_balance_residual'], 0., atol=2e-15)
    assert min(np.log2(np.array(errors[:-1])/errors[1:])) > 3.8
    assert min(temporal_comparison(states, volume)['observed_orders']) > 3.8


@pytest.mark.parametrize('bad', ['valid', 'finite', 'positive'])
def test_failure_no_accepted_checkpoint(bad):
    saved = []
    def numerical(x, t):
        value = jnp.ones_like(x)
        if bad == 'finite': value = value*jnp.nan
        if bad == 'positive': value = value*-100
        return value, bad != 'valid'
    stepper = stage_stepper(numerical, lambda t: jnp.zeros((6, 1)), np.ones(1))
    with pytest.raises(ValueError, match='RK stage'):
        advance(stepper, np.ones((6, 1)), start=0., end=.1, dt=.1, volume=np.ones(1),
            target=lambda t: np.ones((6, 1)), checkpoint=saved.append)
    assert saved == []


def test_checkpoint_restart_and_identity(tmp_path):
    initial = np.ones((6, 1)); stepper = stage_stepper(lambda x, t: (-x, True), lambda t: jnp.zeros((6, 1)))
    path = tmp_path/'state.npz'; identity = source_hash()
    kwargs = dict(start=0., end=1., dt=.1, volume=np.ones(1), target=lambda t: initial*np.exp(-t))
    advance(stepper, initial, **kwargs, max_steps=3, checkpoint=lambda p: save_checkpoint(path, identity, p))
    resume = load_checkpoint(path, identity)
    resumed, report = advance(stepper, initial, **kwargs, resume=resume)
    fresh, fresh_report = advance(stepper, initial, **kwargs)
    np.testing.assert_array_equal(resumed, fresh)
    assert report == fresh_report
    with pytest.raises(ValueError, match='stale checkpoint'):
        load_checkpoint(path, 'different source/input')


def test_live_bc_normal_tangent_and_actual_hsx_t0_replay(bounded):
    bank, data, geo = bounded
    initial_bc = c.boundaries(bank, 1); zero_bc = c.boundaries(bank, 0)
    # State donor values are accepted Q08 weighted observations, not centers.
    state = np.broadcast_to(BASE[:, None], (6, bank.metadata['n_owner'])).copy()
    state[:, data['donor_ids']] = data['state'][1]
    phi = np.zeros(bank.metadata['n_owner']); phi[data['donor_ids']] = data['phi'][1]
    m = Manufactured(jnp.asarray(state), jnp.asarray(phi), initial_bc, zero_bc)
    for t in (0., .4):
        ib, ob, pb = m.boundaries(t)
        for w, row in enumerate(bank.wall_index):
            points = bank.query_table[bank.wall_node_query[w]]
            v, g, pv, pg = smooth_fields(points, t)
            np.testing.assert_allclose(ib.dirichlet_trace[:, row], v.T, atol=1e-14)
            np.testing.assert_allclose(ib.neumann_normal[:, row], np.einsum('qa,qfa->fq', bank.boundary_wall_normal[w], g), atol=1e-14)
            np.testing.assert_allclose(pb.neumann_normal[row], np.einsum('qa,qa->q', bank.boundary_wall_normal[w], pg), atol=1e-14)
            slots = bank.query_table[bank.wall_slot_query[w, [1, 3, 2]]]
            sv, sg, _, _ = smooth_fields(slots, t)
            np.testing.assert_allclose(ib.dirichlet_query_value[:, row], sv.T, atol=1e-14)
            np.testing.assert_allclose(ib.dirichlet_tangent[:, row], np.asarray(sg)[:, :, 1:].transpose(1, 0, 2), atol=1e-14)
    plan = lower_q_plan(bank, diffusion_span=1/32, **geo)
    for kinds, pk in c.KINDS:
        direct = apply_q_plan(plan, state, *initial_bc[:2], phi, initial_bc[2], c.COEFF,
            kinds=kinds, phi_kind=pk, tau=1., mu=1836., characteristic_method='polynomial')
        live = apply_q_plan(plan, m.state(0.), *m.boundaries(0.)[:2], m.phi(0.), m.boundaries(0.)[2], c.COEFF,
            kinds=kinds, phi_kind=pk, tau=1., mu=1836., characteristic_method='polynomial')
        c.check_outputs(live, direct)
    # The actual fixture cannot be admitted to scientific evolution.
    assert validate_frozen_policy(bank, geo)
    provider = PreparedProvider(bank, geo, m, None, np.ones(len(bank.owners)), {}, {'fixture': '0'*64})
    with pytest.raises(ValueError, match='full-domain'):
        provider.validate()


def test_q_payload_dynamic_leaves_and_diffusion_isolation(bounded):
    bank, data, geo = bounded
    no = len(bank.owners); nr = len(bank.raw)
    m, _ = analytic_fixture(no)
    m = Manufactured(m.owner_initial, m.phi_initial, c.boundaries(bank, 1), c.boundaries(bank, 0))
    r = ContinuumReference(jnp.tile(jnp.asarray(BASE), (nr, 1)), jnp.zeros((nr, 6)),
        jnp.zeros(nr), jnp.zeros((len(bank.owners), 6)), jnp.zeros(nr), jnp.ones(nr),
        jnp.asarray(bank.owner_raw), jnp.asarray(bank.owner_weight), {})
    from dataclasses import replace
    plan = replace(lower_q_plan(bank, diffusion_span=1/32, **geo), n_owner=no, donor=bank.donor % no)
    # Deliberately remapped algebra fixture, never a scientific physical patch.
    data = QPayload(plan, m.owner_initial, m.phi_initial,
        m.boundary_initial, m.boundary_constant, tuple(getattr(r, k) for k in
        ('values', 'gradients', 'phi_gradient', 'diffusion', 'kappa', 'bmag', 'owner_raw', 'owner_weight')), jnp.ones(no))
    # Wiring algebra only, unrelated to real HSX evolution.
    stepper = q_stepper('diffusion', ('D',)*6, 'D')
    trace = jax.make_jaxpr(stepper.rhs_fn)(SixState.from_array(m.state(0.)), 0., data)
    assert all(np.size(a) <= 6 for a in trace.consts)  # Only frozen scalar coefficients/base.
    text = str(trace)
    assert 'eig' not in text
    assert len(trace.jaxpr.invars) > 30  # Prepared bank/reference leaves are runtime inputs.
    complete = q_stepper('complete', ('D',)*6, 'D')
    trace_complete = jax.make_jaxpr(complete.rhs_fn)(SixState.from_array(m.state(0.)), 0., data)
    assert all(np.size(a) <= 6 for a in trace_complete.consts)
    assert len(trace_complete.jaxpr.invars) > 30
    fn = jax.jit(lambda x, t, payload: complete.rhs_fn(SixState.from_array(x), t, payload)[0].array())
    value, derivative = jax.jvp(lambda x: fn(x, .2, data), (m.state(.2),), (jnp.ones_like(m.state(.2))*.001,))
    assert np.isfinite(value).all() and np.isfinite(derivative).all()



def test_reference_prepare_t0_matches_independent_q08_algebra(bounded):
    bank, data, geo = bounded
    # Fixed analytic geometry is a reference algebra control only. Actual HSX
    # numerical replay is tested separately; no scientific HSX reference claim.
    def geom(p):
        return np.ones(len(p)), np.broadcast_to([.1, .2, 1.], (len(p), 3)), np.ones(len(p))*2
    fake_geo = {**geo, 'b_eta': np.ones(len(bank.raw)), 'bmag': np.ones(len(bank.raw))*2}
    ref = ContinuumReference.prepare(bank, fake_geo, geom)
    from scripts.q08_rhs_mms_global.science import continuum
    expected, _ = continuum(bank, fake_geo, geom)
    np.testing.assert_allclose(ref.rhs(0., 'complete').T, expected[1, :, 18:24], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(ref.rhs(.3, 'diffusion').T, amplitude(.3)*expected[1, :, 12:18], atol=1e-12)


def test_checkpoint_tamper_and_missing_inputs_cli(tmp_path):
    path = tmp_path/'cp.npz'
    payload = dict(state=np.ones((6, 1)), accepted_steps=1, time=.1, integral_rhs=np.ones(6), stage_times=[0., .05, .05, .1])
    save_checkpoint(path, 'signature', payload)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays['state'] = arrays['state']*1.001
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='content hash'):
        load_checkpoint(path, 'signature')
    from scripts.q09_evolved_mms.cli import main
    with pytest.raises(ValueError, match='prepared-input configuration'):
        main(['--inputs', str(tmp_path), '--output', str(tmp_path/'output'), '--mode', 'diffusion', '--dt', '.001', '--end', '.01'])
    assert not (tmp_path/'output/report.json').exists()


def test_member_observation_identity_and_weighted_derivative():
    from types import SimpleNamespace
    points = np.array([[.1+i*.1, .2+i*.13, .4+i*.15] for i in range(8)])
    ids = np.array([0]*4+[1]*4); weights = np.array([1., 2., 3., 4.]*2)
    volume = np.array([10., 10.])
    # Pure observation algebra fixture; this is not an admitted HSX bank.
    bank = SimpleNamespace(metadata={'n': 2, 'n_owner': 2}, raw=np.arange(8), owners=np.arange(2),
        raw_to_owner=ids, raw_weight=weights/volume[ids], diagnostics={'slot_points': np.repeat(points[:, None], 5, axis=1)},
        wall_index=np.array([], dtype=int), validate=lambda: None)
    m = Manufactured.from_members(bank, points, ids, weights, volume)
    values = c.six_fields(points)[0][:, 1]
    expected = np.stack([np.sum(values[ids == i]*weights[ids == i, None], axis=0)/volume[i] for i in range(2)]).T
    np.testing.assert_allclose(m.state(0.), expected, rtol=2e-15, atol=2e-15)
    assert not np.allclose(expected[:, 0], c.six_fields(points[:1])[0][0, 1])
    np.testing.assert_allclose(jax.jvp(m.state, (.4,), (1.,))[1], m.derivative(.4), atol=2e-15)
    with pytest.raises(ValueError, match='coordinate/owner'):
        Manufactured.from_members(bank, points+.001, ids, weights, volume)
    with pytest.raises(ValueError, match='weights'):
        Manufactured.from_members(bank, points, ids, weights[::-1], volume)


def test_non_autonomous_source_stage_refresh_and_order():
    exact = lambda t: jnp.ones((6, 1))*(1+.1*jnp.sin(t))
    source = lambda t: .1*jnp.cos(t)*jnp.ones((6, 1))+exact(t)
    stepper = stage_stepper(lambda x, t: (-x, True), source)
    result = stepper(SixState.from_array(exact(0.)), time=0., timestep=.2, carry=None)
    k1 = -.0+ .1
    k2 = -(1+.1*k1)+float(source(.1)[0, 0])
    k3 = -(1+.1*k2)+float(source(.1)[0, 0])
    k4 = -(1+.2*k3)+float(source(.2)[0, 0])
    np.testing.assert_allclose([s.rhs_integral[0] for s in result.stage_aux], [k1, k2, k3, k4], atol=2e-15)
    errors = []
    for dt in (.2, .1, .05):
        x, _ = advance(stepper, exact(0.), start=0., end=1., dt=dt, volume=np.ones(1), target=exact)
        errors.append(abs(x[0, 0]-float(exact(1.)[0, 0])))
    assert min(np.log2(np.asarray(errors[:-1])/errors[1:])) > 3.8


def test_transitive_numerical_source_identity(monkeypatch):
    before = source_hash()
    read = Path.read_bytes
    def changed(path):
        data = read(path)
        return data+b'\n# identity regression control\n' if path.name == 'q_characteristic_polynomial.py' else data
    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert source_hash() != before
