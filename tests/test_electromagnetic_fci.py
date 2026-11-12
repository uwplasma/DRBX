"""Electromagnetic four-field FCI model: A_par, magnetic flutter and electron inertia against analytic results.

See ``drbx.validation.electromagnetic_waves`` for the slab harness. Shear-Alfven: ``omega = kz vA / sqrt(1 + k_perp^2 d_e^2)``,
``vA^2 = 1/(beta n)``, ``d_e^2 = 1/(beta mi/me n)``; drift-Alfven: the ``b1.grad n0`` flutter term (``omega_* A_par``) lowers the
Alfven branch.
"""

from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.geometry import build_open_slab_geometry
from drbx.native.electromagnetic import em_energies
from drbx.native.fci_4_field_rhs import Fci4FieldState
from drbx.validation.electromagnetic_waves import (
    dispersion_matrix,
    laplacian,
    measured_frequency,
    rhs,
    rk4,
    slab,
)


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Each test compiles its own RK4 programs; free them so the long fast suite stays inside a small runner's memory."""

    yield
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize("beta, expected_inertia", [(0.5, 0.02), (0.005, 2.0)])
def test_shear_alfven_wave_with_electron_inertia(beta, expected_inertia):
    """omega = kz vA / sqrt(1 + k_perp^2 d_e^2) in the ideal (k_perp d_e = 0.14) and the inertial (1.4) regime."""
    kx, kz, mu = 1.0, 0.25, 100.0
    inertia = kx**2 / (beta * mu)
    assert inertia == pytest.approx(expected_inertia)
    analytic = kz / np.sqrt(beta) / np.sqrt(1 + inertia)
    omega, measured = measured_frequency(kx, 0.0, kz, beta, mu, 0.0, None, (12, 3, 16))
    assert omega == pytest.approx(analytic, rel=1e-12)
    assert measured == pytest.approx(analytic, rel=0.04)
    if inertia > 1:  # the inertial correction (a factor 1/sqrt(3) here) is resolved, not a small shift
        assert abs(measured - kz / np.sqrt(beta)) > 0.3 * analytic


def test_drift_alfven_wave_with_flutter_of_the_background_gradient():
    """beta = 0.03, Ln = 10: the Alfven branch is 0.729 (electrostatic limit 1.207); the flutter b1.grad n0 matters."""
    args = (1.0, 1.0, 0.1, 0.03, 100.0, 1.0, 10.0)
    omega, measured = measured_frequency(*args, (12, 16, 16))
    electrostatic = (1j * np.linalg.eigvals(dispersion_matrix(*args[:3], 1e-8, *args[4:]))).real.max()
    assert omega == pytest.approx(0.7292, abs=2e-4)
    assert measured == pytest.approx(omega, rel=0.03)
    assert abs(measured - electrostatic) > 0.3 * electrostatic


def smooth_state(shape, lengths, *, amp, n_amp=0.0):
    g = build_open_slab_geometry(shape, radial_extent=lengths[0], binormal_extent=lengths[1], parallel_length=lengths[2])
    x, y, z = (np.asarray(a.centers)[idx] for a, idx in zip((g.grid.x, g.grid.y, g.grid.z), (np.s_[:, None, None], np.s_[None, :, None], np.s_[None, None, :])))
    sx = np.sin(np.pi * x / lengths[0])
    ky, kz = 2 * np.pi / lengths[1], 2 * np.pi / lengths[2]
    phi = amp * sx * np.cos(ky * y + kz * z)
    return Fci4FieldState(
        density=jnp.asarray(1 + n_amp * sx * np.cos(ky * y - kz * z) + 0 * z),
        omega=jnp.asarray(-(np.pi / lengths[0]) ** 2 * phi - ky**2 * phi),  # lap phi up to the discretization
        v_ion_parallel=jnp.asarray(amp * sx * np.sin(ky * y + 2 * kz * z)),
        v_electron_parallel=jnp.asarray(0.5 * amp * sx * np.cos(2 * kz * z - ky * y)))


def test_vanishing_beta_recovers_the_electrostatic_right_hand_side():
    """A_par and the flutter are O(beta): the difference to the electrostatic RHS scales linearly with beta."""
    shape, lengths = (8, 6, 8), (np.pi, 2 * np.pi, 8.0)
    state = smooth_state(shape, lengths, amp=1e-2, n_amp=1e-2)
    _, es_kw, *_ = slab(shape, lengths, beta=1.0, mu=100.0, te=1.0, em=False)
    assert es_kw["electromagnetic"] is None
    reference, _ = rhs(state, es_kw)
    errors = []
    for beta in (1e-6, 1e-8):
        _, kw, *_ = slab(shape, lengths, beta=beta, mu=100.0, te=1.0)
        result, _ = rhs(state, kw)
        errors.append(max(float(jnp.max(jnp.abs(getattr(result, f) - getattr(reference, f)))) for f in ("density", "omega", "v_ion_parallel", "v_electron_parallel")))
    scale = max(float(jnp.max(jnp.abs(getattr(reference, f)))) for f in ("density", "omega", "v_ion_parallel", "v_electron_parallel"))
    assert errors[1] < 1e-6 * scale
    assert errors[0] / errors[1] == pytest.approx(100.0, rel=0.2)


def test_energy_is_conserved_over_a_wave_period_with_the_magnetic_energy_included():
    """Cold Alfven wave: ExB + magnetic + electron kinetic energy is constant while ExB and magnetic each swing."""
    shape, kx, ky, kz, beta, mu = (12, 8, 16), 1.0, 1.0, 0.25, 0.05, 100.0
    g, kw, phi_solver, bcs = slab(shape, (np.pi / kx, 2 * np.pi / ky, 2 * np.pi / kz), beta=beta, mu=mu)
    closure, jac = kw["electromagnetic"], g.cell_metric.J
    x, y, z = (np.asarray(a.centers) for a in (g.grid.x, g.grid.y, g.grid.z))
    wave = np.sin(kx * x)[:, None, None] * np.cos(ky * y[None, :, None] + kz * z[None, None, :])
    period = 2 * np.pi * np.sqrt(1 + (kx**2 + ky**2) / (beta * mu)) * np.sqrt(beta) / kz
    steps = 40
    step = jax.jit(lambda s: rk4(s, kw, period / steps))

    def components(state):
        phi = phi_solver(-state.omega, face_bc=bcs.phi_face_bc)
        apar = closure.solve_apar(state.density, state.v_ion_parallel, state.v_electron_parallel, face_bc=bcs.v_electron_parallel_face_bc)
        return np.array([float(e) for e in em_energies(closure, state, phi, apar, jac)[:3]])

    for amp, limit in ((1e-3, 0.01), (0.05, 0.05)):  # linear; then amplitudes where flutter and 1/n are active
        zero = jnp.zeros(shape)
        state = Fci4FieldState(density=jnp.ones(shape), omega=laplacian(amp * jnp.asarray(wave), g, kw, bcs),
                               v_ion_parallel=zero, v_electron_parallel=zero)
        history = [components(state)]
        for _ in range(steps):
            state = step(state)
            history.append(components(state))
        history = np.array(history)
        total = history.sum(1)
        assert (total.max() - total.min()) / total[0] < limit
        assert np.ptp(history[:, 0]) > 0.9 * total[0] and np.ptp(history[:, 1]) > 0.5 * total[0]  # real exchange


def test_ampere_and_flutter_converge_to_manufactured_solution():
    """MMS: A(x, z), phi(x, z), n(x, y), v_i given and w from Ampere's law; A_par and dw/dt converge at O(h^2)."""
    beta, mu, te, errors = 0.3, 50.0, 1.0, []
    for n in (8, 16):
        shape, lengths = (n, n, n), (np.pi, 2 * np.pi, 2 * np.pi)
        g, kw, _, bcs = slab(shape, lengths, beta=beta, mu=mu, te=te, iterations=12)
        x, y, z = (jnp.asarray(a.centers)[i] for a, i in zip((g.grid.x, g.grid.y, g.grid.z), (np.s_[:, None, None], np.s_[None, :, None], np.s_[None, None, :])))
        a_f = lambda x, y, z: 0.1 * jnp.sin(x) * jnp.cos(z)  # noqa: E731
        phi_f = lambda x, y, z: 0.1 * jnp.sin(x) * jnp.cos(z + 1.0)  # noqa: E731
        n_f = lambda x, y, z: 1 + 0.2 * jnp.sin(x) * jnp.cos(y)  # noqa: E731
        vi_f = lambda x, y, z: 0.1 * jnp.sin(2 * x) * jnp.cos(y)  # noqa: E731
        w_f = lambda x, y, z: vi_f(x, y, z) - (jax.grad(jax.grad(a_f, 0), 0)(x, y, z) * -1 + beta * mu * n_f(x, y, z) * a_f(x, y, z)) / (beta * n_f(x, y, z))  # noqa: E731
        ve_f = lambda x, y, z: w_f(x, y, z) + mu * a_f(x, y, z)  # noqa: E731
        field = lambda f: jax.vmap(jax.vmap(jax.vmap(f, (None, None, 0)), (None, 0, None)), (0, None, None))(x[:, 0, 0], y[0, :, 0], z[0, 0, :])  # noqa: E731
        d = lambda f, i: (lambda x, y, z: jax.grad(f, i)(x, y, z))  # noqa: E731
        # exact dw/dt = -{phi, ve} + mu (grad_par phi - Te grad_par n / n), grad_par f = f_z + {A, f},  {a, b} = a_x b_y - a_y b_x
        bracket = lambda f, g_: (lambda x, y, z: d(f, 0)(x, y, z) * d(g_, 1)(x, y, z) - d(f, 1)(x, y, z) * d(g_, 0)(x, y, z))  # noqa: E731
        gpar = lambda f: (lambda x, y, z: d(f, 2)(x, y, z) + bracket(a_f, f)(x, y, z))  # noqa: E731
        dw = lambda x, y, z: -bracket(phi_f, ve_f)(x, y, z) + mu * (gpar(phi_f)(x, y, z) - te * gpar(n_f)(x, y, z) / n_f(x, y, z))  # noqa: E731
        state = Fci4FieldState(density=field(n_f), omega=field(lambda x, y, z: jax.grad(jax.grad(phi_f, 0), 0)(x, y, z)),
                               v_ion_parallel=field(vi_f), v_electron_parallel=field(w_f))
        a_num = kw["electromagnetic"].solve_apar(state.density, state.v_ion_parallel, state.v_electron_parallel, face_bc=bcs.v_electron_parallel_face_bc)
        dw_num = rhs(state, kw)[0].v_electron_parallel
        errors.append((float(jnp.max(jnp.abs(a_num - field(a_f)))), float(jnp.max(jnp.abs(dw_num - field(dw))))))
    errors = np.array(errors)
    assert np.all(errors[0] / errors[1] > 3.0), errors


def test_forward_mode_derivative_through_the_electromagnetic_step_matches_finite_differences():
    """Forward mode (reverse mode through the GMRES potential solve is not available in the electrostatic model either)."""
    shape, lengths = (8, 6, 8), (np.pi, 2 * np.pi, 8.0)
    _, kw, *_ = slab(shape, lengths, beta=0.05, mu=100.0, te=1.0)
    base = smooth_state(shape, lengths, amp=1e-2, n_amp=1e-2)

    def objective(scale):
        state = Fci4FieldState(*[getattr(base, f) * scale if f != "density" else 1 + (base.density - 1) * scale
                                 for f in ("density", "omega", "v_ion_parallel", "v_electron_parallel")])
        out = rk4(state, kw, 0.05)
        return jnp.sum(out.v_electron_parallel**2) + jnp.sum(out.omega**2)

    gradient, h = float(jax.jvp(objective, (1.0,), (1.0,))[1]), 1e-4
    finite = float(objective(1.0 + h) - objective(1.0 - h)) / (2 * h)
    assert gradient == pytest.approx(finite, rel=1e-5)


def test_deck_physics_table_switches_the_electromagnetic_model_on(tmp_path):
    from drbx.deck import run

    deck = {"model": {"backend": "stellarator_turbulence"}, "shifted_torus": {"shape": [6, 8, 4]},
            "run": {"steps": 2, "dt": 1e-3, "amplitude": 0.05}}
    with pytest.raises(ValueError, match="physics"):
        run({**deck, "physics": {"electromagnet": True}})
    off = run(deck, output_dir=tmp_path, case_name="es")
    on = run({**deck, "physics": {"electromagnetic": True, "beta": 1e-2}}, output_dir=tmp_path, case_name="em")
    with np.load(off["paths"][1]) as a, np.load(on["paths"][1]) as b:
        assert "apar_frames" not in a and b["apar_frames"].shape == b["density_frames"].shape
        assert 0 < np.abs(b["apar_frames"][-1]).max() < 1e-2 and np.isfinite(b["energies"]).all()
        assert np.allclose(a["density_frames"], b["density_frames"], atol=2e-3)  # small beta: close to electrostatic
        assert not np.array_equal(a["omega_frames"], b["omega_frames"])
    assert on["results"]["final_energy_components"]["magnetic"] > 0
