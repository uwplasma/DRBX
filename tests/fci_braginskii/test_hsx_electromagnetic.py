"""Electromagnetic closure of the HSX backend: shear-Alfven dispersion, beta -> 0, energy ledger, flutter pair.

A periodic (x, z) slab with uniform ``n = B = 1`` and the backend's production pieces: the support pair of
``build_support_pair`` (mapped central divergence), ``LocalFciDrbEBRhs._conservative_parallel_rates`` for the
material rows and ``flutter_pair`` for the magnetic flutter.  Ampere's law and the polarization solve are
exact Fourier solves of the same equations the model solves with its Helmholtz and Poisson solvers.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs
from drbx.fci_braginskii.native.fci_electromagnetic import FciEm, flutter_pair
from drbx.fci_braginskii.native.fci_support_pair import build_support_pair

jax.config.update("jax_enable_x64", True)

NX, NZ, LX, LZ = 8, 16, 2 * np.pi, 2 * np.pi
MU, RHO = 1836.0, 0.1
DX, DZ = LX / NX, LZ / NZ
KX = 2 * np.pi * np.fft.fftfreq(NX, DX)[:, None]
KZ = 2 * np.pi * np.fft.fftfreq(NZ, DZ)[None, :]
MASS = jnp.full((NX, NZ), DX * DZ)
ACTIVE = jnp.ones((NX, NZ), dtype=bool)
HOST = SimpleNamespace(parameters=SimpleNamespace(tau=1.0, mi_over_me=MU, parallel_dissipation=0.0))


def _central_pair():
    """The production recipe: weighted adjoint of a mapped central divergence, made constant exact."""

    divergence0 = lambda q: (jnp.roll(q, -1, axis=1) - jnp.roll(q, 1, axis=1)) / (2 * DZ)  # noqa: E731
    return build_support_pair(divergence0, MASS, active=ACTIVE, core=ACTIVE)


def _bracket(a, f):
    """Centered periodic Poisson bracket ``{a, f} = a_x f_z - a_z f_x`` (the model's centered bracket in a slab)."""

    ddx = lambda q: (jnp.roll(q, -1, axis=0) - jnp.roll(q, 1, axis=0)) / (2 * DX)  # noqa: E731
    ddz = lambda q: (jnp.roll(q, -1, axis=1) - jnp.roll(q, 1, axis=1)) / (2 * DZ)  # noqa: E731
    return ddx(a) * ddz(f) - ddz(a) * ddx(f)


def _fourier(function, rhs):
    return jnp.fft.ifft2(function * jnp.fft.fft2(rhs)).real


def ampere(w, vi, beta, n=1.0):
    """``(-L_perp + beta mu n) A = beta n (Vi - w)``."""

    return _fourier(beta * n / (KX**2 + beta * MU * n), vi - w) if beta > 0 else jnp.zeros_like(w)


def _rates(w, omega, beta, apar=None, material_rows=True):
    """``(dw, domega)`` of the cold slab: material rows, ``mu G phi`` and ``D(j)``; flutter when ``apar`` is given."""

    gradient, divergence = _central_pair()
    if apar is not None:
        gradient, divergence = flutter_pair(
            gradient, divergence, lambda f: -RHO * (_bracket(apar, f) - 0.0), MASS, ACTIVE, ACTIVE
        )
    n, vi = jnp.ones_like(w), jnp.zeros_like(w)
    ve = w + MU * (apar if apar is not None else ampere(w, vi, beta))
    phi = _fourier(-1.0 / (RHO**2 * np.where(KX == 0, np.inf, KX**2)) * np.ones((1, NZ)), omega)
    primitive = {"density": n, "Te": 0 * n, "Ti": 0 * n, "Vi": vi, "Ve": ve}
    material, _ = LocalFciDrbEBRhs._conservative_parallel_rates(HOST, primitive, gradient, divergence)
    return (material[..., 4] if material_rows else 0.0) + MU * gradient(phi), divergence(n * (vi - ve)), phi, ve


def _frequency(beta, mode):
    """Largest-imaginary eigenfrequency of the linearized (w, omega) system for one Fourier mode."""

    kx, kz = mode
    x, z = (np.arange(NX) + 0.5) * DX, (np.arange(NZ) + 0.5) * DZ
    cx = np.cos(kx * x[:, None])
    modes = (cx * np.cos(kz * z[None, :]), cx * np.sin(kz * z[None, :]))
    flat = lambda a, b: jnp.concatenate((a.ravel(), b.ravel()))  # noqa: E731

    def step(y):
        w, omega = y[: NX * NZ].reshape(NX, NZ), y[NX * NZ :].reshape(NX, NZ)
        dw, domega, _, _ = _rates(w, omega, beta)
        return flat(dw, domega)

    jac = np.asarray(jax.jacfwd(step)(jnp.zeros(2 * NX * NZ)))
    zero = np.zeros(NX * NZ)
    pair = np.stack([np.concatenate((m.ravel(), zero)) for m in modes] + [np.concatenate((zero, m.ravel())) for m in modes], axis=1)
    reduced = np.linalg.lstsq(pair, jac @ pair, rcond=None)[0]
    values = np.linalg.eigvals(reduced)
    return np.max(np.abs(values.imag)), np.max(np.abs(values.real))


def _effective_kz(kz):
    """``k_par`` of the discrete pair: ``G -> i k_eff`` (central differences)."""

    return np.sin(kz * DZ) / DZ


@pytest.mark.parametrize("beta", [1.0e-3, 1.0e-2, 1.0e-1])
@pytest.mark.parametrize("mode", [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)])
def test_shear_alfven_dispersion_matches_the_analytic_frequency(beta, mode):
    kx, kz = mode
    omega, damping = _frequency(beta, mode)
    k_perp2 = kx**2  # the Fourier solves of Ampere's law and the polarization are spectral
    analytic = _effective_kz(kz) * np.sqrt(MU) / (RHO * np.sqrt(k_perp2 + beta * MU))
    assert damping < 1e-8 * omega  # no growth, no damping: the closure conserves the wave
    assert omega == pytest.approx(analytic, rel=1e-6)


def test_vanishing_beta_recovers_the_electrostatic_frequency():
    kx, kz = 1.0, 1.0
    electrostatic = _effective_kz(kz) * np.sqrt(MU) / (RHO * kx)
    em_small = _frequency(1.0e-9, (kx, kz))[0]
    assert em_small == pytest.approx(electrostatic, rel=1e-6)
    # the alfvenic branch is slower than the electrostatic one at finite beta and approaches it as beta -> 0
    assert _frequency(1.0e-2, (kx, kz))[0] < _frequency(1.0e-4, (kx, kz))[0] < electrostatic * (1 + 1e-9)
    assert FciEm().beta == 1.0e-3
    with pytest.raises(ValueError, match="beta > 0"):
        FciEm(0.0)


def _random(seed, scale=0.3):
    rng = np.random.default_rng(seed)
    smooth = lambda: jnp.asarray(jnp.fft.ifft2(jnp.fft.fft2(rng.standard_normal((NX, NZ))) * np.exp(-0.5 * (KX**2 + KZ**2) / 4)).real)  # noqa: E731
    return scale * smooth(), scale * smooth()


def _energy_budget(beta):
    """``(dE/dt, scale)`` of the electron kinetic + magnetic + ExB energy through the phi / current coupling, flutter on."""

    w, omega = _random(0, scale=0.5)
    vi = jnp.zeros_like(w)
    apar = ampere(w, vi, beta)
    dw, domega, phi, ve = _rates(w, omega, beta, apar=apar, material_rows=False)
    j = vi - ve
    # E = <Ve^2>/2mu + <A j>/2 - <phi omega>/2, Ve = w + mu A: dE/dt = <Ve (dw + mu dA)>/mu + <dA j> - <phi domega>
    dapar = ampere(dw, vi, beta)
    area = DX * DZ
    parts = np.array([jnp.sum(ve * (dw + MU * dapar)) / MU, jnp.sum(dapar * j), -jnp.sum(phi * domega)]) * area
    return parts.sum(), np.abs(parts).sum()


@pytest.mark.parametrize("beta", [1.0e-3, 1.0e-1])
def test_energy_ledger_closes_with_the_magnetic_energy_and_the_flutter_pair(beta):
    exchange, scale = _energy_budget(beta)
    assert abs(exchange) < 1e-12 * scale


def test_without_the_magnetic_term_the_ledger_does_not_close():
    beta = 1.0e-1
    w, omega = _random(0, scale=0.5)
    vi = jnp.zeros_like(w)
    apar = ampere(w, vi, beta)
    dw, domega, phi, ve = _rates(w, omega, beta, apar=apar, material_rows=False)
    area = DX * DZ
    dapar = ampere(dw, vi, beta)
    open_ledger = (jnp.sum(ve * (dw + MU * dapar)) / MU - jnp.sum(phi * domega)) * area  # no magnetic term
    assert abs(open_ledger) > 1e-3 * abs(jnp.sum(ve * dw) / MU * area)


def test_flutter_pair_is_the_exact_weighted_adjoint_and_constant_exact():
    gradient, divergence = _central_pair()
    apar = _random(1)[0]
    flutter = lambda f: -RHO * _bracket(apar, f)  # noqa: E731
    g, d = flutter_pair(gradient, divergence, lambda f: flutter(f) - flutter(jnp.ones_like(f)) * f, MASS, ACTIVE, ACTIVE)
    n = NX * NZ
    eye = jnp.eye(n).reshape((n, NX, NZ))
    dense = lambda op: np.asarray(jax.lax.map(lambda e: op(e).reshape(-1), eye)).T  # noqa: E731
    G, D = dense(g), dense(d)
    mass = float(MASS[0, 0])
    assert np.linalg.norm(D + G.T) < 1e-12 * np.linalg.norm(D)  # D = -M^-1 G^T M with uniform M
    assert np.max(np.abs(G @ np.ones(n))) < 1e-12 * np.abs(G).max()
    assert np.max(np.abs(mass * (np.ones(n) @ D))) < 1e-12 * np.abs(D).sum(0).max() * mass
    # the electrostatic default: no flutter, no change of the pair
    assert np.linalg.norm(dense(gradient) - dense(_central_pair()[0])) == 0.0


# ---- the production HSX model (set DRBX_TEST_GEOMETRY_BUNDLE, like test_hsx_parallel_pairs) ----------------------

BUNDLE = os.environ.get("DRBX_TEST_GEOMETRY_BUNDLE")
hsx = pytest.mark.skipif(BUNDLE is None, reason="set DRBX_TEST_GEOMETRY_BUNDLE")


class _Captured(Exception):
    pass


def _capture(tmp_path, beta):
    from drbx import deck
    from drbx.fci_braginskii.run import HsxPhysics

    path = tmp_path / "hsx.toml"
    path.write_text(f'[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "{BUNDLE}"\noutput = "{tmp_path / "o" / "hsx"}"\n'
                    f"[physics]\nelectromagnetic = true\nbeta = {beta}\n")
    name, cfg = deck.load_deck(path)
    assert cfg["physics"] == HsxPhysics(True, beta)
    cfg["fci_braginskii"] = dataclasses.replace(cfg["fci_braginskii"], final_time=0.0075, num_steps=10)
    real = jax.shard_map

    def stop(fun, **kwargs):
        raise _Captured(sys._getframe(1).f_locals)

    jax.shard_map = stop
    try:
        deck.BACKENDS[name].run(cfg, lambda message: None)
    except _Captured as caught:
        return caught.args[0]
    finally:
        jax.shard_map = real
    raise RuntimeError("the run never reached its first shard_map")


def _on_model(cap, function, out_specs=P()):
    body = lambda s, cf, mf, cv: function(cap["build_local_model"](cf, mf, cv), s)  # noqa: E731
    spec = cap["geometry_spec"]
    mapped = jax.shard_map(body, mesh=cap["mesh"], in_specs=(cap["state_spec"], spec, spec, spec), out_specs=out_specs, check_vma=False)
    return jax.jit(mapped)(cap["state"], cap["cell_fields"], cap["map_fields"], cap["control_volume_fields"])


@hsx
@pytest.mark.slow
def test_hsx_ampere_round_trip_and_flutter_pair_are_exact(tmp_path):
    """On the 16^3 HSX model: ``w -> Ve = w + mu A -> w`` closes through the Helmholtz and Poisson solves, and the
    flutter pair is the exact weighted adjoint (dense matrices) with constants exact."""

    beta = 1.0e-2
    cap = _capture(tmp_path, beta)
    shape = tuple(int(n) for n in cap["state"].density.shape)

    def probe(model, state):
        rng = np.random.default_rng(0)
        bump = state.density - jnp.mean(state.density)
        state = state.replace(Vi=0.05 * bump / jnp.max(jnp.abs(bump)), Ve=-0.02 * bump / jnp.max(jnp.abs(bump)))
        physical, em = model.em_physical(state)
        back = em.em_canonical(physical)
        face_bc, context = em._face_bcs(state), em._stencil_builder_context()
        support = em._em_wrap_pair(*em._fci_support_core_pair(face_bc=face_bc, context=context)[:2], em._fci_pair_active_mask(), face_bc, context)
        n = int(np.prod(shape))
        eye = jnp.eye(n).reshape((n,) + shape)
        dense = lambda op: jax.lax.map(lambda e: op(e).reshape(-1), eye, batch_size=64).T  # noqa: E731
        plain = em._fci_support_core_pair(face_bc=face_bc, context=context)[:2]
        return (jnp.max(jnp.abs(back.Ve - state.Ve)) / jnp.max(jnp.abs(state.Ve)), jnp.max(jnp.abs(em.apar)),
                dense(support[0]), dense(support[1]), dense(plain[0]), em._fci_pair_cell_mass(), em.geometry.active_cell_mask_owned,
                jnp.max(jnp.abs(physical.Ve - state.Ve)))

    mismatch, apar_max, gradient, divergence, plain_gradient, volume, active, ve_shift = jax.tree.map(np.asarray, _on_model(cap, probe))
    assert mismatch < 1e-4 and apar_max > 0.0  # Helmholtz vs Poisson closure, to the solver tolerance
    act = np.flatnonzero(active.ravel())
    v = volume.ravel()[act]
    G, D, G0 = (m[np.ix_(act, act)] for m in (gradient, divergence, plain_gradient))
    assert np.linalg.norm(D + (G.T * v[None]) / v[:, None]) <= 1e-12 * np.linalg.norm(D)
    assert np.max(np.abs(G.sum(1))) <= 1e-12 * np.max(np.abs(G).sum(1))
    assert np.linalg.norm(G - G0) > 1e-9 * np.linalg.norm(G0)  # the flutter is present (A_par ~ beta Vi is small: rho_star A)


@hsx
@pytest.mark.slow
def test_electromagnetic_step_gradient_matches_finite_difference(tmp_path, monkeypatch):
    """``jax.grad`` of a scalar of one compiled EM IMEX step against central differences.

    Needs the implicit-function VJP of the Krylov solves (``solvax.implicit.linear_solve`` in
    ``solvax_gmres_solve``).  The loss differentiates the ion velocity that sources Ampere's law, so the
    gradient passes the Helmholtz solve, the flutter pairs and the implicit current/phi pair.  The driver's
    diagnostic ``pmin``/``pmax`` have no JVP rule and are cut from the tape.
    """

    cap, real_shard_map, real_jit = {}, jax.shard_map, jax.jit

    class _Stop(BaseException):
        pass

    def shard_map_spy(f, *a, **k):
        out = real_shard_map(f, *a, **k)
        if getattr(f, "__name__", "") == "full_imex_advance":
            cap["advance"] = out
        return out

    class _JitSpy:
        def __init__(self, f):
            self.f = real_jit(f)

        def lower(self, *args):
            if "advance" in cap and len(args) >= 4 and hasattr(args[0], "Te"):
                cap["args"] = args[:4] + (jnp.asarray(0.0),)
                raise _Stop
            return self.f.lower(*args)

        def __call__(self, *a, **k):
            return self.f(*a, **k)

    from drbx import deck

    monkeypatch.setattr(jax, "shard_map", shard_map_spy)
    monkeypatch.setattr(jax, "jit", lambda f, *a, **k: _JitSpy(f) if not (a or k) else real_jit(f, *a, **k))
    for name in ("pmin", "pmax"):
        real = getattr(jax.lax, name)
        monkeypatch.setattr(jax.lax, name, lambda x, ax, real=real: real(jax.lax.stop_gradient(x), ax))
    path = tmp_path / "hsx.toml"
    path.write_text(f'[model]\nbackend = "fci_braginskii"\n[fci_braginskii]\ngeometry = "{BUNDLE}"\noutput = "{tmp_path / "o" / "hsx"}"\n'
                    "final_time = 0.00075\nnum_steps = 1\n[physics]\nelectromagnetic = true\nbeta = 1.0e-2\n")
    name, config = deck.load_deck(path)
    with pytest.raises(_Stop):
        deck.BACKENDS[name].run(config, lambda message: None)
    monkeypatch.setattr(jax, "jit", real_jit)
    monkeypatch.setattr(jax, "shard_map", real_shard_map)

    advance = cap["advance"]
    state, cell_fields, map_fields, control_volume_fields, time = cap["args"]
    bump = state.density - jnp.mean(state.density)
    bump = bump / jnp.max(jnp.abs(bump))

    # Flows of one sign everywhere: the model's upwinding and |V| signal speeds have kinks at V = 0 that a central
    # difference averages over and a derivative does not.
    base = state.replace(Vi=0.02 * (1.0 + 0.5 * bump), Ve=0.01 * (1.0 - 0.5 * bump))

    def loss(a):
        out = advance(base.replace(Vi=base.Vi + a * 0.02 * bump), cell_fields, map_fields, control_volume_fields, time)
        # the flow and potential fields respond to the ion velocity at order one; the density and temperatures
        # would bury the derivative under the solver tolerance of a loss of order 1e3
        return sum(jnp.sum(getattr(out[0], name) ** 2) for name in ("Vi", "Ve", "vorticity", "phi"))

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.asarray(0.0))
    forward, eps = jax.jit(loss), 1.0e-2
    fd = (float(forward(jnp.asarray(eps))) - float(forward(jnp.asarray(-eps)))) / (2.0 * eps)
    print(f"[gradient] loss {float(value):.6e} jax.grad {float(grad):.6e} central difference {fd:.6e} rel {abs(float(grad) / fd - 1.0):.2e}")
    assert np.isfinite(float(value)) and np.isfinite(float(grad)) and abs(fd) > 0.0
    assert float(grad) == pytest.approx(fd, rel=1.0e-3)
