from dataclasses import dataclass

import jax.numpy as jnp

from drbx.native.fci_boundary_imex_preconditioner import build_coupled_boundary_preconditioner


@dataclass
class _State:
    density: jnp.ndarray
    Te: jnp.ndarray
    Ti: jnp.ndarray
    Vi: jnp.ndarray
    Ve: jnp.ndarray
    phi: jnp.ndarray
    vorticity: jnp.ndarray

    def replace(self, **updates):
        values = self.__dict__.copy()
        values.update(updates)
        return _State(**values)


@dataclass
class _Context:
    model: object
    base: object
    dt: float
    active_owned: jnp.ndarray
    mass_weights: jnp.ndarray
    material_jacobian: jnp.ndarray
    phi_preconditioner: object = None
    coarse_correction: object = None
    augmented: bool = False

    def unpack(self, value):
        return value

    def pack(self, state, multiplier):
        return state, multiplier


def _state(size):
    ones = tuple(jnp.ones(size) for _ in range(5))
    return _State(*ones, jnp.full(size, 2.0), jnp.full(size, 4.0))


def test_material_block_solves_all_five_fields_and_preserves_omega():
    matrix = jnp.asarray([[0.1 if i == j else 0.02 for j in range(5)] for i in range(5)])
    context = _Context(None, None, 0.25, jnp.asarray([True, False]), jnp.ones(2), matrix)
    result, multiplier = build_coupled_boundary_preconditioner(context)((_state(2), jnp.asarray(6.0)))
    expected = jnp.linalg.solve(jnp.eye(5) - 0.25 * matrix, jnp.ones((5, 1)))[..., 0]
    assert jnp.allclose(jnp.stack((result.density[0], result.Te[0], result.Ti[0], result.Vi[0], result.Ve[0])), expected)
    assert jnp.all(result.density[1:] == 1.0)
    assert jnp.all(result.vorticity == 4.0)
    assert multiplier == 6.0


def test_phi_preconditioner_and_coarse_correction_are_reusable():
    context = _Context(None, None, 0.1, jnp.asarray([True]), jnp.ones(1), jnp.zeros((5, 5)))
    context.phi_preconditioner = lambda value: 2.0 * value
    context.coarse_correction = lambda value: 0.5 * value
    result, _ = build_coupled_boundary_preconditioner(context)((_state(1), jnp.asarray(0.0)))
    assert jnp.allclose(result.phi, 5.0)


def test_augmented_constant_and_gauge_rows_are_nonsingular():
    context = _Context(None, None, 0.1, jnp.asarray([True, True]), jnp.ones(2), jnp.zeros((5, 5)), augmented=True)
    constant = _state(2).replace(phi=jnp.ones(2))
    result, multiplier = build_coupled_boundary_preconditioner(context)((constant, jnp.asarray(0.0)))
    assert jnp.allclose(multiplier, 1.0)
    assert jnp.allclose(result.phi, 0.0)
    gauge = _state(2).replace(phi=jnp.zeros(2))
    result, multiplier = build_coupled_boundary_preconditioner(context)((gauge, jnp.asarray(1.0)))
    assert jnp.allclose(multiplier, 0.0)
    assert jnp.allclose(result.phi, 1.0)


def test_augmented_uses_mass_quotient_then_physical_gauge_weights():
    context = _Context(None, None, 0.1, jnp.asarray([True, True]), jnp.asarray([1.0, 3.0]), jnp.zeros((5, 5)), augmented=True)
    context.gauge_weights = jnp.asarray([2.0, 1.0])
    state = _state(2).replace(phi=jnp.asarray([1.0, 2.0]))
    result, multiplier = build_coupled_boundary_preconditioner(context)((state, jnp.asarray(0.0)))
    assert jnp.allclose(multiplier, 1.75)
    assert jnp.allclose(jnp.sum(context.gauge_weights * result.phi), 0.0)
