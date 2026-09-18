"""Tests for fixed-shape and leading-axis-sharded FCI trace execution."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest


BOUNDS = np.asarray([[0.0, 1.0], [0.0, 2.0 * np.pi], [0.0, 1.0]])


def _field(points):
    import jax.numpy as jnp

    bx = 0.25 + 0.1 * points[:, 0]
    by = -0.1 + 0.02 * points[:, 1]
    return jnp.stack((bx, by, jnp.ones_like(bx)), axis=-1), jnp.ones_like(bx)


def _inputs(n=7):
    points = np.column_stack(
        (
            np.linspace(0.1, 0.9, n),
            np.linspace(0.2, 5.8, n),
            np.linspace(0.15, 0.85, n),
        )
    )
    eta = np.linspace(-0.15, 0.25, n)
    return points, eta


def test_chunking_padding_and_reuse_returns_numpy_outputs():
    from drbx.geometry.fci_trace_executor import FciTraceExecutor

    points, eta = _inputs()
    executor = FciTraceExecutor(
        _field,
        batch_size=4,
        device_count=1,
        grid_bounds=BOUNDS,
        substeps=3,
    )
    first = executor.trace(points, eta)
    second = executor.trace(points[:3], eta[:3])

    assert all(isinstance(value, np.ndarray) for value in first.outputs.values())
    assert first.outputs["endpoint"].shape == points.shape
    assert first.outputs["length"].shape == (points.shape[0],)
    assert first.metadata["compiled_batch_size"] == 4
    assert first.metadata["batches"] == 2
    assert first.metadata["padded_rows"] == 1
    assert first.metadata["cross_seed_collectives"] is False
    assert first.metadata["evaluator_state_bytes_per_device"] == 0
    assert first.metadata["peak_host_rss_bytes"] > 0
    assert first.metadata["compiled_memory"]["temp_size_in_bytes"] >= 0
    assert first.metadata["compiled_memory"]["output_size_in_bytes"] > 0
    assert first.timing["compile_plus_first_batch_seconds"] >= 0.0
    assert first.timing["remaining_batches_seconds"] >= 0.0
    assert first.work["field_evaluations"] == (5 * 3 + 2) * (7 + 1)
    np.testing.assert_allclose(second.outputs["endpoint"], first.outputs["endpoint"][:3])


def test_sharding_selection_and_explicit_grid_bounds():
    from drbx.geometry.fci_trace_executor import FciTraceExecutor

    points, eta = _inputs(4)
    executor = FciTraceExecutor(_field, batch_size=3, device_count=1, substeps=2)
    result = executor(points, eta, grid_bounds=BOUNDS)
    assert result.metadata["device_count"] == 1
    np.testing.assert_allclose(result.outputs["endpoint"][:, 2], np.mod(points[:, 2] + eta, 1.0), atol=1.0e-14)


def test_invalid_device_count_and_shapes_fail_early():
    from drbx.geometry.fci_trace_executor import FciTraceExecutor

    with pytest.raises(ValueError, match="device_count"):
        FciTraceExecutor(_field, batch_size=2, device_count=0)
    executor = FciTraceExecutor(_field, batch_size=2, device_count=1, grid_bounds=BOUNDS)
    with pytest.raises(ValueError, match="shape"):
        executor(np.zeros((2, 2)), 0.1)
    with pytest.raises(ValueError, match="eta_step"):
        executor(np.zeros((2, 3)), np.zeros(3))


def test_two_cpu_sharded_and_single_device_results_match_in_subprocess():
    code = r'''
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from drbx.geometry.fci_trace_executor import FciTraceExecutor

bounds = np.asarray([[0., 1.], [0., 2.*np.pi], [0., 1.]])
points = np.asarray([[.1,.2,.15],[.25,1.2,.3],[.4,2.8,.45],[.55,4.1,.6],[.7,5.5,.75]])
eta = np.asarray([-.15,-.05,.1,.2,.25])
def field(q):
    bx = .25 + .1*q[:,0]
    by = -.1 + .02*q[:,1]
    return jnp.stack((bx,by,jnp.ones_like(bx)),axis=-1), jnp.ones_like(bx)

@jax.tree_util.register_pytree_node_class
class FieldState:
    def __init__(self, coefficients):
        self.coefficients = jnp.asarray(coefficients)
    def tree_flatten(self):
        return (self.coefficients,), None
    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(children[0])
    def __call__(self, q):
        bx = self.coefficients[0] + self.coefficients[1]*q[:,0]
        by = self.coefficients[2] + self.coefficients[3]*q[:,1]
        return jnp.stack((bx,by,jnp.ones_like(bx)),axis=-1), jnp.ones_like(bx)

single = FciTraceExecutor(field, batch_size=4, device_count=1, grid_bounds=bounds, substeps=3)
multi = FciTraceExecutor(field, batch_size=4, device_count=2, grid_bounds=bounds, substeps=3)
a, b = single(points, eta), multi(points, eta)
assert len(jax.local_devices()) == 2
assert b.metadata["device_count"] == 2
assert b.metadata["compiled_batch_size"] == 4
assert b.metadata["padded_rows"] == 3
assert b.metadata["cross_seed_collectives"] is False
for name in a.outputs:
    np.testing.assert_allclose(a.outputs[name], b.outputs[name], rtol=1e-12, atol=1e-12)
state_multi = FciTraceExecutor(
    FieldState([.25,.1,-.1,.02]), batch_size=4, device_count=2,
    grid_bounds=bounds, substeps=3,
)
c = state_multi(points, eta)
for name in a.outputs:
    np.testing.assert_allclose(a.outputs[name], c.outputs[name], rtol=1e-12, atol=1e-12)
print("ok")
'''
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        pytest.fail(f"two-device subprocess failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    assert "ok" in completed.stdout
