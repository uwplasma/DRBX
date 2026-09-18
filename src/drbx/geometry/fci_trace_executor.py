"""Batched, optionally sharded execution of the JAX FCI point tracer.

The continuous-field tracer is deliberately a fixed-shape JAX kernel.  This
module supplies the small amount of runtime plumbing needed by a geometry
producer: seed rows are padded to a batch shape divisible by the selected
devices, only the leading row axis is sharded, and the resulting arrays are
copied back to NumPy.  Field/metric state is replicated; there are no
cross-seed reductions or collectives in this executor.

The host-side NumPy tracer remains the reference implementation.  An
``FciTraceExecutor`` owns no geometry and performs no qualification or cache
lookups; it only executes ``trace_fci_points_to_plane_jax`` for the supplied
callable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import resource
import sys
from time import perf_counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .fci_trace_jax import trace_fci_points_to_plane_jax


@dataclass(frozen=True)
class TraceExecutionResult:
    """NumPy outputs and execution accounting for one batched trace call.

    ``outputs`` has the same keys as the underlying JAX tracer and contains
    only the requested seed rows (padding rows have been removed).  Mapping
    access is provided as a convenience, so both ``result.outputs["endpoint"]``
    and ``result["endpoint"]`` are valid.
    """

    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any]
    timing: dict[str, float]
    work: dict[str, int]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.outputs[name]

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    def keys(self):
        return self.outputs.keys()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable result envelope."""

        return {
            "outputs": self.outputs,
            "metadata": self.metadata,
            "timing": self.timing,
            "work": self.work,
        }


def _block_until_ready(tree: Any) -> Any:
    """Synchronize every leaf of a JAX result tree."""

    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def _resolve_devices(devices: Any = None, device_count: int | None = None) -> list[Any]:
    """Resolve a caller selection without silently falling back to another device."""

    available = list(jax.local_devices())
    if devices is None:
        if device_count is None:
            selected = available
        else:
            if not isinstance(device_count, (int, np.integer)) or int(device_count) < 1:
                raise ValueError("device_count must be a positive integer")
            if int(device_count) > len(available):
                raise ValueError(
                    f"requested {int(device_count)} devices but only "
                    f"{len(available)} local JAX devices are available"
                )
            selected = available[: int(device_count)]
    elif isinstance(devices, (int, np.integer)):
        count = int(devices)
        if count < 1:
            raise ValueError("devices integer must be positive")
        if count > len(available):
            raise ValueError(
                f"requested {count} devices but only {len(available)} "
                "local JAX devices are available"
            )
        if device_count is not None and int(device_count) != count:
            raise ValueError("devices and device_count disagree")
        selected = available[:count]
    else:
        selected = list(devices)
        if device_count is not None and len(selected) != int(device_count):
            raise ValueError("devices and device_count disagree")
        if not selected:
            raise ValueError("devices must contain at least one local device")

    if not selected:
        raise ValueError("no local JAX devices are available")
    available_ids = {id(device) for device in available}
    if any(id(device) not in available_ids for device in selected):
        raise ValueError("every selected device must be a local JAX device")
    if len({id(device) for device in selected}) != len(selected):
        raise ValueError("selected devices must be unique")
    return selected


def _is_dynamic_callable_pytree(evaluator: Any) -> bool:
    """Whether a callable evaluator can safely be passed as a JAX argument."""

    if not callable(evaluator):
        return False
    try:
        leaves, _ = jax.tree_util.tree_flatten(evaluator)
    except Exception:
        return False
    # Plain Python functions flatten to one static leaf.  Registered runtime
    # evaluators flatten to their numerical coefficient leaves.
    return bool(leaves) and not all(callable(leaf) for leaf in leaves)


class FciTraceExecutor:
    """Execute fixed-shape FCI traces on one or more local devices.

    Parameters
    ----------
    field_evaluator:
        JAX-compatible callable or callable PyTree.  A callable PyTree (for
        example a coefficient-backed field evaluator) is passed to the JIT
        with a replicated ``PartitionSpec()``; a plain Python callable is
        captured as a static closure.
    batch_size:
        Compiled leading batch size.  It is rounded up to a multiple of the
        selected device count.  Every nonempty chunk uses this same shape.
    devices, device_count:
        Explicit local device list or a prefix count of ``jax.local_devices``.
    """

    def __init__(
        self,
        field_evaluator: Callable,
        batch_size: int,
        *,
        devices: int | Sequence[Any] | None = None,
        device_count: int | None = None,
        grid_bounds: Any | None = None,
        substeps: int = 4,
        periodic_axes: tuple[bool, bool, bool] = (False, True, True),
        axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
        min_abs_bz: float = 1.0e-30,
        axis_epsilon: float | None = None,
    ) -> None:
        if not callable(field_evaluator):
            raise TypeError("field_evaluator must be callable")
        if not isinstance(batch_size, (int, np.integer)) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")
        self.field_evaluator = field_evaluator
        self.devices = tuple(_resolve_devices(devices, device_count))
        self.device_count = len(self.devices)
        self.requested_batch_size = int(batch_size)
        self.batch_size = int((int(batch_size) + self.device_count - 1) // self.device_count * self.device_count)
        self.substeps = substeps
        self.periodic_axes = tuple(bool(value) for value in periodic_axes)
        self.axis_regular_axes = tuple(bool(value) for value in axis_regular_axes)
        self.min_abs_bz = float(min_abs_bz)
        if axis_epsilon is not None and float(axis_epsilon) <= 0.0:
            raise ValueError("axis_epsilon must be positive when provided")
        self.axis_epsilon = None if axis_epsilon is None else float(axis_epsilon)
        self._grid_bounds = None if grid_bounds is None else np.asarray(grid_bounds, dtype=np.float64)
        if self._grid_bounds is not None and self._grid_bounds.shape != (3, 2):
            raise ValueError("grid_bounds must have shape (3, 2)")
        self._mesh = jax.sharding.Mesh(np.asarray(self.devices, dtype=object), ("trace",))
        self._point_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec("trace", None)
        )
        self._vector_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec("trace")
        )
        self._replicated_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec()
        )
        self._dynamic_evaluator = _is_dynamic_callable_pytree(field_evaluator)
        self.evaluator_state_bytes = sum(
            int(getattr(value, "size", 0))
            * int(getattr(getattr(value, "dtype", None), "itemsize", 0))
            for value in jax.tree_util.tree_leaves(field_evaluator)
            if hasattr(value, "shape") and hasattr(value, "dtype")
        )
        self._compiled_kernel = None
        self._compiled_executable = None
        self._explicit_lowering_supported = True
        self._compiled_memory: dict[str, int] = {}
        self._compiled_evaluator_spec = None
        self._device_evaluator = None
        if self._dynamic_evaluator:
            leaves = jax.tree_util.tree_leaves(field_evaluator)
            if leaves and all(
                isinstance(value, jax.Array)
                and value.sharding.is_equivalent_to(
                    self._replicated_sharding, value.ndim
                )
                for value in leaves
            ):
                # Geometry stages may share a coefficient state already
                # replicated by the center-map executor. Reuse its buffers.
                self._device_evaluator = field_evaluator
        self._bounds_device = None

    @property
    def device_evaluator(self) -> Any:
        """Return the once-replicated evaluator, or the unplaced source state."""

        return (
            self.field_evaluator
            if self._device_evaluator is None
            else self._device_evaluator
        )

    def _kernel(self):
        """Build the fixed-shape jitted kernel once per executor."""

        if self._compiled_kernel is not None:
            return self._compiled_kernel
        evaluator_spec = None
        if self._dynamic_evaluator:
            # A single sharding is a valid PyTree prefix and applies to every
            # numerical leaf.  Building a tree of Sharding objects would call
            # evaluator ``tree_unflatten`` methods with non-array placeholders,
            # which immutable numeric dataclasses correctly reject.
            evaluator_spec = self._replicated_sharding

            def kernel(bounds, evaluator, points, eta, valid):
                return trace_fci_points_to_plane_jax(
                    bounds,
                    evaluator,
                    points,
                    eta,
                    substeps=self.substeps,
                    periodic_axes=self.periodic_axes,
                    axis_regular_axes=self.axis_regular_axes,
                    min_abs_bz=self.min_abs_bz,
                    axis_epsilon=self.axis_epsilon,
                    valid_mask=valid,
                )

            self._compiled_kernel = jax.jit(
                kernel,
                in_shardings=(
                    self._replicated_sharding,
                    evaluator_spec,
                    self._point_sharding,
                    self._vector_sharding,
                    self._vector_sharding,
                ),
            )
        else:
            evaluator = self.field_evaluator

            def kernel(bounds, points, eta, valid):
                return trace_fci_points_to_plane_jax(
                    bounds,
                    evaluator,
                    points,
                    eta,
                    substeps=self.substeps,
                    periodic_axes=self.periodic_axes,
                    axis_regular_axes=self.axis_regular_axes,
                    min_abs_bz=self.min_abs_bz,
                    axis_epsilon=self.axis_epsilon,
                    valid_mask=valid,
                )

            self._compiled_kernel = jax.jit(
                kernel,
                in_shardings=(
                    self._replicated_sharding,
                    self._point_sharding,
                    self._vector_sharding,
                    self._vector_sharding,
                ),
            )
        self._compiled_evaluator_spec = evaluator_spec
        return self._compiled_kernel

    def _run_batch(self, bounds, points, eta, valid):
        kernel = self._kernel()
        if self._bounds_device is None:
            self._bounds_device = jax.device_put(
                np.asarray(bounds, dtype=np.float64), self._replicated_sharding
            )
        points_device = jax.device_put(
            np.asarray(points, dtype=np.float64), self._point_sharding
        )
        eta_device = jax.device_put(
            np.asarray(eta, dtype=np.float64), self._vector_sharding
        )
        valid_device = jax.device_put(
            np.asarray(valid, dtype=bool), self._vector_sharding
        )
        if self._dynamic_evaluator:
            if self._device_evaluator is None:
                # The MAKEGRID spline state is large.  Replicate it once per
                # executor, not once per trajectory chunk.
                self._device_evaluator = jax.device_put(
                    self.field_evaluator, self._replicated_sharding
                )
            arguments = (
                self._bounds_device,
                self._device_evaluator,
                points_device,
                eta_device,
                valid_device,
            )
        else:
            arguments = (
                self._bounds_device,
                points_device,
                eta_device,
                valid_device,
            )
        if self._compiled_executable is None and self._explicit_lowering_supported:
            try:
                self._compiled_executable = kernel.lower(*arguments).compile()
            except TypeError:
                # User-defined callable PyTrees may not support abstract
                # placeholder reconstruction. Normal jit dispatch remains
                # valid; only compiler memory reporting is unavailable.
                self._explicit_lowering_supported = False
            if self._compiled_executable is not None:
                memory = self._compiled_executable.memory_analysis()
                if memory is not None:
                    for name in (
                        "argument_size_in_bytes",
                        "output_size_in_bytes",
                        "temp_size_in_bytes",
                        "alias_size_in_bytes",
                        "host_argument_size_in_bytes",
                        "host_output_size_in_bytes",
                        "host_temp_size_in_bytes",
                    ):
                        value = getattr(memory, name, None)
                        if value is not None:
                            self._compiled_memory[name] = int(value)
        result = (
            kernel(*arguments)
            if self._compiled_executable is None
            else self._compiled_executable(*arguments)
        )
        return _block_until_ready(result)

    def trace(
        self,
        seed_points: Any,
        eta_step: Any,
        *,
        grid_bounds: Any | None = None,
        valid_mask: Any | None = None,
    ) -> TraceExecutionResult:
        """Trace arbitrary seed rows in fixed-size chunks and return NumPy arrays."""

        points = np.asarray(seed_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("seed_points must have shape (n, 3)")
        n_seed = int(points.shape[0])
        if np.ndim(eta_step) == 0:
            eta = np.full(n_seed, float(eta_step), dtype=np.float64)
        else:
            eta = np.asarray(eta_step, dtype=np.float64)
            if eta.shape != (n_seed,):
                raise ValueError(f"eta_step must be scalar or shape ({n_seed},)")
        if valid_mask is None:
            valid = np.ones(n_seed, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (n_seed,):
                raise ValueError(f"valid_mask must have shape ({n_seed},)")
        bounds = self._grid_bounds if grid_bounds is None else np.asarray(grid_bounds, dtype=np.float64)
        if bounds is None:
            raise ValueError("grid_bounds must be provided to the executor or trace()")
        if bounds.shape != (3, 2):
            raise ValueError("grid_bounds must have shape (3, 2)")
        if n_seed == 0:
            empty = {
                "endpoint": np.empty((0, 3), dtype=np.float64),
                "length": np.empty((0,), dtype=np.float64),
                "boundary": np.empty((0,), dtype=bool),
                "endpoint_b_contravariant": np.empty((0, 3), dtype=np.float64),
                "endpoint_bmag": np.empty((0,), dtype=np.float64),
            }
            return TraceExecutionResult(empty, self._metadata(0, 0, 0), self._timing(0.0, 0.0, 0.0), self._work(0, 0))

        outputs: dict[str, np.ndarray] = {}
        batch_count = (n_seed + self.batch_size - 1) // self.batch_size
        first_seconds = 0.0
        remaining_seconds = 0.0
        padded_rows = 0
        batch_points = np.zeros((self.batch_size, 3), dtype=np.float64)
        batch_eta = np.zeros(self.batch_size, dtype=np.float64)
        batch_valid = np.zeros(self.batch_size, dtype=bool)
        for batch_index in range(batch_count):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, n_seed)
            rows = stop - start
            pad = self.batch_size - rows
            padded_rows += pad
            batch_points[rows:] = 0.0
            batch_eta[rows:] = 0.0
            batch_valid[:] = False
            batch_points[:rows] = points[start:stop]
            batch_eta[:rows] = eta[start:stop]
            batch_valid[:rows] = valid[start:stop]
            begin = perf_counter()
            result = self._run_batch(bounds, batch_points, batch_eta, batch_valid)
            elapsed = perf_counter() - begin
            if batch_index == 0:
                first_seconds = elapsed
            else:
                remaining_seconds += elapsed
            for name, value in result.items():
                host_value = np.asarray(value)
                if name not in outputs:
                    outputs[name] = np.empty(
                        (n_seed,) + host_value.shape[1:], dtype=host_value.dtype
                    )
                outputs[name][start:stop] = host_value[:rows]
        timing = self._timing(first_seconds, remaining_seconds, first_seconds + remaining_seconds)
        metadata = self._metadata(n_seed, padded_rows, batch_count)
        work = self._work(n_seed, padded_rows, batch_count)
        return TraceExecutionResult(outputs, metadata, timing, work)

    __call__ = trace

    def _metadata(self, seed_rows: int, padded_rows: int, batches: int) -> dict[str, Any]:
        return {
            "device_count": self.device_count,
            "devices": tuple(str(device) for device in self.devices),
            "requested_batch_size": self.requested_batch_size,
            "compiled_batch_size": self.batch_size,
            "seed_rows": int(seed_rows),
            "padded_rows": int(padded_rows),
            "batches": int(batches),
            "leading_axis_sharded": True,
            "field_state_replicated": True,
            "cross_seed_collectives": False,
            "evaluator_state_bytes_per_device": int(
                self.evaluator_state_bytes
            ),
            "evaluator_state_bytes_all_devices": int(
                self.evaluator_state_bytes * self.device_count
            ),
            "peak_host_rss_bytes": _peak_rss_bytes(),
            "compiled_memory": dict(self._compiled_memory),
        }

    def _timing(self, first: float, remaining: float, total: float) -> dict[str, float]:
        return {
            "compile_plus_first_batch_seconds": float(first),
            "remaining_batches_seconds": float(remaining),
            "total_seconds": float(total),
        }

    def _work(self, seed_rows: int, padded_rows: int, batches: int = 0) -> dict[str, int]:
        rows = int(seed_rows + padded_rows)
        return {
            "seed_rows": int(seed_rows),
            "padded_rows": int(padded_rows),
            "batches": int(batches),
            "rk4_substeps": int(self.substeps) * rows,
            "rk4_stages": 4 * int(self.substeps) * rows,
            # One initial sample is carried between substeps. Each substep has
            # three new RK stages, a full-endpoint sample, and a possible
            # wall-hit sample. One final sample supplies returned endpoint B.
            "field_evaluations": (5 * int(self.substeps) + 2) * rows,
        }


def trace_fci_points_batched(
    grid_bounds: Any,
    field_evaluator: Callable,
    seed_points: Any,
    eta_step: Any,
    *,
    batch_size: int,
    devices: int | Sequence[Any] | None = None,
    device_count: int | None = None,
    substeps: int = 4,
    periodic_axes: tuple[bool, bool, bool] = (False, True, True),
    axis_regular_axes: tuple[bool, bool, bool] = (False, False, False),
    min_abs_bz: float = 1.0e-30,
    axis_epsilon: float | None = None,
    valid_mask: Any | None = None,
) -> TraceExecutionResult:
    """Functional convenience wrapper around :class:`FciTraceExecutor`."""

    executor = FciTraceExecutor(
        field_evaluator,
        batch_size,
        devices=devices,
        device_count=device_count,
        grid_bounds=grid_bounds,
        substeps=substeps,
        periodic_axes=periodic_axes,
        axis_regular_axes=axis_regular_axes,
        min_abs_bz=min_abs_bz,
        axis_epsilon=axis_epsilon,
    )
    return executor.trace(seed_points, eta_step, valid_mask=valid_mask)


__all__ = ["TraceExecutionResult", "FciTraceExecutor", "trace_fci_points_batched"]
