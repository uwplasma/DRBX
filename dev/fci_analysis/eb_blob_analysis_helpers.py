"""Analysis-only helpers for saved shifted-torus EB-blob histories.

This module deliberately contains no simulation driver and does not import the
legacy test module.  It keeps the geometry, stencil, history, and plotting
pieces needed by the post-processing scripts after the test is removed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from drbx.geometry import (
    BFieldGeometry,
    FaceBFieldGeometry,
    FciGeometry3D,
    FciMaps3D,
    Spacing3D,
)

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_TESTS_DIR = _REPO_ROOT / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from shifted_torus_4field_mms_helpers import (  # noqa: E402
    build_shifted_torus_4field_geometry,
)


DEFAULT_RESOLUTION = 64
PERIODIC_AXES = (False, True, True)
radial_b_fraction = 1.0e-2
z0 = np.pi


def _bmag(B_contra, g_cov):
    return jnp.sqrt(jnp.maximum(jnp.einsum("...i,...ij,...j->...", B_contra, g_cov, B_contra), 0.0))


def _build_eb_blob_geometry(shape, *, radial_fraction=radial_b_fraction, construct_fci_maps=True):
    shape = tuple(int(value) for value in shape)
    rho_min = 0.5 / shape[0]
    base = build_shifted_torus_4field_geometry(
        shape, x_min=rho_min, x_max=1.0 - rho_min, construct_fci_maps=False
    )
    grid = base.grid

    def field(bfield, metric, theta):
        scale = float(radial_fraction) * jnp.asarray(
            bfield.B_contra[..., 2], dtype=jnp.float64
        )
        contra = jnp.stack(
            (
                scale * jnp.cos(theta),
                bfield.B_contra[..., 1],
                bfield.B_contra[..., 2],
            ),
            axis=-1,
        )
        return BFieldGeometry(
            B_contra=contra,
            Bmag=_bmag(contra, metric.g_cov),
        )

    cell_theta = jnp.broadcast_to(grid.y.centers[None, :, None], shape)
    cell_bfield = field(base.cell_bfield, base.cell_metric, cell_theta)
    face_bfield = FaceBFieldGeometry(
        x=field(
            base.face_bfield.x,
            base.face_metric.x,
            jnp.broadcast_to(grid.y.centers[None, :, None], base.face_metric.x.shape),
        ),
        y=field(
            base.face_bfield.y,
            base.face_metric.y,
            jnp.broadcast_to(grid.y.faces[None, :, None], base.face_metric.y.shape),
        ),
        z=field(
            base.face_bfield.z,
            base.face_metric.z,
            jnp.broadcast_to(grid.y.centers[None, :, None], base.face_metric.z.shape),
        ),
    )
    if construct_fci_maps:
        # The analysis scripts only need map construction when explicitly requested.
        from drbx.geometry import build_fci_maps_from_b_contravariant
        raw = build_fci_maps_from_b_contravariant(grid, cell_bfield.B_contra, cell_bfield.Bmag, periodic_axes=PERIODIC_AXES)
        maps = FciMaps3D(**raw)
    else:
        zeros = jnp.zeros(shape, dtype=jnp.float64)
        maps = FciMaps3D(
            forward_x=zeros, forward_y=zeros, backward_x=zeros, backward_y=zeros,
            forward_endpoint_x=zeros, forward_endpoint_y=zeros, forward_endpoint_z=zeros,
            backward_endpoint_x=zeros, backward_endpoint_y=zeros, backward_endpoint_z=zeros,
            forward_length=jnp.ones(shape), backward_length=jnp.ones(shape),
            forward_boundary=zeros.astype(bool), backward_boundary=zeros.astype(bool),
        )
    spacing = Spacing3D(
        dx=jnp.broadcast_to(grid.x.widths[:, None, None], shape),
        dy=jnp.broadcast_to(grid.y.widths[None, :, None], shape),
        dz=jnp.broadcast_to(grid.z.widths[None, None, :], shape),
    )
    return FciGeometry3D(grid=grid, maps=maps, spacing=spacing, cell_metric=base.cell_metric,
                         face_metric=base.face_metric, cell_bfield=cell_bfield, face_bfield=face_bfield)


def _eb_blob_artifact_stem(run_name):
    name = str(run_name).strip()
    if not name:
        raise ValueError("run_name must be a non-empty string")
    return name


def _resolve_eb_blob_history_path(run_name, output_dir=None):
    path = Path(f"{_eb_blob_artifact_stem(run_name)}_histories.npz")
    return path if output_dir is None else Path(output_dir) / path.name


def _load_eb_blob_history(history_path):
    with np.load(history_path, allow_pickle=False) as history:
        names = ("times", "density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
        arrays = tuple(jnp.asarray(history[name], dtype=jnp.float64) for name in names)
        metadata = {key: history[key].item() if history[key].shape == () else history[key]
                    for key in history.files if key not in names}
    return (*arrays, metadata)


def _resolve_step_dump_dir(run_name, output_path=None):
    if output_path is None:
        candidate = Path(__file__).resolve().parent / str(run_name)
        nested = candidate / "step_dumps"
        if any(nested.glob("step_*.npz")):
            return nested
        if any(candidate.glob("step_*.npz")):
            return candidate
        return candidate
    candidate = Path(output_path)
    if candidate.is_file():
        return candidate.parent
    if candidate.name.endswith("_histories.npz"):
        return candidate.parent / "step_dumps"
    return candidate / "step_dumps" if candidate.name != "step_dumps" else candidate


def _load_eb_blob_step_history(step_dump_dir):
    step_files = sorted(Path(step_dump_dir).glob("step_*.npz"))
    if not step_files:
        raise FileNotFoundError(f"no step_*.npz files found in {step_dump_dir}")
    snapshots = []
    for fallback_index, step_file in enumerate(step_files):
        with np.load(step_file, allow_pickle=False) as step:
            index = int(step["step_index"]) if "step_index" in step.files else fallback_index
            time = float(step["time"]) if "time" in step.files else float(fallback_index)
            fields = tuple(np.asarray(step[name], dtype=np.float64) for name in ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity"))
        snapshots.append((index, time, *fields))
    snapshots.sort(key=lambda item: item[0])
    return tuple(np.asarray([item[index] for item in snapshots], dtype=np.float64) for index in range(1, 9))


def _eb_blob_z_indices(geometry, center, count=4):
    values = np.asarray(geometry.grid.z.centers, dtype=np.float64)
    center_index = int(np.argmin(np.abs(values - float(center))))
    offsets = np.arange(-(count // 2), -(count // 2) + count)
    return tuple(int((center_index + offset) % values.size) for offset in offsets)


def _save_eb_blob_time_traces(times, density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history, geometry, *, output_path, title="Shifted-torus EB blob time traces"):
    import matplotlib.pyplot as plt
    arrays = [np.asarray(value, dtype=np.float64) for value in (density_history, phi_history, te_history, ti_history, vi_history, ve_history, vorticity_history)]
    labels = ("density", "phi", "Te", "Ti", "Vi", "Ve", "vorticity")
    times = np.asarray(times, dtype=np.float64)
    weights = np.asarray(geometry.cell_metric.J, dtype=np.float64)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for axis, label, values in zip(axes.flat, labels, arrays):
        axis.plot(times, np.sqrt(np.mean(values * values, axis=(1, 2, 3))))
        axis.set_title(f"rms({label})")
        axis.grid(True, alpha=0.3)
    axes.flat[-1].plot(times, np.sum(arrays[0] * weights[None, ...], axis=(1, 2, 3)) / np.sum(weights))
    axes.flat[-1].set_title("J-weighted mean(density)")
    fig.suptitle(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_eb_blob_movie(*args, output_path, frame_stride=2, title="Shifted-torus EB blob state evolution", z_indices=None):
    """Render saved states as a lightweight polar movie; no RHS evaluation occurs."""
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    times, density, phi, te, ti, vi, ve, vorticity, geometry = args
    del phi, te, ti, vi, ve, vorticity
    values = np.asarray(density)
    x = np.asarray(geometry.grid.x.centers)
    y = np.asarray(geometry.grid.y.centers)
    z = np.asarray(geometry.grid.z.centers)
    z_indices = z_indices or tuple(np.linspace(0, z.size - 1, 4, dtype=int))
    yg, rg = np.meshgrid(y, x)
    fig, axes = plt.subplots(1, len(z_indices), subplot_kw={"projection": "polar"}, figsize=(16, 4))
    axes = np.atleast_1d(axes)
    finite = values[np.isfinite(values)]
    image = axes[0].pcolormesh(yg, rg, values[0, :, :, z_indices[0]], shading="auto", cmap="magma", vmin=finite.min(), vmax=finite.max())
    for axis, index in zip(axes, z_indices):
        axis.set_ylim(0, x[-1]); axis.set_title(f"z={z[index]:.3f}")
    def update(frame):
        for axis, index in zip(axes, z_indices):
            axis.clear(); axis.set_ylim(0, x[-1]); axis.pcolormesh(yg, rg, values[frame, :, :, index], shading="auto", cmap="magma", vmin=finite.min(), vmax=finite.max())
        fig.suptitle(f"{title}, t={float(times[frame]):.3e}"); return []
    animator = animation.FuncAnimation(fig, update, frames=range(0, len(times), max(1, int(frame_stride))), interval=100)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    animator.save(output_path, writer=animation.PillowWriter(fps=10)); plt.close(fig)


__all__ = [name for name in globals() if not name.startswith("__")]
