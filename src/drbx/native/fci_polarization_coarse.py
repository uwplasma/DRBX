"""Small factorized additive coarse correction for physical owner grids."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


def _global_sum(value, domain):
    if domain is None:
        return jnp.sum(value)
    out = jnp.sum(value)
    for axis in getattr(domain, "mesh_axis_names", ()):
        if axis is not None:
            out = jax.lax.psum(out, axis)
    return out


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoarseData:
    radial: jnp.ndarray
    fourier: jnp.ndarray
    means: jnp.ndarray
    mass_sum: jnp.ndarray
    normalizer: jnp.ndarray
    H_lu: jnp.ndarray
    H_pivots: jnp.ndarray
    global_shape: tuple

    def tree_flatten(self):
        return (self.radial, self.fourier, self.means, self.mass_sum,
                self.normalizer, self.H_lu, self.H_pivots), self.global_shape

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, global_shape=aux)

    @property
    def rank(self):
        return self.normalizer.shape[0]

    def _local_fourier(self, active, domain):
        nz = active.shape[2]; gz = self.global_shape[2]
        if active.shape[0] != self.global_shape[0] or active.shape[1] != self.global_shape[1]:
            raise ValueError("coarse data requires unsplit radial/theta dimensions")
        if nz == gz:
            return self.fourier
        if gz % nz or domain is None:
            raise ValueError("local eta dimension must divide global eta dimension")
        axes = getattr(domain, "mesh_axis_names", ())
        axis = axes[2] if len(axes) > 2 else None
        if axis is None:
            raise ValueError("eta sharding requires a named eta mesh axis")
        start = jnp.asarray(jax.lax.axis_index(axis) * int(nz), dtype=jnp.int32)
        return jax.lax.dynamic_slice(self.fourier, (start, jnp.asarray(0, dtype=jnp.int32)),
                                     (int(nz), int(self.fourier.shape[1])))

    def _project(self, x, active, weights, domain=None):
        w = jnp.where(active, weights, 0.0)
        x = jnp.where(active, x, 0.0)
        return x - jnp.where(active, 1.0, 0.0) * _global_sum(w*x, domain) / self.mass_sum

    def q_apply(self, coeff, active, weights, domain=None):
        """Apply mass-orthonormal coarse Q to physical owner coefficients."""
        coeff = jnp.asarray(coeff).reshape(-1)
        nx, ny, nz = active.shape
        fourier = self._local_fourier(active, domain)
        coeff = self.normalizer @ coeff
        gz = self.global_shape[2]
        cc = jnp.zeros((self.radial.shape[1], gz), dtype=coeff.dtype).ravel().at[1:].set(coeff).reshape(self.radial.shape[1], gz)
        field = self.radial @ cc @ fourier.T
        out = jnp.broadcast_to(field[:, None, :], active.shape)
        out = out - (self.means @ coeff) / self.mass_sum
        return jnp.where(active, out, 0.0)

    def q_transpose(self, x, active, weights, domain=None):
        """Apply the physical mass-weighted transpose Q.T."""
        x = self._project(x, active, weights, domain)
        w = jnp.where(active, weights, 0.0)
        z = jnp.sum(w * x, axis=1)
        fourier = self._local_fourier(active, domain)
        c = self.radial.T @ z @ fourier
        if domain is not None:
            for axis in getattr(domain, "mesh_axis_names", ()):
                if axis is not None:
                    c = jax.lax.psum(c, axis)
        return self.normalizer.T @ c.ravel()[1:]

    def coarse(self, x, active, weights, domain=None):
        """Apply the exact Galerkin coarse inverse Q H^-1 Q.T M."""
        y = self.q_transpose(x, active, weights, domain)
        c = jax.scipy.linalg.lu_solve((self.H_lu, self.H_pivots), y)
        return self.q_apply(c, active, weights, domain)


def build_coarse_data(apply_A, active, weights, global_shape=None, *, radial_modes=8):
    """Build mass-orthonormal Q factors and factor the full coarse operator."""
    active = jnp.asarray(active, dtype=bool)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    shape = tuple(active.shape if global_shape is None else global_shape)
    if active.shape != shape or weights.shape != shape:
        raise ValueError("active and weights must match global_shape")
    if not np.all(np.isfinite(np.asarray(weights)[np.asarray(active)])) or np.any(np.asarray(weights)[np.asarray(active)] <= 0):
        raise ValueError("weights must be finite and positive on active owners")
    nx, _, nz = shape
    radial_modes = min(int(radial_modes), nx)
    if radial_modes <= 0 or nz % 2:
        raise ValueError("coarse space requires a positive radial rank and even eta dimension")
    radial = jnp.cos(jnp.pi * (jnp.arange(nx)[:, None] + .5) * jnp.arange(radial_modes)[None, :] / nx)
    modes = [jnp.ones(nz)] + [jnp.cos(2*jnp.pi*m*jnp.arange(nz)/nz) for m in range(1,nz//2+1)] + [jnp.sin(2*jnp.pi*m*jnp.arange(nz)/nz) for m in range(1,nz//2)]
    fourier = jnp.stack(modes, axis=1)
    g = jnp.einsum("xr,ze->xzre", radial, fourier).reshape(nx*nz, radial_modes*nz)[:, 1:]
    sxz = jnp.sum(jnp.where(active, weights, 0.0), axis=1).reshape(-1)
    mass_sum = jnp.sum(sxz); means = g.T @ sxz
    gram = g.T @ (sxz[:, None] * g) - jnp.outer(means, means) / mass_sum
    cholg = jnp.linalg.cholesky((gram + gram.T) * .5)
    if not np.all(np.isfinite(np.asarray(gram))) or not np.all(np.isfinite(np.asarray(cholg))):
        raise ValueError("coarse Gram matrix is non-finite or not positive definite")
    normalizer = jnp.linalg.solve(cholg, jnp.eye(gram.shape[0])).T
    rank = gram.shape[0]
    data = CoarseData(
        radial, fourier, means, mass_sum, normalizer,
        jnp.eye(rank, dtype=jnp.float64),
        jnp.arange(rank, dtype=jnp.int32), shape,
    )
    def body(j, out):
        q = data.q_apply(jnp.eye(data.rank, dtype=jnp.float64)[:, j], active, weights)
        hcol = data.q_transpose(apply_A(q), active, weights)
        return out.at[:, j].set(hcol)
    H = jax.lax.fori_loop(0, data.rank, body, jnp.zeros((data.rank, data.rank), dtype=jnp.float64))
    H = jax.block_until_ready(H)
    if not np.all(np.isfinite(np.asarray(H))):
        raise ValueError("coarse H is non-finite")
    h_lu, h_pivots = jax.scipy.linalg.lu_factor(H)
    if (not np.all(np.isfinite(np.asarray(h_lu)))
            or np.any(np.asarray(jnp.diag(h_lu)) == 0.0)):
        raise ValueError("coarse H LU factorization is singular or non-finite")
    return CoarseData(radial, fourier, means, mass_sum, normalizer, h_lu, h_pivots, shape)
