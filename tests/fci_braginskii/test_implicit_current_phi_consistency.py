"""Numerical consistency of the implicit current/phi stage reconstruction.

``LocalFciDrbEBRhs.solve_implicit_current_phi_pair`` returns

    Ve*    = Ve_b + h*mu*P(G(E phi))
    omega* = omega_pred - h**2*mu*P(kappa*D(n*G(E phi)))

with ``omega_pred = omega_b + h*P(kappa*D(n*(Vi - E Ve_b)))``.  Pushing the
returned ``Ve*`` back through the same current-divergence path,

    omega_rec = omega_b + h*P(kappa*D(n*(Vi - E Ve*))),

differs from ``omega*`` by ``h**2*mu*P(kappa*D(n*(E P - I) G E phi))``.

E and P below are the production methods
(``_expand_owner_field_for_stencil`` / ``_project_fine_cell_term``, i.e.
``expand_local_control_volume_owner_field`` and
``aggregate_local_control_volume_average`` + masks), called on a stub that
carries a real ``LocalControlVolumeCellGeometry3D`` and ``LocalDomain3D``.
G and D are a synthetic nonuniform-|B| parallel gradient and its
fine-volume-weighted negative adjoint, standing in for the production
``_fci_current_phi_boundary_pair`` (which needs a full traced FCI geometry);
the identity above does not depend on the particular G/D.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drbx.fci_braginskii.geometry.fci_geometry import (
    HaloLayout3D,
    LocalControlVolumeCellGeometry3D,
    LocalDomain3D,
    ShardSpec3D,
)
from drbx.fci_braginskii.native import fci_drb_EB_rhs
from drbx.fci_braginskii.native.fci_drb_EB_rhs import LocalFciDrbEBRhs

jax.config.update("jax_enable_x64", True)

SHAPE = (3, 8, 4)  # (radial, angular, parallel)


def _domain(layout: HaloLayout3D) -> LocalDomain3D:
    spec = ShardSpec3D(
        global_shape=SHAPE,
        owned_start=(0, 0, 0),
        owned_stop=SHAPE,
        shard_index=(0, 0, 0),
        shard_counts=(1, 1, 1),
        periodic_axes=(False, True, True),
        halo_width=1,
    )
    return LocalDomain3D(layout=layout, shard_spec=spec, mesh_axis_names=(None, None, None))


def _cells(layout: HaloLayout3D, volume: jnp.ndarray, agglomerate: bool):
    centroid = jnp.zeros(SHAPE + (3,), dtype=jnp.float64)
    base = LocalControlVolumeCellGeometry3D.identity(layout, volume=volume, centroid=centroid)
    if not agglomerate:
        return base
    # Owner/RLP angular agglomeration on the innermost radial ring: angular
    # pairs (2m, 2m+1) merge onto owner 2m, as the polar angular
    # agglomeration does near the axis.
    owner_j = np.array(base.owner_j)
    merged = np.zeros(SHAPE, dtype=bool)
    received = np.zeros(SHAPE, dtype=np.int32)
    agg_volume = np.array(volume)
    members = np.ones(SHAPE, dtype=np.int32)
    for j in range(1, SHAPE[1], 2):
        owner_j[0, j, :] = j - 1
        merged[0, j, :] = True
        received[0, j - 1, :] = 1
        members[0, j - 1, :] = 2
        agg_volume[0, j - 1, :] += agg_volume[0, j, :]
    agg_volume[merged] = 0.0
    members[merged] = 0
    return LocalControlVolumeCellGeometry3D(**{
        **base.__dict__,
        "owner_j": jnp.asarray(owner_j),
        "is_merged_source": jnp.asarray(merged),
        "is_active_owner": jnp.asarray(~merged),
        "is_aggregate_target": jnp.asarray(received > 0),
        "received_source_count": jnp.asarray(received),
        "member_count": jnp.asarray(members),
        "aggregate_volume": jnp.asarray(agg_volume),
    })


def _stage(agglomerate: bool, monkeypatch):
    rng = np.random.default_rng(1234)
    layout = HaloLayout3D(SHAPE, 1)
    volume = jnp.asarray(0.5 + rng.random(SHAPE))
    cells = _cells(layout, volume, agglomerate)
    active = jnp.ones(SHAPE, dtype=bool)
    stub = SimpleNamespace(
        control_volume_geometry=SimpleNamespace(cells=cells) if agglomerate else None,
        domain=_domain(layout),
        halo_exchange=None,
        geometry=SimpleNamespace(active_cell_mask_owned=active),
    )
    stub._uses_projected_fine_grid = agglomerate
    stub._owner_field = lambda v: LocalFciDrbEBRhs._owner_field(stub, v)
    stub._restrict_fine_field = lambda v: LocalFciDrbEBRhs._restrict_fine_field(stub, v)
    monkeypatch.setattr(
        fci_drb_EB_rhs, "_mask_inactive_owned",
        lambda values, geometry: jnp.where(geometry.active_cell_mask_owned, values, 0.0),
    )
    E = lambda v: LocalFciDrbEBRhs._expand_owner_field_for_stencil(stub, v)  # noqa: E731
    P = lambda v: LocalFciDrbEBRhs._project_fine_cell_term(stub, v)  # noqa: E731
    stub._expand_owner_field_for_stencil = E
    stub._project_fine_cell_term = P
    # The production owner-level gradient used by both the solve and Ve*.
    owner_gradient = lambda phi: LocalFciDrbEBRhs._owner_current_gradient(stub, phi, G)  # noqa: E731

    # Nonuniform |B| and density; G = (1/|B|) forward difference along a
    # twisted field line (k+1, j+1), as FCI maps cross the angular grid,
    # D = -W^{-1} G^T W with fine volumes W (the weighted negative adjoint).
    ii, jj, kk = np.meshgrid(*(np.arange(s) for s in SHAPE), indexing="ij")
    bmag = jnp.asarray(1.0 + 0.3 * np.cos(2 * np.pi * kk / SHAPE[2]) + 0.1 * ii + 0.05 * jj)
    n_dense = jnp.asarray(1.0 + 0.5 * np.exp(-((ii - 1.0) ** 2 + (jj - 3.5) ** 2 / 4.0)))
    N = int(np.prod(SHAPE))
    Gm = np.zeros((N, N))
    idx = np.arange(N).reshape(SHAPE)
    for a in range(N):
        i, j, k = np.unravel_index(a, SHAPE)
        Gm[a, idx[i, (j + 1) % SHAPE[1], (k + 1) % SHAPE[2]]] += 1.0 / float(bmag[i, j, k])
        Gm[a, a] -= 1.0 / float(bmag[i, j, k])
    w = np.asarray(volume).ravel()
    Dm = -(Gm.T * w[None, :]) / w[:, None]
    G = lambda v: jnp.asarray(Gm @ np.asarray(v).ravel()).reshape(SHAPE)  # noqa: E731
    D = lambda v: jnp.asarray(Dm @ np.asarray(v).ravel()).reshape(SHAPE)  # noqa: E731

    owner = np.asarray(cells.is_active_owner)
    h, mu, rho_star, kappa = 0.05, 1836.0, 0.3, bmag**2 / n_dense
    omega_b = stub._owner_field(jnp.asarray(rng.standard_normal(SHAPE)))
    Ve_b = stub._owner_field(jnp.asarray(rng.standard_normal(SHAPE)))
    Vi_dense = E(stub._owner_field(jnp.asarray(rng.standard_normal(SHAPE))))
    Ve_dense = E(Ve_b)  # state_halo.Ve[owned]

    omega_pred = omega_b + h * P(kappa * D(n_dense * (Vi_dense - Ve_dense)))
    coeff = h**2 * mu / rho_star**2

    def extra(phi):
        return -coeff * P(kappa * D(n_dense * owner_gradient(phi)))

    # Owner-space dense solve of [-L + extra] phi = rhs; L is a synthetic
    # negative-definite owner operator, alias rows pinned to identity.
    L = -np.diag(2.0 + w)
    A = np.zeros((N, N))
    for a in range(N):
        e = np.zeros(N); e[a] = 1.0
        col = -L @ e + np.asarray(extra(jnp.asarray(e.reshape(SHAPE)))).ravel()
        A[:, a] = col
    alias = ~owner.ravel()
    A[alias, :] = 0.0
    A[:, alias] = 0.0
    A[alias, alias] = 1.0
    rhs = np.array(-omega_pred / rho_star**2).ravel()
    rhs[alias] = 0.0
    phi = stub._owner_field(jnp.asarray(np.linalg.solve(A, rhs).reshape(SHAPE)))

    g = owner_gradient(phi)
    Ve_star = Ve_b + h * mu * P(g)
    omega_star = omega_pred - h**2 * mu * P(kappa * D(n_dense * g))

    omega_rec = omega_b + h * P(kappa * D(n_dense * (Vi_dense - E(Ve_star))))
    mismatch = float(jnp.linalg.norm(omega_rec - omega_star))
    coupled = float(jnp.linalg.norm(omega_star - omega_pred))
    scale = float(jnp.linalg.norm(omega_star - omega_b))
    print(f"\n  |rec-*|/|*-b| = {mismatch / scale:.3e}  |rec-*|/|coupled| = {mismatch / coupled:.3e}")
    return mismatch / scale


def test_reconstruction_matches_without_agglomeration(monkeypatch):
    rel = _stage(False, monkeypatch)
    print(f"no agglomeration: rel mismatch = {rel:.3e}")
    assert rel < 1.0e-12


def test_reconstruction_matches_with_owner_agglomeration(monkeypatch):
    rel = _stage(True, monkeypatch)
    print(f"owner agglomeration: rel mismatch = {rel:.3e}")
    assert rel < 1.0e-12
