"""Memory-bounded helpers of the M5b campaign built on the M5a package (nothing here changes the package).

The stock ``build_dirichlet_preconditioner`` assembles the in-plane matrix of *all* ``E`` planes on the host and factorises it
into one ``(S, E, 2w + 1, B, B)`` array; at N48/N64 the host transients exceed the campaign's 4 GB budget. The plane blocks are
independent, so the same preconditioner can be built group by group (the in-plane diagonal block of plane ``k`` only needs
the plan data of plane ``k`` and, through the ``eta eta`` diagonal, the half-plane coefficients ``aee_h`` of the planes
``k - 2 .. k + 1``): :func:`window_plan` slices the plan to a window with 3 guard planes on each side, :func:`group_inplane`
assembles its in-plane matrix with the package's ``LaplacianAssembly`` and keeps the centre planes, and
:func:`build_merged_preconditioner` factorises every group with the stock ``build_plane_preconditioner`` and merges the groups
into ONE stock ``PlanePreconditioner`` (identical arrays to ``build_dirichlet_preconditioner``; the solve is the package's).
"""
from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

GUARD = 3


def window_plan(plan, k0: int, k1: int):
    """The plan restricted to planes ``k0 - GUARD .. k1 + GUARD - 1`` (periodic indices): ``(window plan, offset of k0)``."""
    st = plan.structure
    E = st.n_eta
    idx = (np.arange(k0 - GUARD, k1 + GUARD)) % E
    kw = {}
    for f in dataclasses.fields(plan):
        v = getattr(plan, f.name)
        if f.name in ("Hp", "A", "auu_f", "att_h", "aee_h", "kappa", "wall_alpha", "wall_beta_th", "wall_beta_eta") and v is not None:
            kw[f.name] = np.asarray(v)[idx]
    return dataclasses.replace(plan, structure=dataclasses.replace(st, n_eta=len(idx)), **kw), GUARD


def group_inplane(plan, k0: int, k1: int, kind: str = "dirichlet"):
    """In-plane block-diagonal CSR ``((k1 - k0) P, (k1 - k0) P)`` of the planes ``k0 .. k1 - 1`` (windowed assembly)."""
    from drbx.validation.sbp_laplacian_audit import LaplacianAssembly

    wp, off = window_plan(plan, k0, k1)
    asm = LaplacianAssembly(wp)
    M = asm.matrix(kind, inplane=True).tocsr()
    P = plan.structure.P
    lo, hi = off * P, (off + (k1 - k0)) * P
    return M[lo:hi, lo:hi].tocsr()


def group_keys(plan, n_planes: int):
    """``(ring, plane, theta)`` keys of ``n_planes`` planes (as ``plane_keys`` with ``n_eta = n_planes``)."""
    from drbx.geometry.sbp_laplacian import plane_keys

    st = dataclasses.replace(plan.structure, n_eta=n_planes)
    return plane_keys(st)


def build_merged_preconditioner(plan, groups: int = 1, factor_dtype: str = "float64", kind: str = "dirichlet",
                                shift: float = 0.0, log=None, cond_max: float = 1e12):
    """One stock ``PlanePreconditioner`` of the (Dirichlet, or Neumann + ``shift``) in-plane blocks, assembled and factorised
    per group of planes; ``info`` carries the per-group timings and the host peak sizes."""
    from drbx.native.fci_perpendicular_plane_preconditioner import (PlanePreconditioner, _Meta, build_plane_preconditioner)

    st = plan.structure
    E, P = st.n_eta, st.P
    bounds = np.linspace(0, E, groups + 1).astype(int)
    infos, t0 = [], time.perf_counter()
    big = {}                                           # merged lo / up / dinv, filled group by group (donated in-place updates)
    idx_parts, pos_parts = [], []
    upd = jax.jit(lambda arr, part, k0: jax.lax.dynamic_update_slice(arr, part, (0, k0) + (0,) * (arr.ndim - 2)),
                  donate_argnums=0)
    meta0 = None
    off = 0
    for g in range(groups):
        k0, k1 = int(bounds[g]), int(bounds[g + 1])
        t1 = time.perf_counter()
        M = group_inplane(plan, k0, k1, kind)
        if shift:
            M = M + shift * sp.identity(M.shape[0], format="csr")
        t_asm = time.perf_counter() - t1
        ring, pl, th = group_keys(plan, k1 - k0)
        p = build_plane_preconditioner(M, ring, pl, th, factor_dtype=factor_dtype, cond_max=cond_max)
        infos.append(dict(planes=k1 - k0, nnz=int(M.nnz), assemble_seconds=t_asm, **{k: p.info[k] for k in
                          ("S", "B", "w", "storage_bytes", "setup_seconds", "max_block_cond1", "padded_fraction")}))
        del M
        meta0 = meta0 or p.meta
        gi = np.asarray(p.idx)
        idx_parts.append(np.where(gi == p.meta.n_owners, -1, gi + off * P))
        s_, rem = np.divmod(np.asarray(p.pos).astype(np.int64), p.meta.P * p.meta.B)
        pl_, sl_ = np.divmod(rem, p.meta.B)
        pos_parts.append(((s_ * E + (pl_ + off)) * p.meta.B + sl_).astype(np.int32))
        for name in ("lo", "up", "dinv"):
            part = getattr(p, name)
            if name not in big:
                shape = list(part.shape)
                shape[1] = E
                big[name] = jnp.zeros(tuple(shape), part.dtype)
            big[name] = upd(big[name], part, k0)
        off += k1 - k0
        del p, part
        if log:
            log(f"  prec group {g + 1}/{groups}: planes {k0}..{k1 - 1} nnz {infos[-1]['nnz']:.3e} assemble {t_asm:.1f}s "
                f"factor {infos[-1]['setup_seconds']:.1f}s S={infos[-1]['S']} B={infos[-1]['B']} w={infos[-1]['w']}")
    n_tot = E * P
    idx = np.concatenate(idx_parts, axis=1)
    idx = np.where(idx < 0, n_tot, idx).astype(np.int32)
    merged = PlanePreconditioner(big["lo"], big["up"], big["dinv"], jnp.asarray(idx), jnp.asarray(np.concatenate(pos_parts)),
                                 _Meta(S=meta0.S, B=meta0.B, w=meta0.w, P=E, n_owners=n_tot, dtype=meta0.dtype))
    merged.info.update(groups=groups, setup_seconds=time.perf_counter() - t0, storage_bytes=merged.nbytes,
                       S=merged.meta.S, B=merged.meta.B, w=merged.meta.w, factor_dtype=factor_dtype,
                       max_block_cond1=max(i["max_block_cond1"] for i in infos), per_group=infos,
                       padded_fraction=infos[0]["padded_fraction"])
    return merged
