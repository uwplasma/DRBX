"""Memory-bounded helpers of the M5b campaign (the windowed in-plane assembly now lives in the package).

``build_dirichlet_preconditioner`` of the package assembles the in-plane matrix a group of planes at a time
(``drbx.geometry.sbp_laplacian_assembly.iter_inplane_blocks``, bitwise the blocks of the full assembly). What remains here is the
campaign's merge of the *previous* layout (the P07 banded LDU with the dense core as the first super-ring) group by group into ONE
stock ``PlanePreconditioner`` (:func:`build_merged_preconditioner`, also used for the Neumann + shift blocks of the matrix-free
audit and the re-freeze), kept as the "old" arm of the preconditioner benchmark.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def group_inplane(plan, k0: int, k1: int, kind: str = "dirichlet", shared=None):
    """In-plane block-diagonal CSR ``((k1 - k0) P, (k1 - k0) P)`` of the planes ``k0 .. k1 - 1`` (windowed assembly)."""
    from drbx.geometry.sbp_laplacian_assembly import LaplacianAssembly

    return LaplacianAssembly(plan, window=(k0, k1), shared=shared).matrix(kind, inplane=True).tocsr()


def group_keys(plan, n_planes: int):
    """``(ring, plane, theta)`` keys of ``n_planes`` planes (as ``plane_keys`` with ``n_eta = n_planes``)."""
    import dataclasses

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
    from drbx.geometry.sbp_laplacian_assembly import global_assembly_data

    shared = global_assembly_data(plan)
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
        M = group_inplane(plan, k0, k1, kind, shared)
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
