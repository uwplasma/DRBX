"""Build the all-eta products for the scheme-locked SBP operators in blocks of in-plane columns."""

from dataclasses import replace

import numpy as np

from ...operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout, build_noncubic_zernike_core_ring_layout
from ...operators.nodal.nodal_shape import NodalShape
from ...operators.nodal.sbp_laplacian_plan import (
    LaplacianMetric, _check_layout, _core_penalty_peak, _eta_face_values,
    _penalty_from_peaks, _radial_face_values, build_laplacian_plan, core_frame_A,
    fourier_half_interp, radial_face_interp, stag_eta, staggered_pair,
)
from ...operators.nodal.sbp_laplacian_assembly import _staggered_eta_diagonal, iter_inplane_blocks
from ...operators.perp.plane_preconditioner import _choose_rings_per_block


NODE_BLOCK = 64


def _layout(identity):
    N, E = identity["n"], identity["n_eta"]
    if "shape" not in identity:
        return build_zernike_core_ring_layout(N, n_eta=E)
    shape = NodalShape.from_dict(identity["shape"])
    if (shape.n_r, shape.n_eta) != (N, E):
        raise ValueError("Owner SBP shape differs from the saved counts")
    options = {} if shape.version == 1 else dict(
        version=2, field_periods=shape.field_periods,
        replication_period_count=shape.replication_period_count,
    )
    return build_noncubic_zernike_core_ring_layout(N, shape.n_theta, E, **options)


def _map(directory, name, mode="r"):
    return np.load(directory / f"{name}.npy", allow_pickle=False, mmap_mode=mode)


def _column_blocks(core, ring):
    for start in range(0, core.n_nodes, NODE_BLOCK):
        yield slice(start, min(start + NODE_BLOCK, core.n_nodes)), 0
    # The radial contraction keeps every radial term and its order in each
    # theta column. Group whole columns when walking the ring block.
    nodes = np.arange(core.n_nodes, core.n_nodes + ring.m*ring.N).reshape(ring.m, ring.N)
    step = max(1, NODE_BLOCK // ring.m)
    for start in range(0, ring.N, step):
        tile = nodes[:, start:start + step]
        yield tile.ravel(), tile.shape[1]


def _write_columns(directory, name, columns, data):
    target = _map(directory, name, mode="r+")
    target[:, columns] = data
    target.flush()


def _rings_per_block(identity, directory, layout, tau, tau_w, halo):
    info = identity.get("preconditioner", {}).get("info") or {}
    chosen = info.get("rings_per_block", info.get("params", {}).get("rings_per_block"))
    if chosen is not None:
        return chosen
    # Old identities may lack the Core-Schur layout choice. Its default rule
    # reads only the first plane's assembled matrix.
    metric = LaplacianMetric(*(_map(directory, name)[:1] for name in ("A", "jac", "ginv_u")))
    plan = build_laplacian_plan(replace(layout, n_eta=1), metric)
    aee = _map(directory, "aee_h")[halo:halo + 1]
    ge = _map(directory, "ge_diag")[:1]
    plan = replace(plan, aee_h=aee, tau=np.asarray(tau), tau_w=np.asarray(tau_w))
    shared = dict(A=plan.A, auu=plan.auu_f, att=plan.att_h, aee=aee, Ge_diag=ge,
                  c_core=1., c_wall=1., s_in=1., s_wall=1.)
    _, _, matrix = next(iter_inplane_blocks(plan, group_planes=1, shared=shared))
    return _choose_rings_per_block(matrix, plan.structure.Nc, plan.structure.m, plan.structure.N)


def _write_sbp_setup(identity, directories, edges):
    """Stream the full-eta contractions into owner slabs and reduce penalties."""
    layout = _layout(identity)
    E, P = layout.n_eta, layout.P
    if P != identity["P"]:
        raise ValueError("Owner SBP layout differs from the saved node count")
    core, ring = _check_layout(layout)
    halo = 0
    slabs = [(directory, begin, end, np.arange(begin - halo, end + halo) % E)
             for directory, begin, end in zip(directories, edges[:-1], edges[1:], strict=True)]
    for directory, begin, end, _ in slabs:
        for name, shape in (("aee_h", (end - begin + 2 * halo, P)), ("ge_diag", (end - begin, P))):
            np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float64, shape=shape)
    Ie = fourier_half_interp(E, 0.)
    Ge2T = (stag_eta(E, layout.deta) ** 2).T
    W = np.asarray(layout.wxy) * layout.deta
    Iu = radial_face_interp(ring.m)[[0, -1]]
    core_peak = 0.
    radial_peaks = np.zeros(2)
    for cols, ring_width in _column_blocks(core, ring):
        width = len(cols) if ring_width else cols.stop - cols.start
        Aee = np.empty((E, width), dtype=np.float64)
        for directory, begin, end, _ in slabs:
            A = _map(directory, "A")[:, cols]
            if ring_width:
                faces = _radial_face_values(A[..., 0, 0].reshape(end-begin, ring.m, ring_width), Iu)
                radial_peaks = np.maximum(radial_peaks, np.abs(faces).max(axis=(0, 2)))
            else:
                J = _map(directory, "jac")[:, cols]
                A, _ = core_frame_A(A, J, core.u[cols], core.theta[cols])
                core_peak = np.maximum(core_peak, _core_penalty_peak(A))
            Aee[begin:end] = A[..., 2, 2]
        aee_h = _eta_face_values(Aee, Ie)
        ge_diag = _staggered_eta_diagonal(Ge2T, W[cols], aee_h)
        for directory, begin, end, halo_ids in slabs:
            _write_columns(directory, "aee_h", cols, aee_h[halo_ids])
            _write_columns(directory, "ge_diag", cols, ge_diag[begin:end])
    ar, aw = radial_peaks
    wf0 = float(staggered_pair(ring.m).wf[0])
    tau, tau_w, _ = _penalty_from_peaks(ar, aw, core_peak, wf0, layout.du, core.p, core.R_c)
    rings = _rings_per_block(identity, directories[0], layout, tau, tau_w, halo)
    for directory in directories:
        for name, value, dtype in (("tau", tau, np.float64), ("tau_w", tau_w, np.float64),
                                   ("rings_per_block", rings, np.int64)):
            with (directory / f"{name}.npy").open("xb") as stream:
                np.save(stream, np.asarray(value, dtype=dtype), allow_pickle=False)
    return {"aee_h": ((E, P), np.dtype(np.float64), halo),
            "ge_diag": ((E, P), np.dtype(np.float64), 0),
            "tau": ((), np.dtype(np.float64), 0), "tau_w": ((), np.dtype(np.float64), 0),
            "rings_per_block": ((), np.dtype(np.int64), 0)}
