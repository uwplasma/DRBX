"""Produce the case's wall and limiter files using only owner build inputs."""

from contextlib import closing
from dataclasses import replace

import numpy as np
from scipy import sparse

from ._owner_inputs import OwnerParallelInputs, field_manifest, validate_inputs
from ._receipts import write_json


# Locked coefficients used by the coupled loader and gathered wall producer.
CHI = np.array([.01, 400., 11., 5.6, 180., .06])
D = np.array([1., 1.2, 1.4, 0., 0., .8]) * 1e-5


def _canonical_columns(matrix, planes, width, total):
    matrix = matrix.tocsr()
    indices = planes[matrix.indices // width] * width + matrix.indices % width
    return sparse.csr_matrix((matrix.data, indices, matrix.indptr),
                             shape=(matrix.shape[0], total*width))


def _wall_maps(p, owner, layout, sbp):
    from ..drb.owner_load import sbp as owner_sbp
    from ..operators.nodal.sbp_laplacian_plan import LaplacianMetric
    from ..operators.nodal.sbp_laplacian_assembly import LaplacianAssembly

    E, P = layout.n_eta, layout.P
    a, b = p.edges[owner:owner+2]
    width = owner_sbp.F_HALO
    metric_width = owner_sbp.PLAN_HALO
    extended = p.read(owner, halo=width, leaves=("A", "jac", "ginv_u"))
    planes = np.arange(a-width, b+width) % E
    aee = np.zeros((len(planes), P), dtype=np.float64)
    aee[width-metric_width:len(planes)-(width-metric_width)] = sbp.aee_h
    if len(planes) >= E:
        # Use canonical order on a fully covered ring, including its sparse
        # accumulation order. Only the owned rows are retained.
        planes, indices = np.unique(planes, return_index=True)
        extended = {key: v[indices] for key, v in extended.items()}
        wanted = np.arange(a-metric_width,b+metric_width) % E
        aee = np.zeros((E,P), dtype=np.float64)
        aee[wanted] = sbp.aee_h
        owned = np.arange(a,b)
    else:
        owned = np.arange(width, width+b-a)
    lap = owner_sbp._laplacian_window(
        replace(layout, n_eta=len(planes)),
        LaplacianMetric(extended["A"], extended["jac"], extended["ginv_u"]),
        aee, sbp.tau, sbp.tau_w,
    )
    # The full Neumann matrix uses the face metric directly. Ge_diag is only
    # used by assembly of in-plane factors.
    shared = dict(A=lap.A, auu=lap.auu_f, att=lap.att_h, aee=lap.aee_h,
                  c_core=1., c_wall=1., s_in=1., s_wall=1., Ge_diag=None)
    asm = LaplacianAssembly(lap, shared=shared)
    N = asm.st.N
    nodes = (owned[:,None]*P+np.arange(P)).ravel()
    walls = (owned[:,None]*N+np.arange(N)).ravel()
    T = asm.Tw[walls][:, nodes].tocsr()
    F = _canonical_columns(asm.flux_ops()[2][walls], planes, P, E)
    K = _canonical_columns(asm.matrix("neumann", mode="conormal")[nodes], planes, P, E)
    return T, F, K, float(asm.Om[0])


def _wall_rows(p, owner, layout, wall, rr, magnetic, mu):
    import jax.numpy as jnp

    from ..drb.wall_traction import admittance, hit_map
    from ..operators.perp.sbp_curvature import curvature_flux
    from ..operators.perp.sbp_ops import flux_trace, trace
    from ..operators.nodal.nodal_plan import build_nodal_plan

    sbp = p.read_owner_sbp(owner, edges=p.edges)
    T,F,K,Om = _wall_maps(p, owner, layout, sbp)
    E, P = layout.n_eta, layout.P
    a, b = p.edges[owner:owner+2]
    plan = build_nodal_plan(replace(layout, n_eta=b-a), sbp.nodal)
    blk,side,sgn,N = plan.structure.walls[0]
    tc = np.asarray(trace(plan, blk, side, jnp.broadcast_to(jnp.eye(P)[None], (1,P,P))))[0]
    Tc = sparse.block_diag([sparse.csr_matrix(tc)]*(b-a), format="csr")
    cf = curvature_flux(plan)
    cf = np.asarray(flux_trace(plan, blk, side, cf[...,0], cf[...,1])).ravel()*sgn
    H = (np.asarray(plan.Hp)*layout.deta).ravel()
    ref = p.read(owner, leaves=("wall_J", "wall_ginv_u"))
    root = np.sqrt(ref["wall_ginv_u"][...,0]).ravel()
    sg = wall[:,7]
    Y = np.empty((len(sg),2,2))
    for sign in (-1,1):
        Y[sg == sign] = admittance(sign,0.)[0]
    wm = np.array([2.,1.5,1.5])
    A = np.array([[4,2,2],[8/3,14/3,4/3],[8/3,4/3,-2.]])
    ev,V = np.linalg.eigh(np.sqrt(wm)[:,None]*A/np.sqrt(wm)[None,:])
    V = V/np.sqrt(wm)[:,None]
    Ap = np.einsum("ij,hj,kj,k->hik", V, np.maximum(cf[:,None]*ev,0), V, wm)
    Ap *= np.array([True,False,True])[None,:,None]
    rows = dict(H=H, Js=ref["wall_J"].ravel()*root, curvature_flux=cf,
                hw=T@sbp.nodal.h.reshape(-1,3), bn=(T@magnetic["b"][:,0])/root,
                trace_gram=np.asarray(T.multiply(T).sum(1)).ravel(),
                mass_trace_gram=np.asarray(T.multiply(T)@(1/H)).ravel(),
                Y=Y, Khit=hit_map(sg,wall[:,8],0.,CHI[:5]), A0plus=Ap, wall_foot_to_hit_area_ratio=rr)

    def canonical(A, start, cols):
        A = A.tocsr()
        return sparse.csr_matrix((A.data, A.indices+start, A.indptr),shape=(A.shape[0],cols))
    matrices = dict(T=canonical(T,a*P,E*P), TT=canonical(T.T,a*N,E*N),
                    Tc=canonical(Tc,a*P,E*P), TcT=canonical(Tc.T,a*N,E*N), F=F, K=K)
    for key, A in matrices.items():
        rows.update({key+"_data": A.data, key+"_indices": A.indices,
                     key+"_indptr": A.indptr, key+"_shape": A.shape})
    Bdiv = np.zeros((5,5))
    Bdiv[0,4]=-2;Bdiv[1,3]=.71;Bdiv[1,4]=-1.71;Bdiv[2,4]=-1
    return rows, dict(Om=Om, Bdiv=Bdiv, Wm=np.array([2.,1.5,1.5,1.,1/mu]), gi=1.)


def _write_limiter(root, q, ranges):
    edge_paths = []
    prefix = 0
    for owner, (a,b) in enumerate(ranges):
        k = q.K(owner, node_ids=np.arange(a,b))
        k.eliminate_zeros()
        k.sort_indices()
        row = np.repeat(np.arange(a,b,dtype=np.int32),np.diff(k.indptr))
        col = k.indices.astype(np.int32)
        good = (col>row)&(k.data>0)
        i,j,v = row[good], col[good], k.data[good]
        diag = k[:,a:b].diagonal().copy()
        np.add.at(diag,i-a,v)
        path = root/f"routing_{owner:04d}.npz"
        np.savez(path, id=np.arange(prefix,prefix+len(i)), i=i, j=j, v=v, diag=diag)
        edge_paths.append(path)
        prefix += len(i)
    for owner,(a,b) in enumerate(ranges):
        with np.load(edge_paths[owner], allow_pickle=False) as z:
            local = {key:z[key] for key in z.files}
        for path in edge_paths:
            with np.load(path, allow_pickle=False) as z:
                j = z["j"]
                keep = (j>=a)&(j<b)
                np.add.at(local["diag"],j[keep]-a,z["v"][keep])
        np.savez(root/f"owner{owner:04d}"/"limiter_graph.npz", **local)


def build_wall_products(options, out, inputs):
    import jax

    from ..operator_build.perp.sbp_setup import _layout
    from ..operator_build.parallel.indexed_input import IndexedInput
    from ..evaluators.magnetic_field import Evaluator
    from ..evaluators.magnetic_derivatives import FD_STEPS, magnetic_derivatives

    owner_p, qmeta, construction, _ = validate_inputs(inputs, parts=options.owner_count)
    layout = _layout(dict(owner_p.manifest, n=owner_p.manifest["N"]))
    q = OwnerParallelInputs(inputs["owner_parallel_construction"], qmeta, construction, owner_p)
    products = out/"products"
    products.mkdir()
    devices = jax.devices("cpu")[:owner_p.parts]
    with closing(IndexedInput(inputs["indexed_geometry"])) as indexed:
        for owner, device in enumerate(devices):
            folder = products/f"owner{owner:04d}"
            folder.mkdir()
            points = owner_p.read(owner, leaves=("points",))["points"].reshape(-1,3)
            ids, wall, rr = q.hits(owner, inputs["owner_parallel"], indexed)
            with jax.default_device(device):
                files = field_manifest(inputs, owner_p)["files"]
                ev = Evaluator(files["metric_cache"], files["field_table"])
                magnetic = magnetic_derivatives(ev, points, FD_STEPS[-1])
                rows, constants = _wall_rows(owner_p,owner,layout,wall,rr,magnetic,options.mu)
            np.savez(folder/"wall_plan.npz", **rows)
            np.savez(folder/"mms_wall_hit_reference.npz", ids=ids, hit_points=wall[:,1:4])
            if owner == 0:
                np.savez(products/"constants.npz", **constants)
    _write_limiter(products, q, [(a*q.width,b*q.width) for a,b in zip(q.edges[:-1],q.edges[1:])])
    from ..drb.owner_route_store import publish_routes
    boundary = layout.walls[0]
    nw = layout.blocks[boundary.block_idx].sides[boundary.side].N
    publish_routes(products/"routes",products,inputs["owner_parallel"],q,wall_per_plane=nw)
    write_json(products/"manifest.json", dict(edges=q.edges, owner_count=owner_p.parts))
    return {"wall_data": products}, 0
