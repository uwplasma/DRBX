"""Join checksum-bound perpendicular and parallel owner products on the Q records mesh."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from ..operators.perp import sbp_owner as sbp
from ..operators.perp.rhs_options import (
    FIELD_NAMES, PerpendicularOptions, NodalPerpendicularParams,
)
from ..operators.parallel.owner_products import LocalLayout, load_owner_parallel
from .cache import load_owner_limiter_graph, load_owner_wall
from .operator import Plan
from .owner_route_store import addressable, replicate, route_plans
from .rank_runtime import rank_sbp, read_owned_p


@dataclass(frozen=True)
class OwnerResolution:
    plan: Plan
    q: object
    slabs: sbp.SlabLayout
    p_walls: LocalLayout
    wall_grid: object
    node_active: object
    hit_active: object


def perpendicular_partition_specs(plan):
    shard = P(sbp.AXIS)
    def specs(tree, fields):
        shape = jax.tree.map(lambda _: P(), tree)
        return replace(shape, **{name:shard for name in fields if getattr(tree, name) is not None})
    return sbp.OwnerPlan(specs(plan.nodal, sbp.NODAL_PLANE_FIELDS),
                         specs(plan.laplacian, sbp.PLANE_FIELDS),
                         jax.tree.map(lambda _: shard, plan.factors))


def _put_plans(plans, mesh):
    devices = [d for _,d in addressable(mesh)]
    sharded = NamedSharding(mesh, P(sbp.AXIS))
    replicated = NamedSharding(mesh, P())
    def place(spec, x, *others):
        if spec:
            pieces = [jax.device_put(np.asarray(value)[None], device)
                      for value, device in zip((x, *others), devices, strict=True)]
            return jax.make_array_from_single_device_arrays((mesh.size, *x.shape), sharded, pieces)
        if any(not np.array_equal(x, other) for other in others):
            raise ValueError('P in-plane templates or global penalties differ across owners')
        return jax.device_put(x, replicated) if len(devices)==mesh.size else replicate(mesh,x)
    return jax.tree.map(place, perpendicular_partition_specs(plans[0]), *plans)


def _plane_layout(slabs, width, *, owners=None):
    cap = slabs.capacity * width
    ids = {}
    for owner, (a, b) in enumerate(zip(slabs.edges[:-1], slabs.edges[1:])):
        if owners is not None and owner not in owners:
            continue
        ids[owner] = np.pad(np.arange(a*width, b*width, dtype=np.int32),
                            (0, cap-(b-a)*width), constant_values=-1)
    return LocalLayout(slabs.n_eta*width, cap, slabs.parts, ids)


def load_owner_resolution(*, layout, owner_p, q_products, wall_cache, wall_constants,
                          perpendicular_identity_sha256, indexed_geometry, read_parallel_mass, read_parallel_face_stiffness,
                          options, read_owner_sbp=None, devices=None, routes=None):
    """Read and place owner slices only.

    The expected P source hash and Q indexed identity come from the same
    build. read_parallel_mass(owner, *, node_ids) returns owned wall_credited_parallel_mass and MB_V;
    read_parallel_face_stiffness(owner, *, node_ids) returns owned K_N CSR with canonical columns.
    """
    md = json.loads((Path(q_products)/'metadata.json').read_text())
    p_meta = owner_p.manifest
    slabs = sbp.SlabLayout(layout.n_eta, options.owner_count)
    if tuple(owner_p.edges) != slabs.edges:
        raise ValueError('P owner plane cuts differ from Q v3 plane cuts')
    keys = ('N', 'symmetry', 'field_periods')
    if ((p_meta['N'], p_meta['P'], p_meta['n_eta']) != (layout.n, layout.P, layout.n_eta)
        or md['eta_planes'] != layout.n_eta or any(p_meta[k] != md[k] for k in keys)):
        raise ValueError('P layout differs from Q geometry')
    if (p_meta['source_identity_sha256'] != perpendicular_identity_sha256
        or md['indexed_geometry'] != indexed_geometry):
        raise ValueError('Owner P/Q geometry identity differs from the bound inputs')

    q = load_owner_parallel(q_products, options, devices=devices)
    mesh, nodes, hits = q.mesh, q.nodes, q.walls
    local = addressable(mesh)
    owners = {i for i,_ in local}
    rank_mode = routes is not None
    canonical = _plane_layout(slabs, layout.P, owners=owners)
    if ((nodes.n, nodes.cap) != (canonical.n, canonical.cap)
        or any(not np.array_equal(nodes.ids_by_owner[i], ids) for i, ids in canonical.ids_by_owner.items())):
        raise ValueError('perpendicular and parallel canonical node rows differ')
    boundary = layout.walls[0]
    p_walls = _plane_layout(slabs, layout.blocks[boundary.block_idx].sides[boundary.side].N, owners=owners)
    if rank_mode:
        wall,grid,graph = route_plans(routes,mesh,nodes,p_walls,hits,edges=slabs.edges,
            deta=layout.deta,q_products=q_products,constants=wall_constants)
    else:
        wall, grid = load_owner_wall(wall_cache, mesh, nodes, p_walls, hits, eta_planes=layout.n_eta,
                                     nodes_per_plane=layout.P, wall_per_plane=p_walls.n//layout.n_eta,
                                     deta=layout.deta, **wall_constants)
    replica = NamedSharding(mesh, P())
    opts = PerpendicularOptions(fields=FIELD_NAMES, diffusion_kinds=('neumann',)*5+('dirichlet',),
        phi_mode='prescribed', psi='phi_plus_tau_pi', phi_rtol=2e-13, phi_maxit=200)
    reader = owner_p.read_owner_sbp if read_owner_sbp is None else read_owner_sbp
    for owner in owners if not rank_mode else ():
        owner_p.verify(owner,leaves=('points','h','jac','B','K','A','ginv_u'))
    plans, K, mass_ratio, dirichlet_ratio = [], {}, {}, {}
    node_leaves = ({s.device:s.data for s in array.addressable_shards}
                   for array in (q.p['H'],q.p['B2'],wall.H))
    q_H, q_B2, p_H = node_leaves
    for owner, device in local:
        ids = nodes.ids_by_owner[owner]
        ids = ids[ids >= 0]
        owned = read_owned_p(owner_p,owner) if rank_mode else None
        points = owned[0]['points'] if rank_mode else owner_p.read(owner, leaves=('points',))['points']
        planes = slabs.plane_ids(owner)
        u, theta, eta = np.moveaxis(points, -1, 0)
        if (points.shape != (len(planes),layout.P,3)
            or not np.allclose(u, layout.node_u, rtol=0, atol=1e-13)
            or not np.allclose(theta, layout.node_theta, rtol=0, atol=1e-13)
            or not np.allclose(eta, (planes[:,None]+.5)*layout.deta, rtol=0, atol=1e-13)):
            raise ValueError('P owner nodes differ from the bound canonical geometry')
        data = rank_sbp(mesh,slabs,owner,owned) if rank_mode else reader(owner, edges=slabs.edges)
        if not np.array_equal(np.asarray(data.nodal.B).ravel()**2, np.asarray(q_B2[device])[:len(ids)]):
            raise ValueError('P/Q normalized magnetic metric differs at owner nodes')
        plans.append(sbp.build_owner_plan(layout, slabs, owner, data, c_kappa=opts.laplacian_c_kappa))
        if not rank_mode:
            K[owner] = read_parallel_face_stiffness(owner, node_ids=ids)
        wall_credited_parallel_mass, MB_V = map(np.asarray, read_parallel_mass(owner, node_ids=ids))
        Hp = np.asarray(p_H[device])[:len(ids)]
        if wall_credited_parallel_mass.shape != (len(ids),) or MB_V.shape != wall_credited_parallel_mass.shape or not np.array_equal(
                wall_credited_parallel_mass, np.asarray(q_H[device])[:len(ids)]):
            raise ValueError('Q owner mass inputs differ from the stored Q mass')
        mass_ratio[owner] = wall_credited_parallel_mass/Hp
        dirichlet_ratio[owner] = (wall_credited_parallel_mass-MB_V)/wall_credited_parallel_mass

    if not rank_mode:
        graph = load_owner_limiter_graph(wall_cache, mesh, nodes, K)
    def rep(a):
        return replicate(mesh,a) if rank_mode else jax.device_put(np.asarray(a), replica)
    par = NodalPerpendicularParams(rep(.00045), rep(options.tau), rep(np.asarray([1,1.2,1.4,0,0,.8])*1e-5))
    plan = Plan(_put_plans(plans,mesh), par, q.p, q.B, q.BT, q.LN, q.LD, None,
                rep([.01,400.,11.,5.6,180.,.06]), nodes.put(mass_ratio,mesh,1),
                nodes.put(dirichlet_ratio,mesh,1), rep(.0027), rep(19.29), opts, q.params,
                wall_ops=wall, afc=graph, runtime_options=options)
    def active(rows):
        return rows.put({i:v>=0 for i,v in rows.ids_by_owner.items()},mesh)
    return OwnerResolution(plan, q, slabs, p_walls, grid, active(nodes), active(hits))
