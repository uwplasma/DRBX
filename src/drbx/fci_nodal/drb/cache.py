"""Serialization of the prepared wall plan and limiter graph."""
from pathlib import Path

import numpy as np
from scipy import sparse


def load_owner_wall(directory,mesh,nodes,p_walls,q_hits,**geometry):
    """Read one serialized wall cache shard at a time in canonical owner layout."""
    from .wall_plan import build_owner_wall_plan
    root=Path(directory)
    def read(p):
        path=root/f'owner{p:04d}'/'wall_plan.npz'
        matrices=('T','TT','Tc','TcT','F','K')
        with np.load(path,allow_pickle=False) as z:
            rows={k:z[k] for k in z.files if not any(k.startswith(name+'_') for name in matrices)}
            rows.update({k:sparse.csr_matrix((z[k+'_data'],z[k+'_indices'],z[k+'_indptr']),
                                             shape=tuple(z[k+'_shape'])) for k in matrices})
        return rows
    return build_owner_wall_plan(mesh,nodes,p_walls,q_hits,
                                 {p:read(p) for p in range(mesh.size)},**geometry)


def load_owner_limiter_graph(directory,mesh,nodes,K_rows):
    """Load per-owner wrong-sign edge rows and their local low diagonal."""
    from .afc import build_owner_limiter_graph
    root=Path(directory)
    edges={};diag={}
    for p in range(mesh.size):
        with np.load(root/f'owner{p:04d}'/'limiter_graph.npz',allow_pickle=False) as z:
            edges[p]=tuple(z[k] for k in ('id','i','j','v'))
            diag[p]=z['diag']
    return build_owner_limiter_graph(mesh,nodes,K_rows,edges,diag)
