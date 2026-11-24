"""Host assembly of retained resolution 16 wall maps; no tracing or full Jacobian."""
from dataclasses import dataclass
import numpy as np
import jax


@dataclass(frozen=True)
class OwnerWallGrid:
    plane_counts:tuple
    wall_per_plane:int
    deta:float


def build_owner_wall_plan(mesh,nodes,p_walls,q_hits,rows,*,eta_planes,nodes_per_plane,
                          wall_per_plane,deta,Om,Bdiv,Wm,gi):
    """Assemble wall operators from canonical per-owner cache rows."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ..operators.parallel.owner_routes import pack_owner_csr
    from .walls import WallOps
    parts=mesh.size
    platforms = {d.platform for d in mesh.devices.flat}
    if tuple(mesh.axis_names)!=('records',) or platforms not in ({'cpu'}, {'gpu'}):
        raise ValueError('The owner wall plan requires the Q records mesh on one CPU or GPU platform')
    cuts=(np.arange(parts+1)*eta_planes+parts-1)//parts
    counts=tuple(map(int,np.diff(cuts)))
    if min(counts)<4:
        raise ValueError('At least four eta planes are required per owner')
    for layout,width in ((nodes,nodes_per_plane),(p_walls,wall_per_plane)):
        if layout.n!=eta_planes*width or layout.cap!=max(counts)*width:
            raise ValueError('Owner rows differ from the canonical plane slab layout')
        for p,ids in layout.ids_by_owner.items():
            if not np.array_equal(ids[ids>=0],np.arange(cuts[p]*width,cuts[p+1]*width)):
                raise ValueError('Owner rows disagree with the Q v3 plane cuts')
    maps={}
    for name,source,target,width,halo in (
        ('T',nodes,p_walls,nodes_per_plane,0),('TT',p_walls,nodes,wall_per_plane,0),
        ('Tc',nodes,p_walls,nodes_per_plane,0),('TcT',p_walls,nodes,wall_per_plane,0),
        ('F',nodes,p_walls,nodes_per_plane,2),('K',nodes,nodes,nodes_per_plane,4)):
        pieces={p:entry[name] for p,entry in rows.items()}
        target_width=wall_per_plane if target is p_walls else nodes_per_plane
        for p,A in pieces.items():
            rr=np.repeat(target.ids_by_owner[p][target.ids_by_owner[p]>=0],np.diff(A.indptr))//target_width
            cc=A.indices//width
            if np.any(np.minimum((rr-cc)%eta_planes,(cc-rr)%eta_planes)>halo):
                raise ValueError(f'{name} exceeds its periodic eta halo')
        maps[name]=pack_owner_csr(mesh,source,target,pieces)
    for layout,names in (
        (nodes,('H',)),(p_walls,('Js','curvature_flux','hw','bn','trace_gram','mass_trace_gram','A0plus')),
        (q_hits,('Y','Khit','wall_foot_to_hit_area_ratio'))):
        for name in names:
            pad=1 if name in ('H','Js','trace_gram','mass_trace_gram','wall_foot_to_hit_area_ratio') else 0
            maps[name]=layout.put({p:row[name] for p,row in rows.items()},mesh,pad)
    replicated=NamedSharding(mesh,P())
    for name,value in dict(Om=Om,Bdiv=Bdiv,Wm=Wm,gi=gi).items():
        maps[name]=jax.device_put(np.asarray(value),replicated)
    return WallOps(**maps),OwnerWallGrid(counts,wall_per_plane,float(deta))
