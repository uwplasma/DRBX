"""Resolution-pinned manufactured field cache and prepared receipt."""
from pathlib import Path
import json
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from .._hashing import sha256_stream
from ..drb.owner_route_store import addressable, replicate

def digest(path):
    with Path(path).open('rb') as f:return sha256_stream(f)


_OWNER_SCHEMA='drbx.mms.owner_prepared.v1'

WALL_LAW_MEMBER_NAMES = {
    'conducting_wall': ('manufactured_field_data_conducting_wall', 'conducting_wall_D', 'conducting_wall_L'),
    'floating_wall': ('manufactured_field_data_floating_wall', 'floating_wall_D', 'floating_wall_L'),
}


def _manufactured_field_members(data):
    result={}
    for key,value in data.items():
        if hasattr(value,'_fields'):
            result.update({key+'__'+name:np.asarray(a) for name,a in zip(value._fields,value)})
        else:
            result[key]=np.asarray(value)
    return result


def _save_owner_members(root,owner,groups):
    relative=Path(f'owner{owner:04d}')
    directory=root/relative
    directory.mkdir()
    files={}
    for name,arrays in groups.items():
        values={key:np.asarray(a) for key,a in arrays.items()}
        path=directory/(name+'.npz')
        np.savez(path,**values)
        files[name]=dict(path=(relative/path.name).as_posix(),sha256=digest(path),
                         bytes=path.stat().st_size,
                         members={key:dict(shape=list(a.shape),dtype=a.dtype.str)
                                  for key,a in values.items()})
    return files


class OwnerPrepared:
    """Open the catalog once. Read and authenticate only the selected owner."""
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        m=self.manifest
        if m['schema']!=_OWNER_SCHEMA:
            raise ValueError('Unsupported prepared-owner schema')
        self.parts,self.edges=m['owner_count'],tuple(m['edges'])
        if not 1<=self.parts<=m['eta_planes']//4:
            raise ValueError('Prepared owners need at least four planes')
        expected=tuple((p*m['eta_planes']+self.parts-1)//self.parts for p in range(self.parts+1))
        bounds=[(x['begin'],x['end']) for x in m['owners']]
        if self.edges!=expected or bounds!=list(zip(expected[:-1],expected[1:])):
            raise ValueError('Noncanonical prepared owner partition')

    def read(self,owner,*,groups=None):
        if type(owner) is not int or not 0<=owner<self.parts:
            raise ValueError('Invalid prepared owner')
        files=self.manifest['owners'][owner]['files']
        selected=files if groups is None else tuple(groups)
        result={}
        for name in selected:
            entry=files[name]
            relative=f'owner{owner:04d}/{name}.npz'
            path=self.root/relative
            if not name.isidentifier() or entry['path']!=relative or path.resolve()!=path:
                raise ValueError('Invalid prepared owner path')
            if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
                raise ValueError(f'Changed owner prepared file {relative}')
            with np.load(path,allow_pickle=False) as data:
                if set(data.files)!=set(entry['members']):
                    raise ValueError(f'Changed prepared members {relative}')
                members={}
                for key,spec in entry['members'].items():
                    array=data[key]
                    if list(array.shape)!=spec['shape'] or array.dtype.str!=spec['dtype']:
                        raise ValueError(f'Changed prepared member {relative}/{key}')
                    members[key]=array
            result[name]=members
        return result

    def read_manufactured_field_data(self,owner,wall_law,*,device=None):
        """Read the manufactured field data once for repeated stage evaluations on its device."""
        import jax
        import jax.numpy as jnp
        from .evolved import ExactJet
        name, _, _ = WALL_LAW_MEMBER_NAMES[wall_law]
        arrays=self.read(owner,groups=(name,))[name]
        put=jnp.asarray if device is None else lambda a:jax.device_put(a,device)
        result={key:put(a) for key,a in arrays.items() if '__' not in key}
        for key in ('nodal','pwall','A'):
            result[key]=ExactJet(*(put(arrays[key+'__'+field]) for field in ExactJet._fields))
        return result


def read_mms_rows(prepared, owner, wall_law, ids, device):
    hit_layout = prepared.read(owner,groups=('hit_layout',))['hit_layout']
    if not np.array_equal(ids[ids>=0],hit_layout['ids']):
        raise ValueError('Prepared Q hit ordinals differ from the coupled layout')
    local = prepared.read_manufactured_field_data(owner,wall_law,device=device)
    return dict(local,wall_ginv=local['wall_ginv'].reshape(-1,3))


def load_owner_mms(resolution, prepared, *, wall_law):
    """Place the authenticated manufactured field data directly in the coupled device layout."""
    q, p_walls = resolution.q, resolution.p_walls
    if prepared.edges != resolution.slabs.edges:
        raise ValueError('Prepared MMS and coupled owner edges differ')
    if (prepared.manifest['nodes_per_plane'] != q.nodes.n//resolution.slabs.n_eta
        or prepared.manifest['wall_per_plane'] != resolution.wall_grid.wall_per_plane):
        raise ValueError('Prepared MMS and coupled plane sizes differ')
    local = addressable(q.mesh)
    rows = {owner:read_mms_rows(prepared,owner,wall_law,q.walls.ids_by_owner[owner],device)
            for owner,device in local}
    first_owner = local[0][0]

    def place(key):
        if key in ('amplitude','uses_nonlinear_manufactured_potential','current_coefficient','uses_grounded_wall_potential'):
            first = np.asarray(rows[first_owner][key])
            if any(not np.array_equal(first, row[key]) for row in rows.values()):
                raise ValueError('Prepared scalar differs across owners')
            return (jax.device_put(first,NamedSharding(q.mesh,P())) if len(local)==q.mesh.size
                    else replicate(q.mesh,first))
        if key in ('g','parallel_wall_flux_base'):
            layout, neutral = q.walls, ([1,1,1,0,0,0] if key=='g' else 0.)
        elif key in ('pwall','perpendicular_wall_flux_base','wall_ginv'):
            layout, neutral = p_walls, ([1,0,0] if key=='wall_ginv' else 0.)
        else:
            layout, neutral = q.nodes, (1. if key in ('B','jac') else 0.)
        return jax.tree.map(lambda *a:layout.put(dict(zip(rows,a,strict=True)),q.mesh,neutral),
                            *(row[key] for row in rows.values()))
    return {key:place(key) for key in rows[first_owner]}
