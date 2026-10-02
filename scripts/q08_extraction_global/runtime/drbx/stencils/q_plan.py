"""Lower checked compact Q banks to array-data runtime plans.

Provenance remains in the host bank. Only owner count and accepted diffusion
span are static; numerical rows and balanced material coefficients are leaves.
"""
from dataclasses import dataclass, fields
import numpy as np
import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QPlan:
    n_owner: int
    diffusion_span: float
    donor: object
    owner_raw: object
    owner_weight: object
    wall_index: object
    row_value_D: object
    row_value_N_wall: object
    boundary_value_D_trace: object
    boundary_value_N_normal: object
    diffusion_D: object
    diffusion_N_wall: object
    boundary_D_node: object
    boundary_D_tangent: object
    boundary_N_normal: object
    magnetic_L: object
    b_eta: object
    eta_step: object
    bmag: object

    def tree_flatten(self):
        return tuple(getattr(self, f.name) for f in fields(self)[2:]), (self.n_owner, self.diffusion_span)

    @classmethod
    def tree_unflatten(cls, static, arrays):
        return cls(*static, *arrays)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.tree_flatten()[0])


def lower_q_plan(bank, *, diffusion_span, material_runtime=None,
                 magnetic_L=None, b_eta=None, eta_step=None, bmag=None):
    """Use an authoritative SixFieldRuntime or explicit balanced coefficients.

    Explicit coefficients are supplied by the caller's qualified geometry
    authority; lowering never estimates div(b), traces or imports MMS sources.
    """
    bank.validate()
    if diffusion_span not in (1/16, 1/32):
        raise ValueError('explicit accepted diffusion span required')
    nr = len(bank.raw)
    if material_runtime is not None:
        if any(a is not None for a in (magnetic_L, b_eta, eta_step, bmag)):
            raise ValueError('choose material_runtime or explicit coefficients')
        rt = material_runtime
        if rt.metadata.get('schema') != 'drbx.q-six-field-prescribed.v1' or 'balanced_tube' not in rt.material.metadata:
            raise ValueError('authoritative balanced six-field runtime required')
        if rt.metadata['diffusion_span'] != diffusion_span:
            raise ValueError('runtime diffusion span mismatch')
        for scalar, slots in ((rt.material.inner, [1,3,2]), (rt.material.outer, [0,4,2])):
            for name in ('donor', 'owner_raw', 'owner_weight'):
                if not np.array_equal(getattr(scalar, name), getattr(bank, name)):
                    raise ValueError(f'material bank identity mismatch: {name}')
            for key in ('geometry_identity','topology_hash','source_identity','trace_hash','campaign_identity'):
                if scalar.metadata[key] != bank.metadata['span_metadata'][0][key]:
                    raise ValueError(f'material bank identity mismatch: {key}')
            if not np.array_equal(scalar.coefficient_D, bank.row_value_D[:,slots]):
                raise ValueError('material bank scalar row mismatch')
            expected_n = np.array(bank.row_value_D[:,slots], copy=True)
            expected_n[bank.wall_index] = bank.row_value_N_wall[:,slots]
            if not np.array_equal(scalar.coefficient_N, expected_n):
                raise ValueError('material bank normal row mismatch')
            for source, target in (('lift_D_node','boundary_value_D_trace'),
                                   ('lift_N_normal','boundary_value_N_normal')):
                expected = np.zeros_like(getattr(scalar, source))
                expected[bank.wall_index] = getattr(bank,target)[:,slots]
                if not np.array_equal(getattr(scalar,source),expected):
                    raise ValueError('material bank boundary response mismatch')
            query = np.zeros_like(scalar.lift_D_query)
            query[bank.wall_index] = 1
            if not np.array_equal(scalar.lift_D_query,query) or np.any(scalar.lift_D_tangent!=0):
                raise ValueError('material bank query response mismatch')
        ai = 0 if diffusion_span == 1/16 else 1
        for name in ('donor','owner_raw','owner_weight','diffusion_D'):
            expected = bank.diffusion_D[ai] if name=='diffusion_D' else getattr(bank,name)
            if not np.array_equal(getattr(rt.diffusion,name),expected):
                raise ValueError(f'diffusion bank identity mismatch: {name}')
        from .q_parallel import sha256_array
        if (sha256_array(rt.material.magnetic_L)!=rt.material.metadata['balanced_tube']['weights'] or
            sha256_array(rt.bmag)!=rt.metadata['bmag'] or
            any(sha256_array(getattr(rt.material,name))!=rt.material.metadata['arrays'][name]
                for name in ('b_eta','eta_step'))):
            raise ValueError('balanced coefficient authority identity mismatch')
        magnetic_L, b_eta, eta_step, bmag = rt.material.magnetic_L, rt.material.b_eta, rt.material.eta_step, rt.bmag
    arrays = {k: np.asarray(v) for k,v in dict(magnetic_L=magnetic_L,
              b_eta=b_eta, eta_step=eta_step, bmag=bmag).items()}
    for name, shape in (('magnetic_L',(nr,3)), ('b_eta',(nr,)), ('bmag',(nr,))):
        a=arrays[name]
        if a.shape != shape or a.dtype.kind != 'f' or not np.isfinite(a).all():
            raise ValueError(f'{name} requires finite floating shape {shape}')
    if arrays['eta_step'].shape not in ((), (nr,)) or arrays['eta_step'].dtype.kind != 'f' or not np.all(np.isfinite(arrays['eta_step']) & (arrays['eta_step']>0)):
        raise ValueError('eta_step requires finite positive scalar or raw-row array')
    if not np.all(arrays['bmag']>0):
        raise ValueError('bmag requires positive values')
    ai = 0 if diffusion_span == 1/16 else 1
    names = tuple(f.name for f in fields(QPlan)[2:])
    selected = {name: getattr(bank,name) for name in names if name not in arrays}
    for name in ('diffusion_D','diffusion_N_wall','boundary_D_node','boundary_D_tangent','boundary_N_normal'):
        selected[name] = selected[name][ai]
    return QPlan(int(bank.metadata['n_owner']), diffusion_span, **selected, **arrays)
