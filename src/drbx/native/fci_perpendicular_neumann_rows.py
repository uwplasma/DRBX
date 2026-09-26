"""Array-only runtime for structured physical-normal Neumann point rows."""
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np


class NeumannPayload(NamedTuple):
    donor_ids: np.ndarray
    value_weights: np.ndarray
    gradient_weights: np.ndarray
    boundary_ids: np.ndarray
    boundary_value_weights: np.ndarray
    boundary_gradient_weights: np.ndarray
    boundary_query_count: int


def lower_neumann_point_rows(rows):
    """Deduplicate wall nodes and pad owner rows for JIT/JVP application."""
    rows=tuple(rows)
    queries=[];mapping={};ids=[]
    for row in rows:
        local=[]
        for p in row.boundary_points:
            key=tuple(map(float,p))
            if key not in mapping:
                mapping[key]=len(queries);queries.append(np.asarray(p,dtype=np.float64))
            local.append(mapping[key])
        ids.append(local)
    width=max((len(row.donor_ids) for row in rows),default=0)
    donor=np.zeros((len(rows),width),dtype=np.int64)
    value=np.zeros((len(rows),width));gradient=np.zeros((len(rows),3,width))
    for q,row in enumerate(rows):
        d=len(row.donor_ids);donor[q,:d]=row.donor_ids
        value[q,:d]=row.value;gradient[q,:,:d]=row.gradient
    payload=NeumannPayload(donor,value,gradient,np.asarray(ids,dtype=np.int64).reshape(-1,28),
                           np.asarray([row.boundary_value for row in rows]).reshape(-1,28),
                           np.asarray([row.boundary_gradient for row in rows]).reshape(-1,3,28),len(queries))
    return payload,np.asarray(queries,dtype=np.float64).reshape(-1,3)


def apply_neumann_point_rows(payload: NeumannPayload, owner_fields, normal_derivative):
    """Apply coherent value/gradient rows to fields and prescribed g_N."""
    field=jnp.asarray(owner_fields)
    g=jnp.asarray(normal_derivative)
    if field.ndim!=2 or g.ndim!=2 or g.shape[1]!=field.shape[1]:
        raise ValueError("Neumann owner or boundary array shape mismatch")
    sampled=field[payload.donor_ids]
    bv=g[payload.boundary_ids]
    value=jnp.einsum('rd,rdf->rf',payload.value_weights,sampled)
    value+=jnp.einsum('rq,rqf->rf',payload.boundary_value_weights,bv)
    gradient=jnp.einsum('rad,rdf->raf',payload.gradient_weights,sampled)
    gradient+=jnp.einsum('raq,rqf->raf',payload.boundary_gradient_weights,bv)
    return value,gradient


def apply_neumann_integrated_face_rows(payload: NeumannPayload, owner_fields, normal_derivative, integrand):
    """Contract q3 gradients with P07's weighted normal tensor rows."""
    weighted=jnp.asarray(integrand)
    if weighted.ndim!=3 or weighted.shape[1:]!=(9,3) or weighted.shape[0]*9!=len(payload.donor_ids):
        raise ValueError("P07 integrand must have shape (faces, 9, 3)")
    _,gradient=apply_neumann_point_rows(payload,owner_fields,normal_derivative)
    return jnp.einsum('fqa,fqak->fk',weighted,gradient.reshape(weighted.shape[0],9,3,-1))
