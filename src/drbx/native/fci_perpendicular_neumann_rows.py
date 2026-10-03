"""Array-only runtime for structured physical-normal Neumann point rows."""
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np


class NeumannPayload(NamedTuple):
    """``value_weights`` / ``boundary_value_weights`` (``gradient_weights`` / ``boundary_gradient_weights``) are
    ``None`` in a plan whose operators never read the values (gradients) of these rows."""
    donor_ids: np.ndarray
    value_weights: np.ndarray | None
    gradient_weights: np.ndarray | None
    boundary_ids: np.ndarray
    boundary_value_weights: np.ndarray | None
    boundary_gradient_weights: np.ndarray | None
    boundary_query_count: int


class NeumannFacePayload(NamedTuple):
    """Per-face union layout of the rows of ``Fn`` faces with ``Q`` nodes each (face-major, node-minor row order).

    A face holds one donor list ``donor_ids (Fn, U)`` (the union of its rows' donors, zero-weight padded) and one
    boundary-point list ``boundary_ids (Fn, B)``; the weights of node ``q`` are scattered onto these union positions.
    A part the plan's operators never read is ``None``, as in :class:`NeumannPayload`."""
    donor_ids: np.ndarray
    value_weights: np.ndarray | None                # (Fn, Q, U)
    gradient_weights: np.ndarray | None             # (Fn, Q, 3, U)
    boundary_ids: np.ndarray
    boundary_value_weights: np.ndarray | None       # (Fn, Q, B)
    boundary_gradient_weights: np.ndarray | None    # (Fn, Q, 3, B)
    boundary_query_count: int


class IntegratedNeumannPayload(NamedTuple):
    """P07 Neumann restoration of ``Fn`` faces in the union layout, pre-contracted with the q3 integrand:
    ``flux = weights @ owner_fields[donor_ids] + boundary_weights @ g_N[boundary_ids]``."""
    donor_ids: np.ndarray                           # (Fn, U)
    weights: np.ndarray                             # (Fn, U)
    boundary_ids: np.ndarray                        # (Fn, B)
    boundary_weights: np.ndarray                    # (Fn, B)
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


def apply_neumann_point_rows(payload: NeumannPayload | NeumannFacePayload, owner_fields, normal_derivative, *,
                             values=True, gradients=True):
    """Apply coherent value/gradient rows to fields and prescribed g_N (``None`` for a part the payload does not store
    or that ``values`` / ``gradients`` switch off).

    A :class:`NeumannFacePayload` gives the same ``(R, F)`` / ``(R, 3, F)`` rows (``R = Fn * Q``, face-major): one donor
    gather per face and a per-face contraction instead of one gather per row."""
    field=jnp.asarray(owner_fields)
    g=jnp.asarray(normal_derivative)
    if field.ndim!=2 or g.ndim!=2 or g.shape[1]!=field.shape[1]:
        raise ValueError("Neumann owner or boundary array shape mismatch")
    sampled=field[payload.donor_ids]
    bv=g[payload.boundary_ids]
    value=gradient=None
    if isinstance(payload,NeumannFacePayload):
        nf=field.shape[1]
        if values and payload.value_weights is not None:
            value=jnp.einsum('fqu,fuk->fqk',payload.value_weights,sampled)
            value=(value+jnp.einsum('fqb,fbk->fqk',payload.boundary_value_weights,bv)).reshape(-1,nf)
        if gradients and payload.gradient_weights is not None:
            gradient=jnp.einsum('fqau,fuk->fqak',payload.gradient_weights,sampled)
            gradient=(gradient+jnp.einsum('fqab,fbk->fqak',payload.boundary_gradient_weights,bv)).reshape(-1,3,nf)
        return value,gradient
    if values and payload.value_weights is not None:
        value=jnp.einsum('rd,rdf->rf',payload.value_weights,sampled)
        value+=jnp.einsum('rq,rqf->rf',payload.boundary_value_weights,bv)
    if gradients and payload.gradient_weights is not None:
        gradient=jnp.einsum('rad,rdf->raf',payload.gradient_weights,sampled)
        gradient+=jnp.einsum('raq,rqf->raf',payload.boundary_gradient_weights,bv)
    return value,gradient


def apply_neumann_integrated_face_rows(payload: NeumannPayload | IntegratedNeumannPayload, owner_fields,
                                       normal_derivative, integrand=None):
    """Contract q3 gradients with P07's weighted normal tensor rows (``integrand``: ``(Fn, 9, 3)``).

    An :class:`IntegratedNeumannPayload` already holds the contraction with the integrand (``integrand`` is not read)."""
    if isinstance(payload,IntegratedNeumannPayload):
        field=jnp.asarray(owner_fields)
        g=jnp.asarray(normal_derivative)
        if field.ndim!=2 or g.ndim!=2 or g.shape[1]!=field.shape[1]:
            raise ValueError("Neumann owner or boundary array shape mismatch")
        return (jnp.einsum('fu,fuk->fk',payload.weights,field[payload.donor_ids])
                +jnp.einsum('fb,fbk->fk',payload.boundary_weights,g[payload.boundary_ids]))
    weighted=jnp.asarray(integrand)
    if weighted.ndim!=3 or weighted.shape[1:]!=(9,3) or weighted.shape[0]*9!=len(payload.donor_ids):
        raise ValueError("P07 integrand must have shape (faces, 9, 3)")
    _,gradient=apply_neumann_point_rows(payload,owner_fields,normal_derivative)
    return jnp.einsum('fqa,fqak->fk',weighted,gradient.reshape(weighted.shape[0],9,3,-1))
