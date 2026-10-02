"""Compact storage replay using checked actual-HSX callback fixtures."""
from dataclasses import replace
import numpy as np
import pytest

from tests.test_q_parallel_hsx_portable import patch
from drbx.stencils.q_bank import build_q_bank, decode_span, SPANS, SPAN_SLOTS
from drbx.stencils.q_parallel import _ARRAYS
from drbx.stencils.query_tables import ExactQueryTable
from drbx.stencils.artifact import _QueryTable


@pytest.fixture(scope='module')
def bank(patch):
    return build_q_bank(*patch[1], include_gradients=True,
                        identity={'evaluator': 'saved actual HSX fixture callbacks'})


def test_exact_decode_every_member(patch, bank):
    for span, q in zip(SPANS, patch[1]):
        decoded = decode_span(bank, span)
        assert decoded.metadata == q.metadata
        for k in _ARRAYS:
            a, b = getattr(decoded, k), getattr(q, k)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes(), k


@pytest.mark.parametrize('ai', [0, 1])
@pytest.mark.parametrize('kind', ['D', 'N'])
def test_actual_hsx_scalar_and_diffusion_action(patch, bank, ai, kind):
    data, prepared, x, boundaries = patch
    q, bc = prepared[ai], boundaries[ai]
    slots = np.array(SPAN_SLOTS[ai]); wall = bank.wall_index
    gathered = x[:, bank.donor]
    scalar_rows = bank.row_value_D.copy()
    diffusion_rows = bank.diffusion_D[ai].copy()
    if kind == 'N':
        scalar_rows[wall] = bank.row_value_N_wall
        diffusion_rows[wall] = bank.diffusion_N_wall[ai]
    scalar = np.einsum('frd,rsd->frs', gathered, scalar_rows[:, slots])
    action = np.sum(gathered * diffusion_rows, axis=-1)
    if kind == 'D':
        scalar[:, wall] += np.einsum('frj,rsj->frs', bc.dirichlet_trace[:, wall],
                                     bank.boundary_value_D_trace[:, slots]) + bc.dirichlet_query_value[:, wall]
        action[:, wall] += np.sum(bc.dirichlet_trace[:, wall] * bank.boundary_D_node[ai], axis=-1)
        action[:, wall] += np.sum(bc.dirichlet_tangent[:, wall] * bank.boundary_D_tangent[ai], axis=(-1, -2))
        old_scalar = np.einsum('frd,rsd->frs', gathered, q.row_value_D)
        old_scalar += np.einsum('frj,rsj->frs', bc.dirichlet_trace, q.boundary_value_D_trace)
        old_scalar[:, wall] += bc.dirichlet_query_value[:, wall]
    else:
        scalar[:, wall] += np.einsum('frj,rsj->frs', bc.neumann_normal[:, wall],
                                     bank.boundary_value_N_normal[:, slots])
        action[:, wall] += np.sum(bc.neumann_normal[:, wall] * bank.boundary_N_normal[ai], axis=-1)
        old_scalar = np.einsum('frd,rsd->frs', gathered, q.row_value_N)
        old_scalar += np.einsum('frj,rsj->frs', bc.neumann_normal, q.boundary_value_N_normal)
    np.testing.assert_array_equal(scalar, old_scalar)
    projected = np.sum(action[:, bank.owner_raw] * bank.owner_weight, axis=-1)
    np.testing.assert_allclose(projected, data['expected'][:, 0 if kind == 'D' else 1, ai, :].T,
                               rtol=0, atol=1e-8)


def test_no_gradients_are_not_silently_decoded(patch):
    compact = build_q_bank(*patch[1])
    with pytest.raises(ValueError, match='requires captured'):
        decode_span(compact, SPANS[0])
    assert all('gradient' not in k for k in compact.arrays)
    assert all('gradient' not in k for k in compact.diagnostics)
    assert compact.metadata['n_owner'] == patch[1][0].metadata['n_owner']
    assert compact.footprint()['runtime_logical_bytes'] < sum(q.nbytes for q in patch[1])


def test_paired_compatibility_checked_before_dedup(patch):
    outer, inner = patch[1]
    changed = inner.row_value_D.copy(); changed[1, 2, 0] += 1
    with pytest.raises(ValueError, match='center'):
        build_q_bank(outer, replace(inner, row_value_D=changed))
    with pytest.raises(ValueError, match='geometry_identity'):
        build_q_bank(outer, replace(inner, metadata=dict(inner.metadata, geometry_identity='wrong')))
    with pytest.raises(ValueError, match='spans'):
        build_q_bank(inner, outer)
    diagnostics = [dict(row) for row in inner.metadata['row_diagnostics']]
    diagnostics[0]['reproduction'] += 1
    with pytest.raises(ValueError, match='row_diagnostics'):
        build_q_bank(outer, replace(inner, metadata=dict(inner.metadata, row_diagnostics=diagnostics)))


def test_exact_coordinate_table_keeps_signed_zero_and_p_alias():
    assert _QueryTable is ExactQueryTable
    table = ExactQueryTable()
    plus = np.array([0., 1., 2.]); minus = plus.copy(); minus[0] = -0.
    assert table.add(plus) == table.add(plus.copy())
    assert table.add(plus) != table.add(minus)
    assert table.array()[1].tobytes() == minus.tobytes()


def test_raw_member_coverage_and_n_owner(bank):
    arrays = dict(bank.arrays)
    weights = arrays['owner_weight'].copy(); weights[0] = 0
    arrays['owner_weight'] = weights
    with pytest.raises(ValueError, match='coverage/projection'):
        replace(bank, arrays=arrays).validate()
    with pytest.raises(ValueError, match='incomplete Q bank identity'):
        replace(bank, metadata=dict(bank.metadata, source_identity='')).validate()
    diagnostics = dict(bank.diagnostics); diagnostics.pop('choice')
    with pytest.raises(ValueError, match='diagnostic manifest incomplete'):
        replace(bank, diagnostics=diagnostics).validate()


@pytest.mark.parametrize('key',['raw','owners','raw_to_owner','owner_raw','wall_index','row_count'])
def test_noninteger_source_identity_rejected(bank,key):
    arrays=dict(bank.arrays); arrays[key]=arrays[key].astype(float)+.25
    with pytest.raises(ValueError,match='integer identity dtype'):
        replace(bank,arrays=arrays).validate()


def test_source_relabeling_rejected_by_saved_trace_receipt(bank):
    arrays=dict(bank.arrays);raw=bank.raw.copy()
    replacement=next(i for i in range(bank.metadata['n']**3) if i not in set(raw))
    raw[0]=replacement;arrays['raw']=raw
    with pytest.raises(ValueError,match='raw content/trace receipt mismatch'):
        replace(bank,arrays=arrays).validate()


@pytest.mark.parametrize('receipt',['not-a-digest','a'*64+':short','g'*64+':'+'a'*64])
def test_malformed_trace_receipt_rejected(bank,receipt):
    metadata=dict(bank.metadata,trace_hash=receipt)
    metadata['span_metadata']=[dict(m,trace_hash=receipt) for m in bank.metadata['span_metadata']]
    with pytest.raises(ValueError,match='raw content/trace receipt mismatch'):
        replace(bank,metadata=metadata).validate()


@pytest.mark.parametrize('key',['raw_weight','owner_weight','row_value_D','diffusion_D','row_value_N_wall'])
def test_complex_numeric_coefficient_dtype_rejected(bank,key):
    arrays=dict(bank.arrays); arrays[key]=arrays[key].astype(complex)
    with pytest.raises(ValueError,match='real numeric dtype'):
        replace(bank,arrays=arrays).validate()


@pytest.mark.parametrize('key',['row_value_D','row_value_N_wall','diffusion_D','diffusion_N_wall',
                                'row_gradient_D','row_gradient_N'])
def test_nonzero_donor_padding_is_not_a_live_owner_zero_coefficient(bank,key):
    arrays=dict(bank.arrays);diagnostics=dict(bank.diagnostics)
    for name in ('donor','mask','row_value_D','row_value_N_wall','diffusion_D','diffusion_N_wall'):
        arrays[name]=np.pad(arrays[name],[(0,0)]*(arrays[name].ndim-1)+[(0,1)])
    for name in ('row_gradient_D','row_gradient_N'):
        diagnostics[name]=np.pad(diagnostics[name],[(0,0)]*3+[(0,1)])
    padded=replace(bank,arrays=arrays,diagnostics=diagnostics)
    padded.validate()
    group=diagnostics if key.startswith('row_gradient') else arrays
    changed=group[key].copy();changed[(0,)*(changed.ndim-1)+(-1,)]=1
    group[key]=changed
    with pytest.raises(ValueError,match='nonzero padded coefficient'):
        replace(bank,arrays=arrays,diagnostics=diagnostics).validate()
