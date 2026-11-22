"""Data-free checks for the canonical owner-only Q recipe."""


import numpy as np
import pytest

from drbx.fci_nodal.operator_build.parallel.construction_provenance import (
    face_quadrature_provenance, indexed_layout, owner_plane_paths, verify_parallel_construction,
)


class IndexedFaces:
    def __init__(self, directory):
        self.info = {'N':4,'E':8,'P':2,'NN':16}
        self.handoff = directory/'handoff.json'
        (directory/'geometry.json').write_text('{"symmetry":"none","field_periods":1}')
        self.faces = np.column_stack([2*np.arange(8),2*np.arange(8)+1])

    def identity(self):
        return {'index_sha256':'synthetic'}

    def shape(self, name):
        assert name=='faces'
        return self.faces.shape

    def take(self, name, ids):
        assert name=='faces'
        return self.faces[ids]


@pytest.fixture
def q_leaves(tmp_path):
    indexed = IndexedFaces(tmp_path)
    owners = []
    for owner,(a,b) in enumerate(((0,4),(4,8))):
        directory = f'owners/owner_{owner:05d}'
        owners.append(dict(owner=owner,begin=a,end=b,directory=directory))
        for plane in range(a,b):
            path = tmp_path/directory/f'plane_{plane:05d}'
            path.mkdir(parents=True)
            np.save(path/'moment_reduced_face_quadrature.npy',np.array([plane],np.int64))
            np.save(path/'mu.npy',np.array([1.+plane],np.float64))
    manifest = dict(format='drbx.parallel.owner_construction.v1',owners=owners,
                    index_binding=indexed.identity(),layout=indexed_layout(indexed),
                    plane_edges=[0,2,4,6,8,10,12,14,indexed.info['NN']])
    paths = owner_plane_paths(tmp_path,manifest)
    manifest['moment_reduced_face_quadrature'] = face_quadrature_provenance(paths,indexed)
    return indexed,manifest,paths


def test_parallel_face_pair_manifest(q_leaves):
    indexed,manifest,paths = q_leaves
    result = verify_parallel_construction(manifest,indexed,paths)
    assert result==manifest['moment_reduced_face_quadrature']
    assert result['count']==8
    assert [leaf['plane'] for leaf in result['leaves']]==list(range(8))
    assert all(leaf['count']==1 for leaf in result['leaves'])


@pytest.mark.parametrize('member,values,error', [
    ('moment_reduced_face_quadrature',np.array([1],np.int64),'physical owner plane'),
    ('moment_reduced_face_quadrature',np.array([0],np.int32),'recipe leaves'),
    ('mu',np.array([0.],np.float64),'canonical moment-reduced face quadrature weight'),
    ('mu',np.array([.5],np.float64),'differ from the Q construction manifest'),
])
def test_parallel_face_pair_manifest_rejects_changed_leaf(q_leaves, member, values, error):
    indexed,manifest,paths = q_leaves
    np.save(paths[0]/f'{member}.npy',values)
    with pytest.raises(ValueError,match=error):
        verify_parallel_construction(manifest,indexed,paths)
