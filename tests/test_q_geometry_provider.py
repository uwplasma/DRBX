import numpy as np
import pytest

from drbx.stencils.q_geometry import QGeometryProvider


def contract():
    return dict(magnetic_evaluator='compact_c3', coordinate_map='qualified logical map',
                normalization={'length': 1.0}, validity='callback rejects unsupported points',
                batching='original six-slot Q preparation batches',
                input_hashes={'magnetic_input': 'a'*64}, source_hashes={'evaluator.py': 'b'*64})


def test_identity_and_callback_batches_are_preserved():
    calls=[]; points=np.zeros((7,6,3)); result=object(); provenance=contract()
    def geom(p):
        calls.append(p)
        return result
    provider=QGeometryProvider(geom,geom,provenance)
    assert provider.geom(points) is result and provider.jacobian(points) is result
    assert calls==[points,points]
    original=provider.identity
    provenance['normalization']['length']=2
    provider.provenance['normalization']['length']=3
    assert provider.identity==original
    assert provider.identity_record()['geometry_provider']['normalization']['length']==1
    changed=contract();changed['magnetic_evaluator']='spline'
    assert QGeometryProvider(geom,geom,changed).identity!=original


def test_incomplete_provenance_and_callback_errors():
    def unsupported(_): raise ValueError('unsupported geometry range')
    with pytest.raises(ValueError,match='incomplete'):
        QGeometryProvider(unsupported,unsupported,{})
    invalid=contract();invalid['input_hashes']={'field':'mtime-only'}
    with pytest.raises(ValueError,match='SHA256'):
        QGeometryProvider(unsupported,unsupported,invalid)
    with pytest.raises(ValueError,match='unsupported geometry range'):
        QGeometryProvider(unsupported,unsupported,contract()).geom(np.zeros((1,3)))
