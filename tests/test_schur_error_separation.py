"""The inverse/approximation diagnostic must distinguish opposite errors."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'work/boundary_load_audit'))
from schur_error_separation import error_split

def test_exact_surrogate_inverse_does_not_hide_wrong_operator():
    h=np.diag([2.,4.]); s=np.diag([1.,4.]); b=np.array([1.,2.])
    z=np.linalg.solve(s,b)
    out=error_split(b,z,s@z,h@z)
    assert out['inverse_relative']==0
    assert np.isclose(out['approximation_relative'],out['true_H_relative'])
    assert out['decomposition_relative_error']<1e-15

def test_exact_operator_with_poor_inverse_is_identified():
    h=np.diag([2.,4.]); b=np.array([1.,2.]); z=.1*b
    out=error_split(b,z,h@z,h@z)
    assert out['approximation_relative']==0
    assert out['inverse_relative']>0.5
    assert out['decomposition_relative_error']<1e-15
