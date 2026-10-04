"""Focused diagnostic and resource admission regression tests."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.q08_rhs_mms_global.resources import host_guard,GIB
from scripts.q08_rhs_mms_global.ti_replay import (reduce_scalar,references,checked_path,analyze,
    check_boundary_fixture,BOUNDARY_EPS_MULTIPLIER,SCI_ID,LAST)
from scripts.q08_rhs_mms_global.campaign import write,sha,read


class CloseoutTests(unittest.TestCase):
    def test_boundary_fixture_accepts_reported_remote_roundoff_and_records_it(self):
        expected=np.zeros((1,46,35))
        actual=expected.copy()
        expected[0,44:46,30]=0.02529665862220193
        actual[0,44:46,30]=0.025296658622201915
        expected[0,44:46,31]=0.1180991363900236
        actual[0,44:46,31]=0.11809913639002359
        result=check_boundary_fixture(actual,expected,np.array([44,45]))
        self.assertEqual(result['rounded_entries'],4)
        self.assertEqual(result['max_abs'],1.3877787807814457e-17)
        self.assertLess(result['max_budget_fraction'],1.)
        self.assertEqual(check_boundary_fixture(expected,expected,np.array([44,45]))['max_abs'],0.)

    def test_boundary_fixture_rejects_meaningful_errors_and_padding_changes(self):
        expected=np.zeros((1,3,2));expected[:,1]=[.025,.118]
        wall=np.array([1])
        for changed in (expected+np.array([[[0.,0.],[1e-10,0.],[0.,0.]]]),
                        -expected,expected[:,:,::-1]):
            with self.assertRaises(ValueError):check_boundary_fixture(changed,expected,wall)
        changed=expected.copy();changed[0,0,0]=np.finfo(float).eps
        with self.assertRaises(AssertionError):check_boundary_fixture(changed,expected,wall)

    def test_boundary_fixture_rejects_nonfinite_layout_and_precision(self):
        expected=np.zeros((1,3,2));wall=np.array([1])
        for bad in (expected.astype(np.float32),expected[:,:,:1],expected[0],
                    np.full_like(expected,np.nan),np.full_like(expected,np.inf)):
            with self.assertRaises(ValueError):check_boundary_fixture(bad,expected,wall)
        for bad_wall in (np.array([1,1]),np.array([-1]),np.array([3]),np.array([1.])):
            with self.assertRaises(ValueError):check_boundary_fixture(expected,expected,bad_wall)

    def test_boundary_roundoff_policy_matches_frozen_manifest(self):
        manifest=read(Path(__file__).resolve().parents[1]/'manifest.json')
        self.assertEqual(manifest['ti_boundary_fixture_roundoff'],dict(
            eps_multiplier=BOUNDARY_EPS_MULTIPLIER,scale='max(1,abs(expected))',
            dtype='float64',nonwall_exact=True))

    def test_guard_accepts_calibrated_budget_and_rejects_old_budget(self):
        e={'estimated_host_peak_upper_bytes':35*GIB}
        with self.assertRaises(MemoryError):host_guard(e,50)
        self.assertEqual(host_guard(e,113)['required_bytes'],113*GIB)
        with self.assertRaises(MemoryError):host_guard(e,112.9)
        e['observed_host_available_bytes']=100*GIB
        with self.assertRaises(MemoryError):host_guard(e,224)

    def test_guard_rejects_invalid_numbers(self):
        for x in (float('nan'),float('inf'),-1,0):
            with self.assertRaises(ValueError):host_guard({'estimated_host_peak_upper_bytes':1},x)
            with self.assertRaises(ValueError):host_guard({'estimated_host_peak_upper_bytes':x},224)

    def test_scalar_reduction_preserves_weighted_signed_maximum_statistics(self):
        radial=np.array([0,1,10,11,12,20,30,31]);no=len(radial)
        N=np.broadcast_to(np.arange(no)[None,None,:]+1.,(2,22,no))
        O=np.ones((22,no));R=O*.5;w=np.arange(no)+1.
        ref=dict(O=O,R=R,owners=np.arange(no),volume=w,radial=radial)
        s=reduce_scalar(N,ref,32)
        np.testing.assert_array_equal(s['sum2'][0,:,0,0],np.sum(w*np.arange(no)**2))
        np.testing.assert_array_equal(s['signed'][0,:,0,2],np.sum(w*(N[0,0]-.5)))
        np.testing.assert_array_equal(s['max_owner'][0,:,0,0],no-1)
        self.assertEqual(s['count'][0],no)
        bad=N.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):reduce_scalar(bad,ref,32)

    def test_reference_sign_coverage_and_tamper_checks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'references'/'N32';folder.mkdir(parents=True)
            path=folder/'chunk_000000.npz';O=np.zeros((2,22,2,31));R=np.zeros((22,2,31))
            O[...,27]=3;O[...,29]=5;O[...,30]=8
            R[...,27]=2;R[...,29]=4;R[...,30]=6
            np.savez(path,O=O,R=R,owners=np.arange(2),volume=np.array([1.,2.]),radial=np.array([0,1]))
            write(root/'references_N32.json',dict(identity=SCI_ID,passed=True,owners=2,chunks=1,
                  files={str(path.relative_to(root)):sha(path)}))
            with patch('scripts.q08_rhs_mms_global.ti_replay.COUNTS',{32:2}):
                result,receipt=references(root,32)
                np.testing.assert_array_equal(result['O'],-5.)
                np.testing.assert_array_equal(result['R'],-4.)
                self.assertTrue(receipt['diffusion_span_independent'])
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):references(root,32)

    def test_input_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):checked_path(Path(d),'../other')

    def test_completion_independently_reduces_actions_and_keeps_zero_order_undefined(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n,last in LAST.items():
                radial=np.array([0,1,last-1,last,last+1,last+3,n-2,n-1])
                R=np.ones((22,8));R[0]=0;O=R.copy()
                N=np.stack([R+1/n**2,R+2/n**2]);N[:,0]=0
                ref=dict(O=O,R=R,owners=np.arange(8),volume=np.arange(8)+1.,radial=radial)
                stats=reduce_scalar(N,ref,n);path=root/f'N{n}.npz'
                np.savez(path,N=N,**ref,**{'stat_'+k:v for k,v in stats.items()})
                write(root/f'N{n}.json',dict(identity='unit-test',passed=True,backend='gpu',sha256=sha(path)))
            with patch('scripts.q08_rhs_mms_global.ti_replay.COUNTS',{32:8,48:8,64:8}):
                analyze(root,'unit-test')
                a=read(root/'analysis.json')
                self.assertIsNone(a['orders'][0][0][0][0][0])
                self.assertAlmostEqual(a['orders'][0][0][0][1][0],2)
                self.assertTrue(read(root/'completion.json')['passed'])
                (root/'N32.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):analyze(root,'unit-test')


if __name__=='__main__':unittest.main()
