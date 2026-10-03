"""Portable harness tests; no scientific work or GPU emulation in this suite."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parents[1]))
from scripts.q08_rhs_mms_global.campaign import load,sha,write,read,require,check_source,bind
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--baseline-source',type=Path,default=HERE.with_name('q08_extraction_global'))
p.add_argument('--run',type=Path,default=Path(tempfile.gettempdir())/'q08-mms-portable-tests')
a, remaining=p.parse_known_args()
load(a.baseline_source,a.run)
from scripts.q08_rhs_mms_global.science import reduce_case,pack,TERMS,SPANS
from scripts.q08_rhs_mms_global.references import valid
from scripts.q08_rhs_mms_global.gpu import record_valid


class CampaignTests(unittest.TestCase):
    def test_cold_controller_and_spawn_ignore_bare_campaign_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, str(HERE/'verification/import_probe.py'),
                '--baseline-source', str(a.baseline_source.resolve()), '--run', folder],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"spawned_worker": true', result.stdout)

    def test_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x';p.write_bytes(b'abc');s=p.stat();h=sha(p)
            p.write_bytes(b'abd');os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))
            self.assertNotEqual(h,sha(p))

    def test_checkpoint_identity_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x.npz';p.write_bytes(b'abc')
            write(p.with_suffix('.json'),dict(identity='a',input_receipt='b',passed=True,sha256=sha(p)))
            self.assertTrue(valid(p,'a','b'))
            with self.assertRaises(ValueError):valid(p,'a','wrong-input')
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):valid(p,'a','b')

    def test_weighted_norms_not_unweighted(self):
        N=np.zeros((3,len(TERMS)));N[:,0]=[1,2,4]
        O=N*.5;R=N*.25
        s=reduce_case(N,O,R,np.array([8,9,10]),np.array([1.,2.,3.]),np.array([0,1,30]),32,10)
        self.assertEqual(s['count'][0],3)
        self.assertEqual(s['volume'][0],6)
        self.assertEqual(s['sum2'][0,0,0],.25+2+12)
        self.assertEqual(s['signed'][0,0,0],.5+2+6)
        self.assertEqual(s['maximum'][0,0,0],2)
        self.assertEqual(s['max_owner'][0,0,0],10)
        self.assertEqual(s['count'][7],1)
        self.assertEqual(s['count'][8],0)

    def test_exact_N_O_R_decomposition(self):
        rng=np.random.default_rng(72)
        N,O,R=rng.normal(size=(3,7,len(TERMS)))
        s=reduce_case(N,O,R,np.arange(7),np.ones(7),np.arange(7),32,10)
        for i,e in enumerate((N-O,O-R,N-R)):
            np.testing.assert_allclose(s['sum2'][0,i],np.sum(e*e,axis=0),rtol=1e-15)

    def test_term_packing_and_source(self):
        centered=np.arange(12.).reshape(2,6);correction=centered*.1;diffusion=centered*.2
        z=np.zeros(2);v=pack(centered,correction,diffusion,z,z,z,z,z,z,z)
        self.assertEqual(v.shape,(2,31))
        np.testing.assert_array_equal(v[:,18:24],centered+correction+diffusion)
        self.assertEqual(TERMS[23],'combined_omega')
        self.assertEqual(TERMS[24],'current')
        self.assertEqual(TERMS[-1],'electron_generalized_force')

    def test_frozen_contract(self):
        _,m=check_source()
        self.assertTrue(m['new_references']);self.assertFalse(m['new_tracing'])
        self.assertFalse(m['new_preparation']);self.assertEqual(m['actual_gpu_devices'],4)
        self.assertEqual(m['outputs'],31);self.assertEqual(m['diffusion_spans'],list(SPANS))

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):write(Path(folder)/'x',dict(v=float('nan')))
        z=np.zeros((1,len(TERMS)));z[0,0]=np.nan
        with self.assertRaises(ValueError):reduce_case(z,z,z,np.array([0]),np.ones(1),np.zeros(1),32,10)

    def test_output_paths_cannot_alias_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            args=argparse.Namespace(run=p/'old'/'new',baseline_run=p/'old',
                baseline_source=p/'source',implementation_run=p/'impl')
            with self.assertRaises(ValueError):bind(args,'identity')

    def test_stage_gate_rehash(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'data').write_bytes(b'abc')
            write(p/'stage.json',dict(passed=True,identity='a',files={'data':sha(p/'data')}))
            self.assertTrue(require(p,'stage','a')['passed'])
            (p/'data').write_bytes(b'abd')
            with self.assertRaises(ValueError):require(p,'stage','a')

    def test_science_record_not_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'record.json';p.with_suffix('.npz').write_bytes(b'abc')
            write(p,dict(identity='a',n=32,span=1/32,kind=2,case=5,passed=True,sha256=sha(p.with_suffix('.npz'))))
            self.assertTrue(record_valid(p,'a',32,1/32,2,5))
            with self.assertRaises(ValueError):record_valid(p,'a',32,1/32,2,6)

    def test_report_recovers_order_two_and_marks_zero_reference(self):
        from scripts.q08_rhs_mms_global import analyze as module
        shape=(3,2,4,22,9,3,len(TERMS))
        rms=np.broadcast_to((1/np.array([32.,48.,64.])**2).reshape(3,1,1,1,1,1,1),shape)
        volume=np.ones(shape[:-2])
        data=dict(sum2=rms*rms,volume=volume,count=np.ones_like(volume,dtype=int),
            reference_sum2=np.zeros(shape[:-2]+(len(TERMS),)),maximum=rms,
            max_owner=np.zeros(shape,dtype=int),signed=np.zeros(shape))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for n in (32,48,64):
                write(root/f'references_N{n}.json',dict(reference_step_max_by_term=[0]*len(TERMS)))
            with patch.object(module,'collect',return_value=([],data)):
                _,arrays,table,report=module.products(root,'test')
            np.testing.assert_allclose(arrays['orders'],2.,atol=2e-15)
            self.assertTrue(np.isnan(arrays['relative_rms']).all())
            self.assertEqual(len(table.splitlines()),1+2*4*22*9*3*len(TERMS))
            self.assertIn('combined_omega',report)
            self.assertIn('not scientific acceptance',report)


if __name__=='__main__':unittest.main(argv=[sys.argv[0],*remaining])
