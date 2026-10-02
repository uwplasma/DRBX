"""Portable operational gate; no canonical inputs or GPU required."""
import json,os,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE.parents[1]))
from scripts.q08_extraction_global import campaign as c

class CampaignTests(unittest.TestCase):
    def test_hash_ignores_filesystem_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x';p.write_bytes(b'abc');before=p.stat();a=c.sha(p)
            p.write_bytes(b'abd');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertNotEqual(a,c.sha(p))
    def test_digest_dictionary_order(self):
        self.assertEqual(c.digest(dict(a=1,b=2)),c.digest(dict(b=2,a=1)))
    def test_json_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):c.write(Path(folder)/'x.json',{'x':float('nan')})
    def test_gate_receipt_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);c.write(p/'ok.json',dict(passed=True,campaign_identity='a'))
            self.assertTrue(c.require(p,'ok','a')['passed'])
            with self.assertRaises(ValueError):c.require(p,'ok','b')
    def test_checkpoint_rehash_and_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for name in ('bank.npz','geometry.npz'):(p/name).write_bytes(b'abc')
            c.write(p/'stats.json',dict(passed=True,campaign_identity='a',owners=[2,3],files={name:c.sha(p/name) for name in ('bank.npz','geometry.npz')}))
            self.assertTrue(c.chunk_valid(p,'a',[2,3]));self.assertFalse(c.chunk_valid(p,'b'))
            with self.assertRaises(ValueError):c.chunk_valid(p,'a',[3,2])
            (p/'bank.npz').write_bytes(b'abd')
            with self.assertRaises(ValueError):c.chunk_valid(p,'a',[2,3])
    def test_resource_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):c.check_resources(Path(folder),2,6,8)
            with self.assertRaises(ValueError):c.check_resources(Path(folder),0,6,12)
            for invalid in (float('nan'),float('inf'),-float('inf')):
                with self.subTest(worker_gib=invalid),self.assertRaises(ValueError):
                    c.check_resources(Path(folder),1,invalid,12)
                with self.subTest(host_gib=invalid),self.assertRaises(ValueError):
                    c.check_resources(Path(folder),1,6,invalid)
    def test_frozen_source(self):
        d=c.design();self.assertEqual(d['cases'],22);self.assertFalse(d['new_tracing']);self.assertEqual(d['devices'],[1,4])


    def test_extracted_and_canonical_inputs_rehashed_after_verification(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            here=Path(folder)/'source';run=Path(folder)/'run';root=Path(folder)/'canonical'
            here.mkdir();(run/'inputs').mkdir(parents=True);root.mkdir()
            extracted=run/'inputs/traces.npz';canonical=root/'geometry.npz'
            extracted.write_bytes(b'abc');canonical.write_bytes(b'xyz')
            c.write(here/'input_manifest.json',dict(files={'traces.npz':c.sha(extracted)},
                                                    canonical={'geometry.npz':c.sha(canonical)}))
            with patch.object(c,'HERE',here):
                c.check_inputs(run,root)
                stat=extracted.stat();extracted.write_bytes(b'abd')
                os.utime(extracted,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'frozen input changed'):
                    c.check_inputs(run,root)
                extracted.write_bytes(b'abc')
                stat=canonical.stat();canonical.write_bytes(b'xyw')
                os.utime(canonical,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'frozen input changed'):
                    c.check_inputs(run,root)

    def test_verify_cannot_replace_existing_run_identity_or_root(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);root=run/'canonical';design={'immutable':'a'}
            receipt=dict(passed=True,campaign_identity=c.digest(design),canonical_root=str(root))
            c.write(run/'verification.json',receipt)
            before=(run/'verification.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'identity/gate'):
                c.verify(run,root,{'immutable':'b'})
            with self.assertRaisesRegex(ValueError,'canonical root mismatch'):
                c.verify(run,run/'other-root',design)
            self.assertEqual(before,(run/'verification.json').read_bytes())

    def test_verify_refuses_tampered_extraction_instead_of_repairing(self):
        import io,tarfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            here=Path(folder)/'source';run=Path(folder)/'run';root=Path(folder)/'canonical'
            here.mkdir();(run/'inputs').mkdir(parents=True);root.mkdir()
            original=here/'original';original.write_bytes(b'abc')
            c.write(here/'input_manifest.json',dict(files={'traces.npz':c.sha(original)},canonical={}))
            with tarfile.open(here/'inputs.tar.gz','w:gz') as archive:
                member=tarfile.TarInfo('inputs/traces.npz');member.size=3
                archive.addfile(member,io.BytesIO(b'abc'))
            extracted=run/'inputs/traces.npz';extracted.write_bytes(b'abd')
            design={'immutable':'a'}
            c.write(run/'verification.json',dict(passed=True,campaign_identity=c.digest(design),canonical_root=str(root)))
            with patch.object(c,'HERE',here):
                with self.assertRaisesRegex(ValueError,'existing extracted input changed'):
                    c.verify(run,root,design)
            self.assertEqual(extracted.read_bytes(),b'abd')

    def test_cpu_completion_requires_all_cases_leaves_and_finite_metrics(self):
        import copy,numpy as np
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'inputs').mkdir();data=run/'data/N1';data.mkdir(parents=True)
            c.write(run/'inputs/plan.json',{'1':dict(chunks=[[0]],n_owner=1,n_raw=1)})
            np.save(data/'raw_to_owner.npy',np.array([0],dtype=np.int64))
            c.write(run/'data_N1.json',dict(passed=True,campaign_identity='a',files={'raw_to_owner.npy':c.sha(data/'raw_to_owner.npy')}))
            chunk=run/'cpu/N1/chunk_000000';chunk.mkdir(parents=True)
            for name in ('bank.npz','geometry.npz'):(chunk/name).write_bytes(b'abc')
            fields={'centered','correction','diffusion','combined'}
            names=fields|{'raw_current.'+name for name in ('divergence_homogeneous','divergence_lift','divergence_physical','vorticity_homogeneous','vorticity_lift','vorticity_current','electron_phi','electron_ti_compensation','electron_generalized_force','inputs_valid')}|{'raw_electron_material','inputs_valid','eigensystem_admissible'}
            metrics={name:dict(shape=[22,1,6] if name in fields else [22,1],
                dtype='bool' if name.endswith('inputs_valid') or name=='eigensystem_admissible' else 'float64',
                max_abs_error=0.,max_scaled_error=0.) for name in names}
            good=dict(passed=True,campaign_identity='a',owners=[0],raw=[0],
                files={name:c.sha(chunk/name) for name in ('bank.npz','geometry.npz')},
                components=[dict(span=span,kind=kind,leaves=copy.deepcopy(metrics)) for span in (1/16,1/32) for kind in range(4)],
                max_scaled=0.,all_arrays_bitwise=True)
            c.write(chunk/'stats.json',good)
            self.assertTrue(c.validate_cpu(run,1,'a')['passed'])
            for defect in ('missing_case','missing_leaf','short_state','nonfinite'):
                bad=copy.deepcopy(good)
                if defect=='missing_case':bad['components'].pop()
                elif defect=='missing_leaf':bad['components'][0]['leaves'].pop('combined')
                elif defect=='short_state':bad['components'][0]['leaves']['combined']['shape']=[21,1,6]
                else:bad['components'][0]['leaves']['combined']['max_scaled_error']=float('nan')
                # Deliberate malformed JSON bypasses the normal finite writer.
                (chunk/'stats.json').write_text(json.dumps(bad))
                with self.subTest(defect=defect),self.assertRaises(ValueError):
                    c.validate_cpu(run,1,'a')

if __name__=='__main__':unittest.main()
