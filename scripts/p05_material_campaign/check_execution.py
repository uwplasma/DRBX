"""Bounded actual-HSX serial/parallel/restart equivalence; not a scaling study."""
import argparse
import multiprocessing
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from p05_material_campaign import campaign as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True);p.add_argument('--centered-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.input_root=a.input_root.resolve();a.centered_root=a.centered_root.resolve()
    if a.workers<1:p.error('positive --workers required')
    with c.runner._exclusive_output(a.output/'controller'):
        identity=c.verify(a);_,path,_=c.runtime(a);n=48
        fixture=c.arrays(c.HERE/'preflight_N48.npz')
        # Spread across the complete bounded sample; include seam/wall cases.
        indices=fixture['indices'][np.linspace(0,len(fixture['indices'])-1,48,dtype=int)]
        config=c.settings(a,n,identity,path);c.initialize(config)
        serial=a.output/'execution_check/serial.npz';c.compute((indices,serial))
        jobs=[(ids,a.output/f'execution_check/parallel_{j}.npz') for j,ids in enumerate(np.array_split(indices,4))]
        with multiprocessing.get_context('spawn').Pool(a.workers,initializer=c.initialize,
                initargs=(config,),maxtasksperchild=2) as pool:
            list(pool.imap_unordered(c.compute,jobs,chunksize=1))
        expected=c.arrays(serial);actual=np.concatenate([c.arrays(p)['delta_flux'] for _,p in jobs],axis=1)
        # Different metric-query batch lengths may change floating reduction
        # order. Preserve scientific equivalence, not bitwise arithmetic.
        np.testing.assert_allclose(actual,expected['delta_flux'],rtol=1e-10,atol=1e-20)
        stamps=[p.stat().st_mtime_ns for _,p in jobs]
        for job in jobs:c.compute(job)
        assert stamps==[p.stat().st_mtime_ns for _,p in jobs]
        # Explicit missing/corrupt/wrong-identity receipt checks on a scratch copy.
        import shutil
        bad=a.output/'execution_check/corrupt.npz';shutil.copyfile(serial,bad)
        shutil.copyfile(serial.with_suffix('.json'),bad.with_suffix('.json'))
        with bad.open('ab') as stream:stream.write(b'corrupt')
        try:c.validate_chunk(bad,identity,indices)
        except RuntimeError:pass
        else:raise AssertionError('corruption accepted')
        try:c.validate_chunk(serial,'different',indices)
        except RuntimeError:pass
        else:raise AssertionError('different identity accepted')
        bad.unlink();bad.with_suffix('.json').unlink()
        c.write(a.output/'execution_check.json',{'passed':True,'identity':identity,'N':n,'faces':len(indices),
            'serial_parallel_delta_max_abs':float(np.max(abs(actual-expected['delta_flux']))),
            'resume_preserved_checkpoints':True,'corrupt_and_incompatible_rejected':True})
        print('bounded parallel/restart check passed',flush=True)


if __name__=='__main__':main()
